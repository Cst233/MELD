import argparse
import concurrent.futures
import json
import os
import pickle
import time
from pathlib import Path
from string import Template

import numpy as np
import torch
import transformers
from datasets import load_from_disk
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams

from data_util.dataset_process import TARGET_2_LABEL, LABEL_2_TARGET, extract_all_first_choices
from data_util.template import template_0shot, template_0shot_gtl

# --- User Configuration ---
MODEL_PATHS = {
    "Qwen2.5-7B-Instruct": "PATH_TO_MODELS/Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct": "PATH_TO_MODELS/Llama-3.1-8B-Instruct",
    "Qwen2.5-14B-Instruct": "PATH_TO_MODELS/Qwen2.5-14B-instruct/",
    "Qwen3-14B": "PATH_TO_MODELS/Qwen3-14B",
    "Qwen3-8B": "PATH_TO_MODELS/Qwen3-8B",
    "Ministral-8B-it": "PATH_TO_MODELS/Ministral-8B/",
    'gemma-2-9b-it': "PATH_TO_MODELS/gemma-2-9b-it/",
    "gemma-3-12b-it": "PATH_TO_MODELS/gemma-3-12b-it",
    "phi4": "PATH_TO_MODELS/phi-4",
}

client = OpenAI(
    api_key="<YOUR_API_KEY>",
)

IS_VLLM = True 
DATASET_SERIALIZED_DIR = Path("datasets_json_serialized")
OUTPUT_DIR = Path('pkls/5bin_0_shot_prompt')

# ---------------------------------------------------------

def load_local_model(model_name: str):
    """Loads model and tokenizer for local inference."""
    trust_remote = True
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Model {model_name} path not found in configuration.")
    
    path = Path(MODEL_PATHS[model_name])
    
    if not IS_VLLM:
        model = AutoModelForCausalLM.from_pretrained(
            path, 
            torch_dtype="auto", 
            device_map="auto", 
            trust_remote_code=trust_remote
        )
        model.eval()
    else:
        # Set specific GPU if needed
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        model = LLM(
            model=str(path),
            dtype=torch.bfloat16,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            max_num_batched_tokens=4096,
            max_model_len=512
        )
    
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left", trust_remote_code=trust_remote)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def create_templated_dataset(serial_dataset, dataset_name, model_name):
    """Applies prompts templates to data."""
    template_str = template_0shot if model_name != 'GTL' else template_0shot_gtl
    template = Template(template_str)
    
    if model_name == 'GTL':
        def apply_template(ex):
            # Dynamic import of variables for GTL prompts if needed, or assume they are available
            # role_prompt = eval(f'{dataset_name}_role_prompt')
            # answer_prompt = eval(f'{dataset_name}_answer_prompt')
            return {
                'prompt': template.substitute(
                    task_prompt=ex['task_description'],
                    role_prompt=f"Role prompt placeholder for {dataset_name}", # Adapted to prevent eval() errors if vars missing
                    note=ex['note'],
                    answer_prompt=f"Answer prompt placeholder for {dataset_name}",
                ),
                'target': LABEL_2_TARGET[dataset_name][ex['label']],
            }
    else:
        def apply_template(ex):
            label_options = " | ".join(TARGET_2_LABEL[dataset_name].keys())
            return {
                'prompt': template.substitute(
                    task_description=ex['task_description'],
                    features=ex['features'],
                    note=ex['note'],
                    label_str=label_options,
                    n=len(TARGET_2_LABEL[dataset_name])
                ),
                'target': LABEL_2_TARGET[dataset_name][ex['label']],
            }

    return serial_dataset.map(apply_template)

def load_data(dataset_name, model_name):
    """Loads simplified dataset from disk."""
    serial_dataset = load_from_disk(DATASET_SERIALIZED_DIR / dataset_name)
    return create_templated_dataset(serial_dataset, dataset_name, model_name)

def call_openai_api(model_name, sys_content, user_content):
    """Safe API call wrapper."""
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content}
            ],
            logprobs=False,
            stream=False,
            max_tokens=10,
            n=1
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API error: {e}")
        return "None"

@torch.no_grad()
def run_batch_inference(prompts, model, tokenizer, dataset_name, is_api, model_name):
    """Executes inference on a batch of prompts."""
    responses = []

    if not is_api:
        if not IS_VLLM:
            # HuggingFace Transformers inference
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
                add_special_tokens=False
            ).to(model.device)
            
            gen_kwargs = {
                "pad_token_id": tokenizer.eos_token_id,
                "max_new_tokens": 128,
                "use_cache": True
            }

            if model_name in ['Qwen3-14B', 'Qwen3-8B']:
                gen_kwargs.update({
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0
                })
            
            outputs = model.generate(**inputs, **gen_kwargs)
            
            responses = [
                tokenizer.decode(out[len(inp):], skip_special_tokens=True)
                for out, inp in zip(outputs, inputs.input_ids)
            ]
        else:
            # vLLM inference
            sampling_params = SamplingParams(
                max_tokens=10,
                temperature=0,
                top_p=0.9
            )
            outputs = model.generate(prompts, sampling_params=sampling_params)
            responses = [output.outputs[0].text for output in outputs]
    else:
        # API inference
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(prompts))) as executor:
            future_to_idx = {}
            for idx, (sys_c, user_c) in enumerate(prompts):
                future = executor.submit(call_openai_api, model_name, sys_c, user_c)
                future_to_idx[future] = idx
            
            responses = [None] * len(prompts)
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    print(f"Error in API call (index {idx}): {e}")
                    responses[idx] = "None"
        time.sleep(0.1)

    if model_name != 'GTL':
        return extract_all_first_choices(dataset_name, responses)
    else:
        # GTL specific response handling (assumes integer output)
        return [LABEL_2_TARGET[dataset_name][int(r.strip())] for r in responses]

def save_predictions(all_prompts, true_labels, model, tokenizer, batch_size, dataset_name, is_api, model_name):
    """Batches inference and saves results to disk."""
    all_predictions = []
    
    for i in range(0, len(all_prompts), batch_size):
        batch = all_prompts[i : i + batch_size]
        preds = run_batch_inference(batch, model, tokenizer, dataset_name, is_api, model_name)
        all_predictions.extend(preds)
    
    data_to_save = {
        'all_results': all_predictions,
        'prompts': all_prompts,
        'true_labels': true_labels
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = OUTPUT_DIR / f'results_{dataset_name}_{model_name}.pkl'
    
    with open(save_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"Saved predictions to {save_path}")
    return all_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--datafile", type=str, default="heart", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    
    args = parser.parse_args()

    is_api_model = (args.model in ['gpt-4o-mini'])

    model_engine, model_tokenizer = None, None
    if not is_api_model:
        model_engine, model_tokenizer = load_local_model(args.model)
    
    dataset = load_data(args.datafile, args.model)

    formatted_prompts = []
    ground_truth_labels = []

    # Format Prompts
    if not is_api_model:
        for ex in dataset:
            ground_truth_labels.append(ex["target"])
            
            if args.model == 'GTL':
                formatted_prompts.append(ex['prompt'])
                continue

            sys_content, user_content = ex["prompt"].split("\n\n", 1)
            
            # Message formatting
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content}
            ]
            
            if args.model == 'gemma-2-9b-it':
                 # Gemma specific: merge into one user message or handle differently
                 messages = [{"role": "user", "content": ex['prompt']}]

            
            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True
            }

            # Qwen 3 specific flag
            if args.model in ['Qwen3-14B', 'Qwen3-8B']:
                chat_template_kwargs["enable_thinking"] = False

            formatted_text = model_tokenizer.apply_chat_template(messages, **chat_template_kwargs)
            formatted_prompts.append(formatted_text)
            
    else:
        # For API, keep system/user separate
        for ex in dataset:
            ground_truth_labels.append(ex["target"])
            sys_content, user_content = ex["prompt"].split("\n\n", 1)
            formatted_prompts.append((sys_content, user_content))

    # Convert label strings to integers
    ground_truth_ints = [TARGET_2_LABEL[args.datafile][t] for t in ground_truth_labels]
    
    save_predictions(
        formatted_prompts, 
        ground_truth_ints, 
        model_engine, 
        model_tokenizer,
        args.batch_size, 
        args.datafile, 
        is_api_model, 
        args.model
    )