import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from snorkel.labeling.model import LabelModel
from string import Template
from tqdm import tqdm

# Custom imports
# Ensure these modules exist in your data_util package
from data_util.create_external_datasets import load_train_validation_test
from data_util.evaluate_external_dataset import prepare_data, read_orig_dataset
from data_util.dataset_process import LABEL_2_TARGET, TARGET_2_LABEL, target_2_label
from data_util.template import template_0shot

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment for Weak-to-Strong Generalization with EM/DS voting.")
    
    # Experiment settings
    parser.add_argument("--method", type=str, default="ours", help="Method to run: 'ours', '0shot', 'snorkel'")
    parser.add_argument("--model", type=str, default="Qwen3-14B", help="Main model name (used for single mode or 0-shot)")
    parser.add_argument("--datafile", type=str, default="car", help="Dataset name")
    parser.add_argument("--bin", type=int, default=5, help="Number of bins for discretization")
    
    # Student model settings
    parser.add_argument("--ml_model", type=str, default="catboost", help="Student model type: 'catboost', 'lr'")

    # Distillation / Voting settings
    parser.add_argument(
        "--distill_mode", 
        type=str, 
        default="ensemble", 
        choices=["ensemble", "single"],
        help="Distillation mode: 'ensemble' (multiple LLMs) or 'single' (specified --model)"
    )
    parser.add_argument(
        "--voting",
        type=str,
        default="expert_simple", 
        choices=["f1_weighted", "acc_weighted", "simple", "expert_simple", "ds", "expert_ds"],
        help="Voting logic mechanism"
    )
    parser.add_argument(
        "--em_temperature",
        type=float,
        default=0.1,
        help="Temperature for EM algorithm softmax"
    )
    parser.add_argument(
        "--auto_temp",
        action='store_true',
        help="Automatically tune EM temperature"
    )
    
    return parser.parse_args()

# Global Configuration
args = parse_args()
tqdm.pandas(bar_format="{l_bar}{bar:40}{r_bar}")

# Constants
DATASET_SERIALIZED_DIR = Path("datasets_json_serialized") 
DATA_DIR = Path("datasets") 
PKL_DIR = Path(f'pkls/{args.bin}bin_0_shot_prompt')

# Models used for ensemble
ENSEMBLE_MODELS = [
    "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Qwen2.5-14B-Instruct",
    "Qwen3-14B", "Qwen3-8B", "Ministral-8B-it", "gemma-2-9b-it",
    "gemma-3-12b-it", "phi4", "gpt-4o-mini"
]

# Experiment Variables
DATASET_NAME = args.datafile
SEEDS = [42, 123, 456, 10, 11]
N_LABEL = len(TARGET_2_LABEL[DATASET_NAME])

# ================= Data Loading Helpers =================

def create_template_dataset(serial_dataset, seed):
    """Applies templates to the serialized dataset."""
    split_ratio = [0.5, 0.1, 0.4] 
    splits = serial_dataset.train_test_split(test_size=split_ratio[1]+split_ratio[2], seed=seed)
    test_val = splits['test'].train_test_split(test_size=split_ratio[2]/(split_ratio[1]+split_ratio[2]), seed=seed)
    
    datasets = DatasetDict({
        'train': splits['train'],
        'valid': test_val['train'],
        'test': test_val['test']
    })

    # Combine train and valid for this experiment setup
    new_train = concatenate_datasets([datasets['train'], datasets['valid']])
    datasets['train'] = new_train

    template = Template(template_0shot)
    
    def apply_template(ex):
        return {
            'prompt': template.substitute(
                task_description=ex['task_description'],
                features=ex['features'],
                note=ex['note'],
                label_str=" | ".join(TARGET_2_LABEL[DATASET_NAME].keys()),
                n=len(TARGET_2_LABEL[DATASET_NAME])
            ),
            'target': LABEL_2_TARGET[DATASET_NAME][ex['label']],
        }

    datasets['train'] = datasets['train'].map(apply_template)
    return datasets

def preload_datasets(dataset_name, seeds):
    """Loads datasets and cached model predictions."""
    datasets = {}
    serial_dataset = load_from_disk(DATASET_SERIALIZED_DIR / dataset_name)
    model_results = {}

    # Determine which models to load
    if args.distill_mode == 'single':
        models_to_load = [args.model]
    else:
        models_to_load = ENSEMBLE_MODELS
    
    # Ensure the main model is loaded for 0shot evaluation
    if args.method == '0shot' and args.model not in models_to_load:
        models_to_load.append(args.model)

    for model_name in models_to_load:
        pkl_path = PKL_DIR / f'results_{dataset_name}_{model_name}.pkl'
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            model_results[model_name] = Dataset.from_dict({
                'preds': [res for res in data['all_results']]
            })
        except FileNotFoundError:
            print(f"Warning: Results for {model_name} not found at {pkl_path}. Skipping.")
    
    for seed in seeds:
        seed_datasets = create_template_dataset(serial_dataset, seed)
        for split in ['train', 'valid', 'test']:
            key = f"{split}_{seed}"
            datasets[key] = seed_datasets[split]
            datasets[key].info.description = dataset_name
            
    return datasets, model_results

# Preload data once
all_datasets, model_results = preload_datasets(DATASET_NAME, SEEDS)

def get_subset(ds, seed, subset_type='train'):
    """Splits dataset to get the specific train/test subset based on seed."""
    split_ratio = [0.5, 0.1, 0.4] 
    splits = ds.train_test_split(test_size=split_ratio[1]+split_ratio[2], seed=seed)
    test_val = splits['test'].train_test_split(test_size=split_ratio[2]/(split_ratio[1]+split_ratio[2]), seed=seed)
    
    datasets = DatasetDict({
        'train': splits['train'],
        'valid': test_val['train'],
        'test': test_val['test']
    })

    new_train = concatenate_datasets([datasets['train'], datasets['valid']])
    datasets['train'] = new_train
    
    if subset_type == 'train':
        return datasets['train']
    elif subset_type == 'test':
        return datasets['test']
    return None

def get_dataset_without_guidance(split, seed):
    """Prepares dataset for 0-shot evaluation (no chain-of-thought/guidance)."""
    serial_dataset = all_datasets[f"{split}_{seed}"]
    template = Template(template_0shot)
    
    def add_guidance(ex):
        return {
            'prompt': template.substitute(
                task_description=ex['task_description'],
                features=ex['features'],
                note=ex['note'],
                label_str=" | ".join(TARGET_2_LABEL[DATASET_NAME].keys())
            ),
            'target': LABEL_2_TARGET[DATASET_NAME][ex['label']],
        }
        
    serial_dataset = serial_dataset.map(add_guidance, remove_columns=['task_description', 'features', 'note', 'label'])
    return serial_dataset

def initialize_student_model(x_train, y_train, model_type, random_state=42):
    """Initializes and trains the student classifier."""
    if model_type == 'lr':
        model = LogisticRegression(class_weight='balanced', penalty='l2', random_state=random_state, max_iter=300)
        model.fit(x_train, y_train)
        return model
    elif model_type == 'catboost': 
        model = CatBoostClassifier(iterations=300, random_seed=random_state, verbose=0, auto_class_weights='Balanced')
        cat_features = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
        model.fit(x_train, y_train, cat_features=cat_features)
        return model
    raise ValueError(f"Unknown model type: {model_type}")

def prepare_data_matrices(seed, split, bin_count):
    """Loads and preprocesses data into feature matrices."""
    student_type = args.ml_model
    data_dir = DATA_DIR / DATASET_NAME
    
    # Load raw data
    dataset = load_train_validation_test(DATASET_NAME, data_dir, bin_count, quantify=False) 
    dataset = DatasetDict({k: Dataset.from_pandas(v, preserve_index=False) for k, v in dataset.items()})
    dataset = concatenate_datasets(list(dataset.values()))
    
    # Apply seed-specific splitting rules
    dataset = DatasetDict({k: read_orig_dataset(dataset, seed, k) for k in ['train', 'validation', 'test']})
    dataset = DatasetDict({k: v.to_pandas() for k, v in dataset.items()})
    
    # Encode features
    if student_type == 'catboost':
        dataset_orig = dataset
    else:
        enc_method = "ordinal" if student_type == 'lgb' else "one-hot"
        dataset_orig = prepare_data(DATASET_NAME, None, dataset, enc=enc_method, scale=True) 
    
    if split == 'train':
        dataset = Dataset.from_pandas(dataset_orig['train'], preserve_index=False)
        dataset = dataset.remove_columns(['idx'])
        x_train = dataset.remove_columns(['label']).to_pandas()
        
        if student_type == 'lgb':
            for col in x_train.columns:
                x_train[col] = x_train[col].astype('int32')
        
        # Deduplicate
        x_train_unique = x_train.drop_duplicates(keep='first', ignore_index=False)
        y_train_unique = dataset[x_train_unique.index.to_numpy()]['label']
        return x_train_unique, x_train.columns.to_list(), y_train_unique
        
    elif split == 'test':
        dataset_test = Dataset.from_pandas(dataset_orig['test'], preserve_index=False)
        x_test = dataset_test.remove_columns(['idx', 'label']).to_pandas()
        return x_test, x_test.columns.to_list()
    
    return None

# ================= Algorithms: EM & Dawid-Skene =================

def dawid_skene_algorithm(L, n_classes, max_iter=100, tol=1e-4):
    """
    Dawid-Skene algorithm implementation.
    Estimates the confusion matrix provided by each model and returns Expected Accuracy as weights.
    """
    N, K = L.shape
    if K == 1:
        return np.array([1.0])

    # Initialization: Majority Vote
    counts = np.zeros((N, n_classes))
    for i in range(N):
        for k in range(K):
            if L[i, k] != -1:
                counts[i, L[i, k]] += 1
    
    row_sums = counts.sum(axis=1, keepdims=True)
    T_z = np.divide(counts, row_sums, out=np.ones_like(counts)/n_classes, where=row_sums>0)
    class_priors = T_z.sum(axis=0) / N
    
    pi = np.zeros((K, n_classes, n_classes)) # Confusion matrices
    
    for _ in range(max_iter):
        prev_T_z = T_z.copy()
        
        # M-Step
        class_priors = T_z.sum(axis=0) / N
        for k in range(K):
            for j in range(n_classes):
                weights_j = T_z[:, j] 
                pred_k = L[:, k]
                for l in range(n_classes):
                    match_l = (pred_k == l).astype(float)
                    pi[k, j, l] = np.sum(match_l * weights_j)
            
            cm_row_sums = pi[k].sum(axis=1, keepdims=True)
            pi[k] = np.divide(pi[k], cm_row_sums, out=np.ones_like(pi[k])/n_classes, where=cm_row_sums>0)

        # E-Step
        new_T_z = np.zeros((N, n_classes))
        log_priors = np.log(class_priors + 1e-9)
        log_pi = np.log(pi + 1e-9)
        
        for i in range(N):
            log_prob = log_priors.copy()
            for k in range(K):
                pred = L[i, k]
                if pred != -1:
                    log_prob += log_pi[k, :, pred]
            new_T_z[i, :] = log_prob
            
        new_T_z = new_T_z - np.max(new_T_z, axis=1, keepdims=True)
        new_T_z = np.exp(new_T_z)
        row_sums_z = new_T_z.sum(axis=1, keepdims=True)
        T_z = np.divide(new_T_z, row_sums_z, out=np.zeros_like(new_T_z), where=row_sums_z>0)

        if np.max(np.abs(T_z - prev_T_z)) < tol:
            break
            
    # Calculate scalar accuracy weights
    weights = np.zeros(K)
    for k in range(K):
        acc = 0
        for j in range(n_classes):
            acc += class_priors[j] * pi[k, j, j]
        weights[k] = acc
        
    return weights

def em_algorithm(L, n_classes, max_iter=20, temperature=0.1, metric='f1', tol=1e-4):
    """
    EM Algorithm to optimize Latent Truth estimates based on Macro-F1 or Accuracy.
    """
    N, K = L.shape
    if K == 1:
        return np.array([1.0])

    weights = np.ones(K) / K
    
    for i in range(max_iter):
        prev_weights = weights.copy()

        # E-Step: Estimate latent truth probability q(z)
        q = np.zeros((N, n_classes))
        for c in range(n_classes):
            match_c = (L == c)
            q[:, c] = np.dot(match_c, weights)
        
        row_sums = q.sum(axis=1, keepdims=True)
        q = np.divide(q, row_sums, out=np.zeros_like(q), where=row_sums > 0)
        
        # M-Step: Update model weights based on q(z)
        pi = np.zeros(K)
        true_counts = q.sum(axis=0) 
        true_counts[true_counts == 0] = 1e-8

        for k in range(K):
            valid_mask = (L[:, k] != -1)
            if np.sum(valid_mask) == 0:
                pi[k] = 0
                continue
            
            metric_sum = 0
            valid_classes_count = 0
            
            for c in range(n_classes):
                indices = np.where((L[:, k] == c) & valid_mask)[0]
                tp_weight = np.sum(q[indices, c]) if len(indices) > 0 else 0
                
                score = 0
                if metric == 'f1':
                    pred_count = len(indices)
                    precision = tp_weight / pred_count if pred_count > 0 else 0
                    recall = tp_weight / true_counts[c]
                    if precision + recall > 0:
                        score = 2 * (precision * recall) / (precision + recall)
                elif metric == 'acc':
                    score = tp_weight / true_counts[c]
                
                if true_counts[c] > 1e-5 or (metric == 'f1' and pred_count > 0):
                    metric_sum += score
                    valid_classes_count += 1
            
            if valid_classes_count > 0:
                pi[k] = metric_sum / valid_classes_count
            else:
                pi[k] = 0

        # Softmax normalization with temperature
        scaled_pi = pi / temperature
        exp_pi = np.exp(scaled_pi - np.max(scaled_pi))
        weights = exp_pi / np.sum(exp_pi)

        if np.max(np.abs(weights - prev_weights)) < tol:
            break
        
    return weights

def calculate_entropy(weights):
    """Calculates Shannon entropy (base 2)."""
    w = weights[weights > 1e-9]
    return -np.sum(w * np.log2(w))

def auto_tune_em_temperature(L_matrix, n_label, metric='f1', max_iter=20):
    """
    Automatically finds the lowest temperature that maintains diversity (Entropy) 
    and avoids dictatorship.
    """
    temp_candidates = [0.5, 0.3, 0.2, 0.1, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01]
    
    best_weights = None
    best_temp = temp_candidates[0]
    n_models = L_matrix.shape[1]
    
    # Constraints
    if n_models == 2:
        min_effective_models = 1.0
    elif n_models == 3:
        min_effective_models = 1.99
    else:
        min_effective_models = max(3.0, n_models / 2.0)
    
    min_entropy = np.log2(min_effective_models)
    max_single_weight_threshold = 0.5 if n_models >= 3 else 1.0
    
    print(f"  Auto-tuning Temperature (N={n_models}). Target: ENM >= {min_effective_models:.2f}, MaxIndividualWeight <= {max_single_weight_threshold}")
    
    for temp in temp_candidates:
        weights = em_algorithm(L_matrix, n_label, max_iter=max_iter, temperature=temp, metric=metric)
        
        current_entropy = calculate_entropy(weights)
        effective_num = 2 ** current_entropy
        max_w = np.max(weights)
        
        entropy_violation = current_entropy < min_entropy
        dictator_violation = (n_models >= 3) and (max_w > max_single_weight_threshold)

        if entropy_violation or dictator_violation:
            if best_weights is not None:
                reason = "Low Entropy" if entropy_violation else "Dictatorship"
                print(f"    -> Stop at T={temp} ({reason}: ENM={effective_num:.2f}, MaxW={max_w:.2f}). Reverting to T={best_temp}")
                return best_weights, best_temp
            else:
                print(f"    -> Warning: Even highest temp {temp} violates constraints.")
                return weights, temp
        
        best_weights = weights
        best_temp = temp
    
    print(f"    -> Reached lowest temp T={best_temp} while maintaining diversity.")
    return best_weights, best_temp

# ================= Main Process Logic =================

def run_distillation_process(method, seed):
    """
    Main controller for the distillation process.
    Iterates through teacher models, creates student models, and weights them via EM/DS.
    """
    if method == "snorkel":
        # === Snorkel Baseline ===
        student_type = args.ml_model
        x_train, _, _ = prepare_data_matrices(seed, 'train', args.bin)
        
        if student_type == 'catboost':
            x_train_input = x_train
        else:
            x_train_input = x_train.to_numpy()

        teacher_models = ENSEMBLE_MODELS if args.distill_mode == 'ensemble' else [args.model]
        teacher_preds_collection = []

        # 1. Collect teacher predictions
        for model_name in teacher_models:
            if model_name not in model_results:
                teacher_preds_collection.append(np.full(len(x_train), -1))
                continue
            
            # Align indices
            train_results = get_subset(model_results[model_name], seed, 'train').select(x_train.index.to_numpy())
            preds = []
            for pred in train_results['preds']:
                if pred in TARGET_2_LABEL[DATASET_NAME]:
                    preds.append(TARGET_2_LABEL[DATASET_NAME][pred])
                else:
                    preds.append(-1) # Abstain
            teacher_preds_collection.append(preds)
        
        L_matrix = np.array(teacher_preds_collection).T 
        
        # 2. Run Snorkel LabelModel
        print(f"Seed {seed} - [Snorkel] fitting LabelModel on shape {L_matrix.shape}...")
        label_model = LabelModel(cardinality=N_LABEL, verbose=False)
        label_model.fit(L_train=L_matrix, n_epochs=500, log_freq=100, seed=seed)
        
        # 3. Get Aggregated Labels
        snorkel_preds = label_model.predict(L=L_matrix)
        
        # 4. Train Single Student Model
        valid_mask = (snorkel_preds != -1)
        if np.sum(valid_mask) == 0:
            print(f"Seed {seed} - [Snorkel] Failed to generate any valid labels. Aborting.")
            return [None], np.array([0.0])
            
        x_train_filtered = x_train.iloc[valid_mask] if isinstance(x_train, pd.DataFrame) else x_train_input[valid_mask]
        y_train_filtered = snorkel_preds[valid_mask]
        
        print(f"Seed {seed} - [Snorkel] Training {student_type} on {len(y_train_filtered)} aggregated samples.")
        student_model = initialize_student_model(x_train_filtered, y_train_filtered, student_type, random_state=seed)
        student_model.is_binary_subclass = False 
        
        return [student_model], np.array([1.0])

    # === Our Method ===
    ml_model_list = []
    model_weights = 0
    valid_model_indices = []

    if method == "ours":
        student_type = args.ml_model
        x_train, _, _ = prepare_data_matrices(seed, 'train', args.bin)
        
        if student_type == 'catboost':
            x_train_input = x_train
        else:
            x_train_input = x_train.to_numpy()

        teacher_preds_collection = []

        if args.distill_mode == 'single':
            teacher_models = [args.model]
        else:
            teacher_models = ENSEMBLE_MODELS

        for idx, model_name in enumerate(teacher_models):
            if model_name not in model_results:
                print(f"Skipping {model_name} as it is not loaded.")
                ml_model_list.append(None) 
                continue

            train_results = get_subset(model_results[model_name], seed, 'train').select(x_train.index.to_numpy())
            all_preds = [(pred, None) for pred in train_results['preds']]
            
            # --- 1. Validate Predictions ---
            valid_indices = [i for i, (pred, _) in enumerate(all_preds) if pred in TARGET_2_LABEL[DATASET_NAME]]
            y_train_labels = [all_preds[i] for i in valid_indices] 
            y_train_labels = [target_2_label(pred, DATASET_NAME) for pred, _ in y_train_labels]

            # Skip if model predicts only one class
            if len(set(y_train_labels)) < 2:
                print(f"Skipping {model_name}: All predictions are constant {set(y_train_labels)}.")
                ml_model_list.append(None)
                continue
            
            valid_model_indices.append(idx)

            # --- 2. Collect Predictions for EM ---
            current_model_preds_int = []
            for pred, _ in all_preds:
                if pred in TARGET_2_LABEL[DATASET_NAME]:
                    current_model_preds_int.append(TARGET_2_LABEL[DATASET_NAME][pred])
                else:
                    current_model_preds_int.append(-1)
            teacher_preds_collection.append(current_model_preds_int)

            # --- 3. Prepare Training Data for specific Student ---
            is_binary_subclass = False
            label_map = None
            if N_LABEL > 2:
                unique_labels = sorted(set(y_train_labels))
                if len(unique_labels) == 2:
                    is_binary_subclass = True
                    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                    y_train_labels = [label_map[y] for y in y_train_labels]

            if student_type == 'catboost':
                x_train_reset = x_train.reset_index(drop=True)
                x_train_subset = x_train_reset.loc[valid_indices]
            else:
                x_train_subset = x_train_input[valid_indices]

            # --- 4. Train Student Model ---
            ml_model = initialize_student_model(x_train_subset, y_train_labels, student_type, random_state=seed)
            ml_model.is_binary_subclass = is_binary_subclass
            ml_model.label_map = label_map
            ml_model.inv_label_map = {v: k for k, v in label_map.items()} if label_map else None
            ml_model_list.append(ml_model)

        # --- Weight Calculation (EM/DS/Voting) ---
        L_matrix = np.array(teacher_preds_collection).T 
        if len(teacher_preds_collection) == 0:
             return [None]*len(teacher_models), np.zeros(len(teacher_models))

        if args.voting == 'simple' or args.distill_mode == 'single':
            # Uniform weights
            valid_weights = np.ones(L_matrix.shape[1]) / L_matrix.shape[1]
            full_weights = np.zeros(len(teacher_models))
            for i, valid_idx in enumerate(valid_model_indices):
                full_weights[valid_idx] = valid_weights[i]
            model_weights = full_weights
            
        else:
            # Latent Truth Estimation
            metric = 'acc' if args.voting == 'acc_weighted' else 'f1' 
            
            if args.voting in ['ds', 'expert_ds']:
                valid_weights = dawid_skene_algorithm(L_matrix, N_LABEL, max_iter=100)
                if np.sum(valid_weights) > 0:
                    valid_weights = valid_weights / np.sum(valid_weights)
                print(f"Seed {seed} - DS Weights: {valid_weights}")
                
            elif args.auto_temp:
                valid_weights, final_temp = auto_tune_em_temperature(L_matrix, N_LABEL, metric=metric, max_iter=100)
                args.em_temperature = final_temp 
                print(f"Seed {seed} - Auto-Tuned (T={final_temp}): {valid_weights}")
            else:
                valid_weights = em_algorithm(L_matrix, N_LABEL, max_iter=100, temperature=args.em_temperature, metric=metric)
                print(f"Seed {seed} - EM Weights: {valid_weights}")

            # Map back to full model list
            full_weights = np.zeros(len(teacher_models))
            for i, valid_idx in enumerate(valid_model_indices):
                full_weights[valid_idx] = valid_weights[i]

            if np.sum(full_weights) > 0:
                full_weights = full_weights / np.sum(full_weights)

            # --- Expert Selection (Mean Filter) ---
            if args.voting in ['expert_simple', 'expert_ds']:
                mean_w = np.mean(valid_weights)
                
                experts_indices = [i for i, w in enumerate(full_weights) if w >= mean_w]
                
                # Fallback to Top-1 if filter is too strict
                if len(experts_indices) == 0:
                    print(f"  [Expert] No model met threshold. Fallback to Top-1.")
                    max_idx = np.argmax(full_weights)
                    experts_indices = [max_idx]
                
                # Reset expert weights to uniform
                new_weights = np.zeros_like(full_weights)
                expert_weight_value = 1.0 / len(experts_indices)
                for idx in experts_indices:
                    new_weights[idx] = expert_weight_value
                
                print(f"Seed {seed} - Selected {len(experts_indices)} experts. Weights reset to uniform.")
                full_weights = new_weights

            if np.sum(full_weights) > 0:
                full_weights = full_weights / np.sum(full_weights)
            
            model_weights = full_weights

        return ml_model_list, model_weights


if __name__ == "__main__":
    print(f"\n🚀 Starting experiment for dataset: {DATASET_NAME}")
    print(f"Method: {args.method}, Distill Mode: {args.distill_mode}, Voting: {args.voting}\n")

    all_acc = []
    all_f1 = []
    all_seed_weights = []
    
    for seed in SEEDS:
        print(f"\n==================== Seed {seed} ====================")
        if args.method != '0shot':
            ml_model_list, model_weights = run_distillation_process(args.method, seed) 
        else:
            ml_model_list, model_weights = [], []

        if args.method in ['ours', 'snorkel']:
            if isinstance(model_weights, np.ndarray):
                all_seed_weights.append(model_weights.tolist())
            elif isinstance(model_weights, list):
                all_seed_weights.append(model_weights)

            x_test_df, _ = prepare_data_matrices(seed, 'test', args.bin)
            test_dataset = all_datasets[f"test_{seed}"]
            y_true = test_dataset['label']

            models_y_pred = []
            for ml_model in ml_model_list:
                if ml_model is None:
                    models_y_pred.append(None)
                    continue
                
                if args.ml_model == 'catboost':
                    y_pred = ml_model.predict(x_test_df)
                else:
                    y_pred = ml_model.predict(x_test_df.to_numpy())
                y_pred = y_pred.ravel()
                
                if hasattr(ml_model, "is_binary_subclass") and ml_model.is_binary_subclass:
                    y_pred = [ml_model.inv_label_map.get(int(p), p) for p in y_pred]
                models_y_pred.append(y_pred)
            
            n_test = len(y_true)
            vote_matrix = np.zeros((n_test, N_LABEL))
            
            for k, preds in enumerate(models_y_pred):
                if preds is None:
                    continue
                    
                w = model_weights[k]
                for i, p in enumerate(preds):
                    p_int = int(p)
                    if 0 <= p_int < N_LABEL:
                        vote_matrix[i, p_int] += w
            
            final_preds = np.argmax(vote_matrix, axis=1)

            acc = accuracy_score(y_true, final_preds)
            f1 = f1_score(y_true, final_preds, average='macro')

        elif args.method == '0shot':
            test_dataset = get_dataset_without_guidance("test", seed)
            test_results = get_subset(model_results[args.model], seed, 'test')
            preds = test_results['preds']

            acc = accuracy_score(test_dataset['target'], preds)
            f1 = f1_score(test_dataset['target'], preds, average='macro')
            
        print(f"Seed {seed}: Test Acc = {acc:.4f}, F1 = {f1:.4f}")

        all_acc.append(acc)
        all_f1.append(f1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create result log entry
    result_entry = {
        "data_file": DATASET_NAME,
        "method": args.method,
        "mean_acc": float(np.mean(all_acc)),
        "std_acc": float(np.std(all_acc)),
        "mean_f1": float(np.mean(all_f1)),
        "std_f1": float(np.std(all_f1)),
    }

    if args.method == 'ours':
        result_entry["distill_mode"] = args.distill_mode

    if not (args.method == 'snorkel' or (args.method == 'ours' and args.distill_mode == 'ensemble')):
        result_entry["model"] = args.model

    if args.method == 'ours' and args.distill_mode == 'ensemble':
        result_entry["voting"] = args.voting

    if args.method in ['ours', 'snorkel']:
        result_entry.update({
            "em_temperature": args.em_temperature,
            "ml_model": args.ml_model
        })
        if all_seed_weights:
            avg_weights = np.mean(all_seed_weights, axis=0).tolist()
            result_entry["avg_model_weights"] = avg_weights
    
    # Save results
    if args.method == 'ours':
        output_file = "meld_result.jsonl" if args.distill_mode == "ensemble" else "single_llm_distillation_result.jsonl"
    elif args.method == '0shot':
        output_file = "single_llm_result.jsonl"
    elif args.method == 'snorkel':
        output_file = "snorkel_result.jsonl"
    else:
        output_file = "other_result.jsonl"

    with open(output_file, "a") as f:
        f.write(json.dumps(result_entry) + "\n")

    print(f"\n✅ Results saved to {output_file}")