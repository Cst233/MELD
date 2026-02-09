# ==============================================================================
# 1. Single LLM Zero-shot Results Retrieval
# ==============================================================================

# Step 1: Run save_prediction.py to save raw model predictions.
# (Ensure this is run before Step 2)

# for model in "Llama-3.1-8B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-14B-Instruct" "Qwen3-14B" "Qwen3-8B" "Ministral-8B-it" 'gemma-2-9b-it' "gemma-3-12b-it" "phi4" "gpt-4o-mini"; do
#     for dataset in 'heart' 'diabetes' 'blood' 'calhousing' 'car' 'student' 'employee' 'jungle' 'health' 'bank' 'income'; do
#         python save_prediction.py \
#         --model $model \
#         --datafile $dataset 
#     done
# done

# Step 2: Run new_exp.py with method '0shot'.

# for model in "Llama-3.1-8B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-14B-Instruct" "Qwen3-14B" "Qwen3-8B" "Ministral-8B-it" 'gemma-2-9b-it' "gemma-3-12b-it" "phi4" "gpt-4o-mini"; do 
#     for dataset in 'heart' 'diabetes' 'blood' 'calhousing' 'car' 'student' 'employee' 'jungle' 'health' 'bank' 'income'; do
#         python new_exp.py --method "0shot" --model "$model" --datafile "$dataset"
#     done
# done


# ==============================================================================
# 2. MeLD Results Retrieval 
# ==============================================================================

# Step 1: Run new_exp.py directly with method='ours'.
# Note: 'distill_mode' defaults to 'ensemble' here.

for ml_model in "lr" "catboost"; do
    for dataset in 'heart'; do
        for voting in "expert_simple"; do
            python new_exp.py \
                --method "ours" \
                --datafile "$dataset" \
                --ml_model "$ml_model" \
                --auto_temp \
                --voting $voting
        done
    done
done


# ==============================================================================
# 3. Different ML Models distilled from Single LLMs
# ==============================================================================

# Step 1: Ensure LLM predictions are already saved (via save_prediction.py in Section 1, Step 1).

# Step 2: Run new_exp.py with method='ours' and distill_mode='single', specifying the ml_model.

# for model in "Llama-3.1-8B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-14B-Instruct" "Qwen3-14B" "Qwen3-8B" "Ministral-8B-it" 'gemma-2-9b-it' "gemma-3-12b-it" "phi4" "gpt-4o-mini"; do 
#     for ml_model in "lr" "catboost"; do
#         for dataset in 'heart' ; do
            # python new_exp.py \
            #     --method "ours" \
            #     --datafile "$dataset" \
            #     --ml_model "$ml_model" \
            #     --model $model \
            #     --distill_mode single
#         done
#     done
# done