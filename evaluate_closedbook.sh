# ------------------------------------------------------------------
# Note
# Before running this script, please make sure you are logged into Hugging Face Hub.
# You need access to gated models such as mistralai/Mistral-7B-Instruct-v0.2.
#
# Login with your HF token on terminal:
#    huggingface-cli login
#    # Paste your hf_xxx... token when prompted
#
# This login step only needs to be done once per environment,
# and the token will be cached under ~/.huggingface/token.
# ------------------------------------------------------------------

GPU=0

data="nq"
path=mistralai/mistral-7b-instruct-v0.2

echo "Evaluating ${path} / Eval data ${data}"
CUDA_VISIBLE_DEVICES=${GPU} python -m src.eval.run_eval --data ${data} \
                                                        --model_name_or_path ${path} \
                                                        --eval_batch_size 8 \
                                                        --save_results \
                                                        --eval_file_name train.jsonl \
                                                        --retrieval_file_name train.jsonl \
                                                        --results_path nq_train_closed_book_mistral.json