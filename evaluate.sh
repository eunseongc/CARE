GPU=0

path=eunseong/care_mistral
# path={your_path}, e.g., wandb/care/files/checkpoint/step_1800

# datasets="nq triviaqa hotpotqa webqa truthfulqa"
datasets="nq"

for data in ${datasets};
do
    echo "Evaluating ${path} / Eval data ${data}"
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.eval.run_eval --data ${data} \
                                                            --checkpoint_path ${path} \
                                                            --use_rag \
                                                            --eval_batch_size 8 \
                                                            --save_results \
                                                            --retrieval_file_name test_question_aware.jsonl \
                                                            --results_path care_mistral_${data}.json
done