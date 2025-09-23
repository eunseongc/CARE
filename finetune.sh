model_name=$1

PORT=22446
NUM_PROCESSES=$2
CONFIG=config/language_modeling/finetune.yaml


if [ -z $NUM_PROCESSES ]; then
    NUM_PROCESSES=2
    CUDA_VISIBLE_DEVICES=0,1
elif [ $NUM_PROCESSES -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=0
fi

if [ $model_name == "mistral" ]; then
    model_name_or_path=mistralai/mistral-7b-instruct-v0.2
    train_file=data_care/finetune/nq_mistral.jsonl
    checkpoint_path=eunseong/care_mistral_pt # Pre-trained checkpoint
    learning_rate=1e-4
    num_train_epochs=2
    per_device_train_batch_size=8
    gradient_accumulation_steps=4 # 2 GPUs, total batch size: 64
    alpha_kl=2
    exp_name=care_mistral
elif [ $model_name == "llama" ]; then
    model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
    train_file=data_care/finetune/nq_llama.jsonl
    checkpoint_path=eunseong/care_llama_pt # Pre-trained checkpoint
    learning_rate=3e-4
    num_train_epochs=2
    per_device_train_batch_size=16
    gradient_accumulation_steps=2 # 2 GPUs, total batch size: 64
    alpha_kl=4
    exp_name=care_llama
elif [ $model_name == "qwen" ]; then
    model_name_or_path=Qwen/Qwen3-8B
    train_file=data_care/finetune/nq_qwen.jsonl
    checkpoint_path=eunseong/care_qwen_pt # Pre-trained checkpoint
    learning_rate=3e-4
    num_train_epochs=4
    per_device_train_batch_size=16
    gradient_accumulation_steps=2 # 2 GPUs, total batch size: 64
    alpha_kl=2
    exp_name=care_qwen
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes ${NUM_PROCESSES} \
    --main_process_port ${PORT} \
	-m src.language_modeling.train \
    --config ${CONFIG} \
    --dev_file data_care/finetune/nq_valid.jsonl \
    --train_file ${train_file} \
    --model_name_or_path ${model_name_or_path} \
    --checkpoint_path ${checkpoint_path} \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --alpha_kl ${alpha_kl} \
    --exp_name ${exp_name}