model_name=$1

PORT=22446
NUM_PROCESSES=$2
CONFIG=config/language_modeling/pretrain.yaml


if [ -z $NUM_PROCESSES ]; then
    NUM_PROCESSES=2
    CUDA_VISIBLE_DEVICES=0,1
elif [ $NUM_PROCESSES -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=0
fi

icae_mem_size=16

if [ $model_name == "mistral" ]; then
    model_name_or_path=mistralai/mistral-7b-instruct-v0.2
    learning_rate=1e-4
    per_device_train_batch_size=8 # 2 GPUs, total batch size: 384
    gradient_accumulation_steps=24
    exp_name=care_mistral_pt
elif [ $model_name == "llama" ]; then
    model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
    learning_rate=2e-4
    per_device_train_batch_size=24 # 2 GPUs, total batch size: 384
    gradient_accumulation_steps=8
    exp_name=care_llama_pt
elif [ $model_name == "qwen" ]; then
    model_name_or_path=Qwen/Qwen3-8B
    learning_rate=2e-4
    per_device_train_batch_size=24 
    gradient_accumulation_steps=8 # 2 GPUs, total batch size: 384
    exp_name=care_qwen_pt
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes ${NUM_PROCESSES} \
    --main_process_port ${PORT} \
	-m src.language_modeling.train \
    --config ${CONFIG} \
    --model_name_or_path ${model_name_or_path} \
    --icae_mem_size ${icae_mem_size} \
    --learning_rate ${learning_rate} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --exp_name ${exp_name}