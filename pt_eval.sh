checkpoint_path={your_path} # e.g.,wandb/pretrain_icae_mem16_mistral/files/checkpoint/last"
CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_pt  --checkpoint_path ${checkpoint_path} \
												   --eval_file data/pretrain/wikipedia/dev.jsonl \
												   --config config/language_modeling/pretrain.yaml