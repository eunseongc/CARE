import re
import os
import math
import logging
import random
import pickle
import json
import shutil
import inspect
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"]='*.pth' ## not upload ckpt to wandb cloud
import sys
import datasets
import torch
import datetime
torch.distributed.distributed_c10d._default_pg_timeout = datetime.timedelta(seconds=60)
import torch.distributed as dist
import torch.nn.functional as F
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from peft import LoraConfig
from pathlib import Path
import transformers
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    get_scheduler, PreTrainedTokenizerFast
)
from tokenizers import AddedToken
    
from src.model import (
    ICAE,
    MistralICAEConfig,
    LlamaICAEConfig,
    QwenICAEConfig,
)
from src.language_modeling.utils import (
    get_nll_loss,
    get_kl_loss,
    save_with_accelerate,
    XRAG_TOKEN,
    get_retrieval_embeds,
    collator,
    validate_during_pretrain, validate_during_finetune, 
    update_raw_datasets,
)
from src.language_modeling.preprocessing import (
    encode_with_chat_format_pretrain,
    encode_with_chat_format_finetune,
)
from src.utils import (
    parse_args,
)

logger = get_logger(__name__)


def extract_number(key):
    match = re.search(r"rq\.vq_layers\.(\d+)\.embedding\.weight", key)
    return int(match.group(1)) if match else float('inf')  # Convert to int for numeric sorting


def main():
    args = parse_args()
    set_seed(args.seed)

    icae_mem_size = args.icae_mem_size
    mem_hidden_size = 4096
    print(f"> chat_format is ICAE, ICAE Memory Size: {icae_mem_size}")

    print(f"> Mem size: {icae_mem_size}, Mem Hidden Size: {mem_hidden_size}")
    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs]
        )
    
    accelerator.init_trackers(
        project_name=args.project_name, 
        config=args,
        init_kwargs={
            "wandb": {
                "dir": args.workdir, 
                "name": args.exp_name if args.exp_name is not None else None,
                "notes": args.exp_note if args.exp_note is not None else None,
                "save_code": False,
            },
        }
    )

    accelerator.print(json.dumps(vars(args),indent=4))
    checkpoint_dir = [None]
    if accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        wandb_tracker.run.log_code("")
        checkpoint_dir = [os.path.join(wandb_tracker.run.dir, 'checkpoint')]
        shutil.copy(args.config, os.path.join(wandb_tracker.run.dir))
            
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast_object_list(checkpoint_dir, src=0)

    # if accelerator.use_distributed:dist.broadcast_object_list(checkpoint_dir,src=0)
    args.output_dir = checkpoint_dir[0]

    print(f"> output_dir: {args.output_dir}")


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side='left',
        use_fast=False,
    )

    if args.model_name_or_path == 'mistralai/mistral-7b-instruct-v0.2':
        lora_r, lora_alpha, target_modules = 512, 32, ["q_proj", "v_proj"]
        CONFIG_CLASS = MistralICAEConfig
    elif args.model_name_or_path == 'meta-llama/Meta-Llama-3-8B-Instruct':
        lora_r, lora_alpha, target_modules = 64, 128, ["q_proj", "v_proj"]
        CONFIG_CLASS = LlamaICAEConfig
    elif args.model_name_or_path == 'Qwen/Qwen3-8B':
        lora_r, lora_alpha, target_modules = 8, 16, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        CONFIG_CLASS = QwenICAEConfig
    else:
        raise ValueError(f"Unsupported model: {args.model_name_or_path}")

    config = CONFIG_CLASS.from_pretrained(args.model_name_or_path,
                                          icae_mem_size= args.icae_mem_size, 
                                          mem_hidden_size=mem_hidden_size,
                                          torch_dtype=torch.bfloat16 if accelerator.mixed_precision == 'bf16' else 'auto',
    )                          
    config.init_train_args(args)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = ICAE(config, 
                args.model_name_or_path,
                lora_config)

    retriever = "ICAE_mem"
    retriever_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if isinstance(retriever_tokenizer, LlamaTokenizer) or isinstance(retriever_tokenizer, LlamaTokenizerFast) or isinstance(tokenizer, PreTrainedTokenizerFast):
        num_added_tokens = retriever_tokenizer.add_special_tokens({
            "pad_token": "<pad>", ## same as eos, it will be ignored anyway
        })

    ### -------------- ####
    model_file = inspect.getfile(ICAE)
    shutil.copy(model_file, os.path.join(args.output_dir, os.path.basename(model_file)))
    print(f"> Save the model code to the output directory, {model_file}")
    ### -------------- ####
    num_added_tokens = 0
    ## mistral tokenizer is also a LLamaTokenizer
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast) or isinstance(tokenizer, PreTrainedTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>", ## same as eos, it will be ignored anyway
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."

    ## XRAG_TOKEN simply functions as a placeholder, would not be trained
    num_added_tokens += tokenizer.add_tokens([AddedToken(XRAG_TOKEN, lstrip=False, rstrip=False)])
    xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    model.set_xrag_token_id(xrag_token_id)

    if num_added_tokens > 0:
        with torch.no_grad():
            model.resize_token_embeddings(len(tokenizer), len(retriever_tokenizer))  # Resize on CPU (faster memory allocation)

    if args.checkpoint_path is not None:
        if os.path.exists(os.path.join(args.checkpoint_path, "pytorch_model.bin.index.json")):
            print(f"> loading checkpoint from {args.checkpoint_path} by from_pretrained")
            config = CONFIG_CLASS.from_pretrained(args.checkpoint_path,
                                          mem_hidden_size=mem_hidden_size,
                                          icae_mem_size=icae_mem_size,
                                          torch_dtype=torch.bfloat16 if accelerator.mixed_precision == 'bf16' else 'auto',
            )                                     
            

            model = ICAE.from_pretrained(
                args.checkpoint_path,
                config=config,
                use_flash_attention_2=args.use_flash_attn,
                torch_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else 'auto',
            )
            model.set_xrag_token_id(xrag_token_id)

        else:
            print(f"> loading checkpoint from {args.checkpoint_path}")
            
            if os.path.exists(os.path.join(args.checkpoint_path, "ckpt.pth")):
                ckpt_path = os.path.join(args.checkpoint_path, "ckpt.pth")
            else:
                ckpt_path = hf_hub_download(
                    repo_id=args.checkpoint_path,
                    filename="ckpt.pth",
                    repo_type="model"
                )
            ckpt = torch.load(ckpt_path, map_location="cpu")

            # ckpt = torch.load(os.path.join(args.checkpoint_path, "ckpt.pth"), map_location=torch.device('cpu'))
            ## Update for those parameters that are in the ckpt.keys()
            # 2. Get the model’s full state_dict
            model_state_dict = model.state_dict()

            # 3. Overwrite parameters from the checkpoint if they exist in the model
            for name, param in ckpt.items():
                if name in model_state_dict:
                    ## update the model state_dict
                    ## if shape mismatch, exit
                    if model_state_dict[name].shape != param.shape:
                        print(f"> Warning: parameter '{name}' shape mismatch. Skipping.")
                        exit()
                    else:
                        model_state_dict[name].copy_(param)
                else:
                    print(f"> Warning: parameter '{name}' not found in model state_dict. Skipping.")
                    exit()
            model.load_state_dict(model_state_dict)

    # Prepare optimizer
    params_requires_grad = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        params_requires_grad,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Prepare the data
    #########################################################################################

    data_files = {}
    dataset_args = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.dev_file is not None:
        data_files['dev'] = args.dev_file
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        **dataset_args,
    )

    ## select N samples, mainly for debug
    if args.max_train_samples is not None and len(raw_datasets['train']) > args.max_train_samples:
        selected_indices = random.sample(range(len(raw_datasets['train'])),args.max_train_samples)
        raw_datasets['train'] = raw_datasets['train'].select(selected_indices)
    
    vocab_size = len(tokenizer)
    
    #edit
    if args.select_criteria is not None:
        print(f"> select_criteria is not None, {args.select_criteria}")
        print(f"> Convert the background to {args.ctx_select}_background if the {args.select_criteria} is satisfied")
        raw_datasets['train'] = raw_datasets['train'].map(update_raw_datasets(args.select_criteria, args.ctx_select))
    
    # f"{args.ctx_select}_background)
    if args.task_type == 'pretrain':
        encode_function = partial(
            encode_with_chat_format_pretrain,
            tokenizer=tokenizer,
            max_seq_length = args.max_seq_length,
            icae_mem_size=icae_mem_size,
            chat_format = args.chat_format,
        )
    elif args.task_type == 'finetune':
        encode_function = partial(
            encode_with_chat_format_finetune, # if "messages" in raw_datasets["train"].column_names else encode_with_completion_format_finetune,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            icae_mem_size=icae_mem_size,
            retrieval_context_length=args.retrieval_context_length,
            use_rag_tuning = args.use_rag_tuning,
            use_retriever_embed = not (retriever is None), ## False if retriever is None
            retriever_tokenizer = retriever_tokenizer,
            chat_format = args.chat_format,
            ctx_set = set([args.ctx_nll, args.ctx_teacher, args.ctx_student, args.ctx_select, 'none']),
        )

    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Tokenizing and reformatting data on rank: {accelerator.local_process_index}",
        )
            # remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["id", "input_ids", "labels", "attention_mask", "random_background_id"]],

        lm_datasets.set_format(type="pt")
        if args.task_type == 'finetune':
            lm_datasets['train'] = lm_datasets['train'].filter(lambda example: (example['labels'] != -100).any())
            if (args.alpha_kl is not None and args.alpha_kl > 0.0):
                lm_datasets['train'] = lm_datasets['train'].filter(
                    lambda example:
                    (example['gt_labels']!=-100).sum() == (example['xrag_labels']!=-100).sum()
                )
                if 'random_labels' in lm_datasets['train'].column_names:
                    lm_datasets['train'] = lm_datasets['train'].filter(
                        lambda example:
                        (example['random_labels']!=-100).sum() == (example['xrag_labels']!=-100).sum()
                    )
                ## if 'hn_labels' in example keys
                if 'hn_labels' in lm_datasets['train'].column_names:
                    lm_datasets['train'] = lm_datasets['train'].filter(
                        lambda example:
                        (example['hn_labels']!=-100).sum() == (example['xrag_labels']!=-100).sum()
                    )

    train_dataset = lm_datasets["train"]
    dev_dataset = lm_datasets['dev'] if args.dev_file is not None else None
    collate_fn = partial(
        collator,
        task_type=args.task_type,
        llm_tokenizer=tokenizer,
        retriever_tokenizer=retriever_tokenizer,
        retrieval_context_length=args.retrieval_context_length,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size
    )

    dev_dataloader = None
    if dev_dataset is not None:
        dev_dataloader = DataLoader(
            dev_dataset,
            shuffle=False, 
            collate_fn=collate_fn,
            batch_size=args.per_device_train_batch_size
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.he
    params_requires_grad = [param for param in model.parameters() if param.requires_grad]

    if dev_dataset is not None:
        model, optimizer, train_dataloader, lr_scheduler, dev_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, dev_dataloader)
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Max Sequence Length = {args.max_seq_length}")
    logger.info(f"  Trainable Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)/(10**6):.2f} M") ## not applicable for deepspeed


    completed_steps = 0
    starting_epoch = 0

    # logging_interval_grad_norm = 0
    logging_interval_loss = 0
    logging_interval_kl_loss, logging_interval_nll_loss = 0, 0

    total_loss = 0
    total_kl_loss, total_nll_loss = 0, 0

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.update(completed_steps)
    
    # update the progress_bar if load from checkpoint
    save_one_sample = True
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        active_dataloader = train_dataloader

        for batch_idx, batch in enumerate(active_dataloader):
            # if batch_idx > 10:
                # break
            if save_one_sample:
                if accelerator.is_local_main_process:
                    pickle.dump(
                        batch,
                        open(os.path.join(os.path.dirname(args.output_dir),"sample_data.pkl"),'wb'),
                    )
                accelerator.print("**"*20,"show one example","**"*20)
                accelerator.print(batch.keys())
                accelerator.print(tokenizer.decode(batch['xrag_input_ids'][0]))
                accelerator.print(batch['xrag_input_ids'][0])
                if "retriever_input_text" in batch:
                    accelerator.print(batch['retriever_input_text'][0])
                if 'input_ids' in batch:
                    for input_id, label_id, attention_mask in zip(batch['input_ids'][0],batch['labels'][0],batch['attention_mask'][0]):
                        accelerator.print(f"{tokenizer.convert_ids_to_tokens([input_id])[0]}({label_id.item()})({attention_mask})",end=" ")
                accelerator.print()    
                for input_id, label_id, attention_mask in zip(batch['xrag_input_ids'][0],batch['xrag_labels'][0],batch['xrag_attention_mask'][0]):
                    accelerator.print(f"{tokenizer.convert_ids_to_tokens([input_id])[0]}({label_id.item()})({attention_mask})",end=" ")
                accelerator.print('\n'+"**"*20,"show one example","**"*20)
                save_one_sample=False

            with accelerator.accumulate(model):
                retrieval_kwargs = {}

                ctx_nll = args.ctx_nll

                if ctx_nll == 'gt':
                    indices = batch['ids']
                    retriever_input_ids = batch['retriever_input_ids']
                    retriever_attention_mask = batch['retriever_attention_mask']

                    xrag_input_ids = batch['xrag_input_ids']
                    xrag_attention_mask = batch['xrag_attention_mask']
                    nll_labels = batch['xrag_labels']

                else: 
                    indices = batch[f'{ctx_nll}_background_ids']
                    retriever_input_ids = batch[f'{ctx_nll}_retriever_input_ids']
                    retriever_attention_mask = batch[f'{ctx_nll}_retriever_attention_mask']
                    xrag_input_ids = batch[f'{ctx_nll}_xrag_input_ids']
                    xrag_attention_mask = batch[f'{ctx_nll}_xrag_attention_mask']
                    nll_labels = batch[f'{ctx_nll}_xrag_labels']

                base_model = model.module if hasattr(model, 'module') else model
                memory_slots = base_model.icae_forward(
                    input_ids= retriever_input_ids,
                    attention_mask= retriever_attention_mask,
                )

                retrieval_kwargs['retrieval_embeds'] = memory_slots

                outputs = model(
                    input_ids = xrag_input_ids,
                    attention_mask = xrag_attention_mask,
                    **retrieval_kwargs,
                )

                loss = None

                ####### > Calculate NLL Loss
                if args.alpha_nll is not None and args.alpha_nll > 0.0:
                    nll_loss = get_nll_loss(
                        labels = nll_labels,
                        logits = outputs.logits,
                        vocab_size = vocab_size,
                    )

                    logging_interval_nll_loss += nll_loss.detach().float()
                    if loss is not None:
                        loss += args.alpha_nll * nll_loss
                    else:
                        loss = args.alpha_nll * nll_loss

                ####### > Calculate KL Loss
                if args.alpha_kl is not None and args.alpha_kl > 0.0:
                    ctx_student = args.ctx_student

                    with torch.no_grad():
                        model.eval()

                        ctx_teacher = args.ctx_teacher

                        teacher_input_ids = batch[f'{ctx_teacher}_input_ids']
                        teacher_attention_mask = batch[f'{ctx_teacher}_attention_mask']
                        teacher_labels = batch[f'{ctx_teacher}_labels']

                        teacher_outputs = model(
                            input_ids = teacher_input_ids,
                            attention_mask = teacher_attention_mask,
                        )
                        model.train()

                    kl_loss = get_kl_loss(
                        teacher_logits=teacher_outputs.logits,
                        teacher_labels=teacher_labels,
                        student_logits=outputs.logits,
                        student_labels=nll_labels,
                        temperature=args.kl_temperature,
                        vocab_size=vocab_size, ## [ES]
                    )

                    logging_interval_kl_loss += kl_loss.detach().float()
                    if loss is not None:
                        loss += args.alpha_kl * kl_loss
                    else:
                        loss = args.alpha_kl * kl_loss

                logging_interval_loss += loss.detach().float()
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                        
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(logging_interval_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    total_loss += accelerator.gather(logging_interval_loss).mean().item() / args.gradient_accumulation_steps 

                    to_be_logged = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "train_loss": avg_loss,
                        "rolling_loss":total_loss / completed_steps,
                    }
                    if args.alpha_nll is not None and args.alpha_nll > 0.0:
                        total_nll_loss += accelerator.gather(logging_interval_nll_loss).mean().item() / args.gradient_accumulation_steps
                        to_be_logged["rolling_nll_loss"] = total_nll_loss  / completed_steps

                    if args.alpha_kl is not None and args.alpha_kl > 0.0:
                        total_kl_loss  += accelerator.gather(logging_interval_kl_loss).mean().item() / args.gradient_accumulation_steps
                        to_be_logged["rolling_kl_loss"] = total_kl_loss  / completed_steps

                    accelerator.log(to_be_logged, step=completed_steps)
                    
                    # logging_interval_grad_norm = 0
                    logging_interval_loss = 0
                    logging_interval_kl_loss = 0
                    logging_interval_nll_loss = 0
                    
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")

                        save_with_accelerate(accelerator, model, tokenizer, output_dir, args, save_projector_only=args.update_projector_only)

                        if dev_dataloader is not None:
                            if args.task_type == 'pretrain':
                                ppl = validate_during_pretrain(model, dev_dataloader, accelerator, vocab_size, retriever, id2indices, id2embed_for_retriever, ret_embeddings)
                                print(f"> {completed_steps} steps, dev ppl: {ppl}")
                                accelerator.log({"dev_ppl":ppl},step=completed_steps)
                            else: 
                                score = validate_during_finetune(model, dev_dataloader, accelerator, retriever, tokenizer, output_dir)
                                print(f"> {completed_steps} steps, EM score: {score}")
                                accelerator.log({"Span-EM":score},step=completed_steps)

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args, save_projector_only=args.update_projector_only)

    output_dir = os.path.join(args.output_dir, "last")

    if dev_dataloader is not None:
        if args.task_type == 'pretrain':
            ppl = validate_during_pretrain(model, dev_dataloader, accelerator, vocab_size, retriever, id2indices, id2embed_for_retriever, ret_embeddings)
            print(f"> {completed_steps} steps, dev ppl: {ppl}")
            accelerator.log({"dev_ppl":ppl},step=completed_steps)
        else: 
            score = validate_during_finetune(model, dev_dataloader, accelerator, retriever,tokenizer, output_dir)
            print(f"> {completed_steps} steps, EM score: {score}")
            accelerator.log({"Span-EM":score},step=completed_steps)
    accelerator.end_training()

    save_with_accelerate(accelerator, model, tokenizer, output_dir, args, save_projector_only=args.update_projector_only)
    if accelerator.is_main_process:
        new_output_dir_name = f"wandb/{args.exp_name}"
        if os.path.exists(new_output_dir_name):
            cur_date_time = datetime.datetime.now().strftime("%m%d%H%M")
            new_output_dir_name = new_output_dir_name + f"_{cur_date_time}"
        os.rename('/'.join(wandb_tracker.run.dir.split('/')[:-1]), new_output_dir_name)
        print(f"> Successfully renamed {'/'.join(wandb_tracker.run.dir.split('/')[:-1])} to {new_output_dir_name}")

if __name__ == "__main__":
    main()