import os
import argparse

from pathlib import Path
from .utils import get_yaml_file

from transformers import SchedulerType

def add_args(parser):
    parser.add_argument(
        "--use_fast_tokenizer",
        type=eval,
    )
    parser.add_argument(
        "--use_rag_tuning",
        type=eval,
        help='whether to use retrieval-augmented instruction tuning'
    )
    parser.add_argument(
        "--chat_format",
        choices=['mistral','tulu','mixtral','qwen','yi','gemma']
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
    )
    parser.add_argument(
        "--update_projector_only",
        type=eval,
    )
    parser.add_argument(
        "--workdir",
        type=str,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config file to launch the training"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        help="pretrain or finetune"
    )
    parser.add_argument(
        "--retrieval_context_length",
        type=int,
        help="max token number for document encoder in dense retrieval",
    )
    parser.add_argument(
        "--alpha_nll",
        type=float,
        help="coefficient for multi-task learning",
    )
    parser.add_argument(
        "--alpha_kl",
        type=float,
        help="coefficient for multi-task learning",
    )
    parser.add_argument(
        "--kl_temperature",
        type=float,
        help="Temperature coefficient for calculation KL-Divergency loss",
    )
    parser.add_argument(
        "--train_file",
        type=str, 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--dev_file",
        type=str, 
        help="A csv or a json file containing the dev data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--use_flash_attn",
        type=eval,
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument(
        "--project_name",
        type=str
    )
    parser.add_argument(
        "--exp_name",
        type=str
    )
    parser.add_argument(
        "--exp_note",
        type=str
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=eval, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )

    ########################### ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ are by ES.C ###############################

    parser.add_argument(
        '--ret_embedding_path',
        type=str,
        default=None,
        help='path to the retriever embedding, id2embed.pkl, containing the retrieval embeddings according to the document id',
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--demo",
        type=eval,
        help="Whether to run the demo or not."
    )
    parser.add_argument(
        "--icae_mem_size",
        type=int,
        help="memory size for ICAE",
    )
    parser.add_argument(
        "--ctx_nll",
        type=str,
        default='gt',
        choices=['gt', 'random', 'hn'] ## none does not make sense
    )
    parser.add_argument(
        "--ctx_student",
        type=str,
        default='gt',
        choices=['gt', 'random', 'hn', 'none']
    )
    parser.add_argument(
        "--ctx_teacher",
        type=str,
        default='gt',
        choices=['gt', 'random', 'hn', 'none', 'adaptive']
    )
    # options: gt, random, none
    parser.add_argument(
        "--ctx_nll_gt_prob",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--ctx_student_gt_prob",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--ctx_teacher_gt_prob",
        type=float,
        default=0.0,
    )

    parser.add_argument( ## Work for only students
        "--select_criteria",
        type=str,
        help="criteria for selecting background",
        default=None,
        choices=['closed_book_correct', 'priori_judge_rag'] ## I'm not planning to do vanila priori_judge
    )
    parser.add_argument( ## Work for only students
        "--ctx_select",
        type=str,
        help="background for selecting background",
        default=None,
        choices=['random', 'hn']
    )
    parser.add_argument(
        "--same_nll_kl",
        type=eval,
    )

def parse_args():
    
    parser_temp = argparse.ArgumentParser()
    _, args_input = parser_temp.parse_known_args()
    args_input = set([arg.replace('--','') for arg in args_input[::2]])
    print(args_input)
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    
    yaml_config = get_yaml_file(args.config)

    ## priority: CLI > YAML (with all default value set to None in argument parser)
    for k, v in yaml_config.items():
        assert hasattr(args, k), f"{k} not in parsed arguments"
        if k in args_input:
            print(f"> [ES] CLI argument is given for {k}, so ignore the YAML config.")
            continue
        setattr(args, k, v) ## [ES] none if문 제거함.
    
    if args.demo:
        print("*" * 100)
        print("*" * 50)
        print("\n\n> [ES] Running the demo mode.\n\n")
        print("*" * 50)
        print("*" * 100)

        ## train, dev are essential values for training
        args.train_file = args.train_file + "_demo"
        args.dev_file = args.dev_file + "_demo"
        if args.eval_file is not None:
            args.eval_file = args.eval_file + "_demo"
        if args.ret_embedding_path is not None:
            args.ret_embedding_path = args.ret_embedding_path + "_demo"
        # if args.index_path is not None:
        #     args.index_path = args.index_path + "_demo"
        args.checkpointing_steps = 1
        args.logging_steps = 1
        args.preprocessing_num_workers = 1


    ## Sanity check
    if args.select_criteria is not None:
        assert args.ctx_select is not None, "ctx_select must be set if select_criteria is set"
    if args.ctx_select is not None:
        assert args.select_criteria is not None, "select_criteria must be set if ctx_select is set"
        assert args.ctx_nll == 'gt', "ctx_nll must be gt if ctx_select is set"
        assert args.ctx_student == 'gt', "ctx_student must be gt if ctx_select is set"
        assert args.ctx_select != "gt", "ctx_select must not be gt"

    return args