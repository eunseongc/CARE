## built-in
import argparse, json, os, pickle
import time
import numpy as np

from peft import LoraConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
)
import torch
import datasets
from tqdm import tqdm
import pandas as pd

from pathlib import Path
from tokenizers import AddedToken

from src.model import (
    ICAE,
    MistralICAEConfig,
    LlamaICAEConfig,
    QwenICAEConfig,
)

from src.language_modeling.utils import (
    XRAG_TOKEN,
    get_retrieval_embeds,
    get_memory_slots
)

from src.eval.utils import (
    stop_sequences_criteria,
    get_substring_match_score,
    eval_fact_checking,
    eval_truthfulqa,
    keyword_extraction_with_tfidf,
)
from src.utils import (
    get_jsonl,
    get_yaml_file,
)

from huggingface_hub import hf_hub_download



def create_prompt_with_icae_chat_format(messages, tokenizer, *args, **kwargs):
    formatted_text = ""
    for message in messages:
        if message['role'] == 'user':
            formatted_text += "[INST] " + message['content'] + " [/INST]"
        elif message['role'] == 'assistant':
            formatted_text += message['content'] + tokenizer.eos_token
        else:
            raise ValueError(
                "icae chat template only supports 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    # formatted_text += " The answer is:"
    return formatted_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_fast_tokenizer",
        type=eval,
    )
    parser.add_argument(
        "--retrieval_prefix",
        default='colbertv2'
    )
    parser.add_argument(
        "--tf_idf_topk",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--base_model",
    )
    parser.add_argument(
        "--use_rag",
        action='store_true',
    )
    parser.add_argument(
        "--enable_progress_bar",
        type=eval,
        default=True,
    )
    parser.add_argument(
        "--data",
    )
    parser.add_argument(
        "--data_root",
        default="data_care",
    )
    parser.add_argument(
        "--eval_file_name",
        default="test.jsonl",
    )
    parser.add_argument(
        "--retriever_name_or_path",
    )
    parser.add_argument(
        "--retrieval_file_name",
        type=str,
        default="test_question_aware.jsonl",
        help="Name of the retrieval file.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="mistralai/mistral-7b-instruct-v0.2"
    )
    parser.add_argument(
        "--checkpoint_path",
    )
    parser.add_argument(
        "--eval_metrics",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--retrieval_topk",
        type=int,
        default=[1],
        nargs='+',
    )
    parser.add_argument(
        "--icae_mem_size",
        type=int, default=0,
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        help="for debug",
    )
    parser.add_argument(
        "--save_dir",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--chat_format",
        default='icae',
    )
    parser.add_argument(
        "--save_results",
        action='store_true',
        help="save the generated results",
    )
    parser.add_argument(
        "--step_folders",
        default=None,
        type=str,
        help="path to the retrieval embedding",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=30,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="results.json",
        help="Path to save the results.",
    )

    args = parser.parse_args()

    # args.checkpoint_path is None when predicting on rag or closed book setting
    if args.checkpoint_path is not None and 'config.yaml' in os.listdir(Path(args.checkpoint_path).parents[1]):
        checkpoint_config = get_yaml_file(os.path.join(Path(args.checkpoint_path).parents[1], 'config.yaml'))
        for k, v in vars(args).items():
            if k == 'checkpoint_path': continue
            if checkpoint_config.get(k) is not None:
                setattr(args, k, checkpoint_config[k]['value'])
    else:
        if args.checkpoint_path is not None:
            if 'mistral' in args.checkpoint_path:
                args.model_name_or_path = 'mistralai/mistral-7b-instruct-v0.2'
            elif 'llama' in args.checkpoint_path:
                args.model_name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
            elif 'qwen' in args.checkpoint_path:
                args.model_name_or_path = 'Qwen/Qwen3-8B'
            else:
                raise ValueError(f"Unsupported model: {args.checkpoint_path}, please check the checkpoint path")

        if args.use_rag:
            args.icae_mem_size = 16
            args.chat_format = 'icae'

    if args.save_results is not None:
        if not os.path.exists("ft_results"):
            os.makedirs("ft_results", exist_ok=True)
        if args.step_folders is not None:
            os.makedirs(f"ft_results/{args.step_folders}", exist_ok=True)
            folder_prefix = f"ft_results/{args.step_folders}"
        else:
            folder_prefix = "ft_results"

        args.results_path = os.path.join(folder_prefix, args.results_path)

        print(f"> [ES] save results is enabled, saving to {args.results_path}")

    ## post-process
    if args.data in ['nq', 'hotpotqa', 'triviaqa', 'webqa']:
        args.task_type = 'open_qa'
        args.eval_metrics = 'substring_match'
    elif args.data in ['truthfulqa']:
        args.task_type = 'open_qa'
        args.eval_metrics = 'truthfulqa_f1_rl'
    elif args.data in ['factkg']:
        args.task_type = 'fact_checking'
        args.eval_metrics = 'fact_checking_acc'
    
    args.retrieval_topk = [x-1 for x in args.retrieval_topk] ## rank starts from 1

    return args


QA_PROMPT = "Question: {question}?\n"
FECT_CHECKING_PROPMT = "Claim: {question}\n"
BACKGROUND_PROMPT_TEMPLATE = "Background: {background}\n\n"

PROMPT_TEMPLATES = {
    "open_qa":QA_PROMPT,
    'fact_checking':FECT_CHECKING_PROPMT,
}

def get_start_prompt(task_type, use_rag, sample=None):
    if task_type == 'open_qa':
        return {
            True: "Refer to the background document and answer the questions:",
            False:"Answer the questions:"
        }[use_rag]
    elif task_type == 'fact_checking':
        return {
            True: "Refer to the background document and verify the following claims with \"True\" or \"False\":",
            False:"Verify the following claims with \"True\" or \"False\":"
        }[use_rag]
        

@torch.no_grad()
def prepare_retrieval_embeds(backgrounds, retriever, tokenizer, batch_size = 16, memory_slots = False):
    print(f"> Preparing retrieval embeds, total number of backgrounds: {len(backgrounds)}, batch_size: {batch_size}, memory_slots: {memory_slots}")
    backgrounds = [backgrounds[idx:idx+batch_size] for idx in range(0, len(backgrounds), batch_size)]
    device = retriever.device
    ret = []

    for background in tqdm(backgrounds, dynamic_ncols=True, desc="> Preparing retrieval embeds"):
        tokenized_retrieval_text = tokenizer(
            background, 
            max_length=180,
            padding=True, truncation=True, return_tensors="pt"
        )
        
        if memory_slots:
            embeds = get_memory_slots(
                model = retriever,
                input_ids = tokenized_retrieval_text['input_ids'].to(device),
                attention_mask = tokenized_retrieval_text['attention_mask'].to(device),
            ).cpu()
        else:
            ## return a torch tensor of shape [batch_size,d_model]
            embeds = get_retrieval_embeds(
                model = retriever,
                input_ids = tokenized_retrieval_text['input_ids'].to(device),
                attention_mask = tokenized_retrieval_text['attention_mask'].to(device),
            ).cpu()

        embeds = [embeds[idx] for idx in range(embeds.shape[0])]
        ret.extend(embeds)

    return ret

@torch.no_grad()
def llm_for_open_generation(
    args, llm, llm_tokenizer,
    prompts,
    retrieval_embeds,
    batch_size = 4,
    enable_progress_bar = True,
    ret_embeddings = None,
):
    generated_answers = []
    total_test_number = len(prompts)
    device = llm.device
    batched_prompts = [prompts[idx:idx+batch_size] for idx in range(0, len(prompts), batch_size)]
    if retrieval_embeds is not None:
        batched_retrieval_embeds = [retrieval_embeds[idx:idx+batch_size] for idx in range(0,len(retrieval_embeds),batch_size)]
        assert len(batched_prompts) == len(batched_retrieval_embeds)
    
    progress_bar = tqdm(range(total_test_number), ncols=60, disable= not enable_progress_bar, desc="> generating answers")
    for batch_idx in range(len(batched_prompts)):
        prompt = batched_prompts[batch_idx]
        tokenized_propmt = llm_tokenizer(prompt,padding='longest',return_tensors='pt')
        input_ids = tokenized_propmt.input_ids.to(device)
        attention_mask = tokenized_propmt.attention_mask.to(device)
        # stopping_criteria = stop_sequences_criteria(llm_tokenizer, input_ids.shape[1], input_ids.shape[0])
        retrieval_kwargs = {}

        if retrieval_embeds is not None:
            embeds = batched_retrieval_embeds[batch_idx]
            embeds = [x for y in embeds for x in y]
            embeds = torch.stack(embeds).to(device)
            retrieval_kwargs['retrieval_embeds'] = embeds

        ## actual computation
        generated_output = llm.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            **retrieval_kwargs,
        )
        ## because HF generate with inputs_embeds would not return prompt
        input_length = 0 if retrieval_kwargs else input_ids.shape[1]
        results = tokenizer.batch_decode(generated_output[:, input_length:], skip_special_tokens=True)
        generated_answers.extend(results)

        progress_bar.update(batch_size)

    generated_answers = [x.strip() for x in generated_answers]
    return generated_answers

def format_one_example(
    sample, include_answer, use_rag, icae_mem_size, task_type
):
    
    question = sample['question']
    prompt_dict = dict(question=question)
    prompt = PROMPT_TEMPLATES[task_type].format_map(prompt_dict).strip()
    backgrounds = []

    if use_rag:
        backgrounds = sample['background'] ## a list
        background_prompts = ""
        
        for background in backgrounds:
            if icae_mem_size > 0:
                background_prompts += "".join([XRAG_TOKEN] * icae_mem_size) + " "            
            else:
                background_prompts += background + " "
        background_prompts = background_prompts.strip()
        prompt = BACKGROUND_PROMPT_TEMPLATE.format_map(dict(background=background_prompts)) + prompt

    return prompt, backgrounds

def prepare_prompts(
    test_data,
    task_type,
    tokenizer,
    n_shot = 0,
    use_rag = False,
    icae_mem_size=0,
    chat_format_fn = None,
):
    splitter = "\n\n"
    prompts = []
    backgrounds = []
    for idx, sample in enumerate(test_data):
        prompt_start  = get_start_prompt(task_type, use_rag=use_rag, sample=sample) 
        prompt_end, background = format_one_example(
            sample,
            include_answer=False,
            use_rag=use_rag,
            icae_mem_size=icae_mem_size,
            task_type=task_type
        )

        prompt = prompt_start + splitter + prompt_end

        if chat_format_fn is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_format_fn(messages, tokenizer) + " The answer is:"
    
        prompts.append(prompt)
        backgrounds.append(background)

    print("**"*20,"show one example","**"*20)
    print(prompts[0])
    print("**"*20,"show one example","**"*20)

    return prompts, backgrounds

def load_dataset(data, use_rag, args):
        
    test_path = f"{args.data_root}/eval/{data}/{args.eval_file_name}"
    test_data = None
    if os.path.isfile(test_path):
        test_data = get_jsonl(test_path)

    if use_rag:
        print("> [ES] loading retrieval data for RAG (use_rag) setting")
        test_retrieval_path = f"{args.data_root}/eval/{data}/retrieval/{args.retrieval_prefix}/{args.retrieval_file_name}"
        test_retrieval = get_jsonl(test_retrieval_path)
        if args.max_test_samples is not None:
            test_retrieval = test_retrieval[:args.max_test_samples]

        assert len(test_retrieval) == len(test_data)

        for idx in range(len(test_data)):
            test_data[idx]['background'] = [test_retrieval[idx]['topk'][rank]['text'] for rank in args.retrieval_topk]
        
        if args.tf_idf_topk > 0:
            assert args.use_rag
            documents = [x['background'][0] for x in test_data]
            keywords = keyword_extraction_with_tfidf(documents,topk=args.tf_idf_topk)
            for idx in range(len(test_data)):
                test_data[idx]['background'] = [keywords[idx]]
        
        if args.retriever_name_or_path is not None and args.retriever_name_or_path.lower() == "intfloat/e5-large-v2":
            for idx in range(len(test_data)):
                test_data[idx]['background'] = ["passage: " + x for x in test_data[idx]['background']]

    return test_data



if __name__ == "__main__":
    args = parse_args()

    if args.checkpoint_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint_path,
            padding_side='left',
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            padding_side='left',
        )

    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ## load retriever and retriever_tokenizer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ## prepare prompt
    test_data = load_dataset(
        args.data,
        args.use_rag,
        args,
    )

    if args.max_test_samples is not None:
        test_data = test_data[:args.max_test_samples]

    chat_format_fn = eval(f"create_prompt_with_{args.chat_format}_chat_format")
    prompts, backgrounds = prepare_prompts(
        test_data = test_data,
        task_type = args.task_type,
        tokenizer = tokenizer,
        n_shot = args.n_shot,
        use_rag = args.use_rag,
        icae_mem_size = args.icae_mem_size,
        chat_format_fn = chat_format_fn, 
    )
    
    avg_prompt_length = tokenizer(prompts, return_length=True).length
    avg_prompt_length = sum(avg_prompt_length)/len(avg_prompt_length)

    if args.use_rag and args.chat_format == 'icae':
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

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

        config = CONFIG_CLASS.from_pretrained(args.model_name_or_path,
                                            torch_dtype=torch.bfloat16)                                     

        model = ICAE(config, 
                    args.model_name_or_path,
                    lora_config,
        )
        retriever = "ICAE_mem"

        retriever_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if isinstance(retriever_tokenizer, LlamaTokenizer) or isinstance(retriever_tokenizer, LlamaTokenizerFast) or isinstance(retriever_tokenizer, PreTrainedTokenizerFast):
            num_added_tokens = retriever_tokenizer.add_special_tokens({
                "pad_token": "<pad>", ## same as eos, it will be ignored anyway
            })

        xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
        model.set_xrag_token_id(xrag_token_id)

        s = time.time()
        model.resize_token_embeddings(len(tokenizer))
        print(f"> {time.time()-s:.3f}s seconds took for resizing token embedding")
    else:
        retriever = None
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype = torch.bfloat16,
            low_cpu_mem_usage = True,
            device_map='auto',
        )

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if args.checkpoint_path is not None:
        print(f"> [ES] loading checkpoint from {args.checkpoint_path}")

        if os.path.exists(os.path.join(args.checkpoint_path, "pytorch_model.bin.index.json")):
            print(f"> [ES] loading checkpoint from {args.checkpoint_path} by from_pretrained")
            config = CONFIG_CLASS.from_pretrained(args.checkpoint_path,
                                          retriever_hidden_size=retriever_hidden_size,
                                          icae_mem_size=icae_mem_size,
                                          torch_dtype=torch.bfloat16
            )                                     
            config.init_train_args(args)

            model = ICAE.from_pretrained(
                args.checkpoint_path,
                config=config,
                torch_dtype = torch.bfloat16,
                use_flash_attention_2=True,
            )
            model.set_xrag_token_id(xrag_token_id)
            model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            if os.path.exists(os.path.join(args.checkpoint_path, "ckpt.pth")):
                ckpt_path = os.path.join(args.checkpoint_path, "ckpt.pth")
            else:
                ckpt_path = hf_hub_download(
                    repo_id=args.checkpoint_path,
                    filename="ckpt.pth",
                    repo_type="model"
                )
   
            ckpt = torch.load(ckpt_path, map_location="cpu")
            ## Update for those parameters that are in the ckpt.keys()
            model_state_dict = model.state_dict()

            for name, param in ckpt.items():
                if name in model_state_dict:
                    if model_state_dict[name].shape != param.shape:
                        print(f"> Warning: parameter '{name}' shape mismatch. Skipping.")
                        exit()
                    else:
                        model_state_dict[name].copy_(param.to(torch.bfloat16))
                else:
                    print(f"> Warning: parameter '{name}' not found in model state_dict. Skipping.")
                    exit()
            model.load_state_dict(model_state_dict)
            for param in model.parameters():
                param.data = param.data.to(torch.bfloat16)
    else:
        print("**"*20)
        print(f"> [ES] no checkpoint path is provided, loading from {args.model_name_or_path}")
        print("**"*20)

    model.eval()

    retrieval_embeds = None
    if args.use_rag and args.chat_format == "icae":
        num_samples = len(backgrounds)
        original_orders = []
        for idx, background in enumerate(backgrounds):
            original_orders.extend(
                [idx] * len(background)
            )
        backgrounds = [x for y in backgrounds for x in y]
        print(f"Preparing document embedding with {args.retriever_name_or_path}...")
        _retrieval_embeds = prepare_retrieval_embeds(
            backgrounds,
            model,
            retriever_tokenizer,
            memory_slots=True,
        )
        assert len(_retrieval_embeds) == len(original_orders), f"{len(_retrieval_embeds)} != {len(original_orders)}"

        retrieval_embeds = [[] for _ in range(num_samples)]
        for id, embeds in zip(original_orders, _retrieval_embeds):
            retrieval_embeds[id].append(embeds)

    if retriever is not None:
        assert XRAG_TOKEN in tokenizer.get_vocab() 
        model.set_xrag_token_id(tokenizer.convert_tokens_to_ids(XRAG_TOKEN))

    generated_results = llm_for_open_generation(
        args = args,
        llm = model,
        llm_tokenizer = tokenizer,
        prompts = prompts,
        retrieval_embeds = retrieval_embeds,
        batch_size = args.eval_batch_size,
        enable_progress_bar = args.enable_progress_bar,
    )

    answers = [x['answer'] for x in test_data]
    if args.save_results:
        dict_list = []
        for d, gen in zip(test_data, generated_results):
            gen_ = gen.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").strip()
            dict_list.append({
                "id":d['id'],
                "question":d['question'],
                "answer":d['answer'],
                "pred":gen_,
            })  
        with open(args.results_path, 'wt') as outf:
            json.dump(dict_list, outf, indent=4)

    if args.eval_metrics == 'substring_match':
        score, score_per_sample = get_substring_match_score(generated_results, answers)
    elif args.eval_metrics == 'fact_checking_acc':
        score, score_per_sample = eval_fact_checking(generated_results, answers)
    elif args.eval_metrics == 'truthfulqa_f1_rl':
        f1, rl, f1_scores, rl_scores = eval_truthfulqa(generated_results, answers)
        score = f"{f1}-{rl}"
        score_per_sample = [(f1_score, rl_score) for f1_score, rl_score in zip(f1_scores, rl_scores)]

    result_dict = {
        "checkpoint": args.checkpoint_path,
        "dataset": args.data,
        "batch_size": args.eval_batch_size,
        "use_rag": args.use_rag,
        "avg_prompt_length": avg_prompt_length,
        "model": args.model_name_or_path,
        f"{args.eval_metrics}": score,
    }

    if args.retriever_name_or_path is not None:
        result_dict['retriever'] = args.retriever_name_or_path
    print(json.dumps(result_dict, indent=4))

    generated_results = []
    answers = []
    for pred in dict_list:
        generated_results.append(pred['pred'])
        answers.append(pred['answer'])
    score, score_per_sample_list = get_substring_match_score(generated_results, answers)

    print(f"{score:.4f}", end=" ")
    line = f"{args.checkpoint_path} {args.data} {score:.4f}\n"
    with open("ft_result_logs.txt", "a") as f:
        f.write(line)
        f.write("#" * 20 + "\n")