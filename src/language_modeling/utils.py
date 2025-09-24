import os
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm

from src.eval.utils import get_substring_match_score

def get_nll_loss(logits, labels, vocab_size):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :vocab_size].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def get_kl_loss(teacher_logits, teacher_labels, student_logits, student_labels, temperature, vocab_size):
    teacher_logits = teacher_logits[:, :, :vocab_size]
    student_logits = student_logits[:, :, :vocab_size]
    ## make sure the teacher_logits and student_logits have the same shape

    loss_fct = nn.KLDivLoss(reduction="batchmean")

    ## only compute loss in the completion part, not propmt
    
    student_mask = (student_labels!=-100).unsqueeze(-1).expand_as(student_logits) ## batch_size, num_tokens, vocab_size
    student_logits_selected = torch.masked_select(student_logits, student_mask).view(-1, vocab_size)

    teacher_mask = (teacher_labels != -100).unsqueeze(-1).expand_as(teacher_logits)
    teacher_logits_selected = torch.masked_select(teacher_logits, teacher_mask).view(-1, vocab_size)

    assert teacher_logits_selected.shape == student_logits_selected.shape, (f"The shape of teacher logits is {teacher_logits_selected.shape}, while that of student is {student_logits_selected.shape}")

    kl_loss = loss_fct(
        F.log_softmax(student_logits_selected / temperature, dim=-1),
        F.softmax(teacher_logits_selected / temperature, dim=-1),
    ) * temperature ** 2
    
    return kl_loss

def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = copy.copy(input_ids)

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    # attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids,
        'labels': labels,
        # 'attention_mask': attention_mask.flatten(),
    }

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    prompt = example['prompt']
    completion = example['completion']

    background = example['background']
    background_embedding = example['background_embedding']

    prompt = f"Background: {background}\n\n{prompt}"

    prompt = prompt.strip()
    completion = completion.strip()

    if not prompt.endswith((' ', '\n', '\t')) and not completion.startswith((' ', '\n', '\t')):
        example_text = prompt + ' ' + completion
    else:
        example_text = prompt + completion

    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = copy.copy(input_ids)
    tokenized_prompt_length = tokenizer(prompt, max_length=max_seq_length, truncation=True,return_length=True).length
    # mask the prompt part for avoiding loss
    labels[:tokenized_prompt_length] = [-100]*tokenized_prompt_length
    # attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids,
        'labels': labels,
        "background_embedding":background_embedding,
        # 'attention_mask': attention_mask.flatten(),
    }



def save_with_accelerate(accelerator, model, tokenizer, output_dir, args, save_projector_only=False):
    unwrapped_model = accelerator.unwrap_model(model)
    if save_projector_only:    
        # params_to_save = {
        #     n:p.float() for n,p in unwrapped_model.named_parameters() 
        #     if any(
        #         sub_string in n 
        #         for sub_string in ['embed_tokens', 'projector', 'lm_head', 'rqvae']
        #         )
        #     }
        # save weights that are required grad
        
        params_to_save = {
            n:p.float() for n,p in unwrapped_model.named_parameters()
            if p.requires_grad
        }

        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(params_to_save, os.path.join(output_dir,'ckpt.pth'))
            unwrapped_model.config.save_pretrained(output_dir)

    else:    
        # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
        # Otherwise, sometimes the model will be saved with only part of the parameters.
        # Also, accelerator needs to use the wrapped model to get the state_dict.
        state_dict = accelerator.get_state_dict(model)

        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict,
            safe_serialization=False, ## safetensors is buggy for now
        )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

XRAG_TOKEN = "<xRAG>" 

ParaphraseInstructions = [
    'Background: {xrag_token} means the same as',
    "Background: {xrag_token} Can you put the above sentences in your own terms?",
    "Background: {xrag_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {xrag_token}\n(2)",
    "Background: {xrag_token} is a paraphrase of what?",
    "Background: {xrag_token} Could you give me a different version of the background sentences above?",
    "In other words, background: {xrag_token} is just another way of saying:",
    "You're getting across the same point whether you say background: {xrag_token} or",
    "Background: {xrag_token} After uppacking the ideas in the background information above, we got:",
    "Background: {xrag_token} Please offer a restatement of the background sentences I've just read.",
    "Background: {xrag_token}, which also means:",
    "Strip away the mystery, and you'll find background: {xrag_token} is simply another rendition of:",
    "The essence of background: {xrag_token} is captured again in the following statement:",
]

def get_memory_slots(model, input_ids, attention_mask=None):
    with torch.no_grad():
        memory_slots = model.icae_forward(  #june edit
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
    memory_slots = memory_slots.reshape(-1, model.mem_size, memory_slots.shape[-1])
    return memory_slots 

def get_retrieval_embeds(model, input_ids, attention_mask=None):
    with torch.no_grad():
        embeds = model.get_doc_embedding(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
    embeds = embeds.reshape(-1, embeds.shape[-1])
    return embeds 

def calculate_grad_norm(model, norm_type=2):  
    total_norm = 0  
    for p in model.parameters():  
        if p.grad is not None:  
            param_norm = p.grad.data.norm(norm_type)  
            total_norm += param_norm.item() ** norm_type  
    total_norm = total_norm ** (1. / norm_type)  
    return total_norm


def find_matched_index(main_seq, sub_seq):  
    # Lengths of the sequences  
    assert len(sub_seq)>0 and len(main_seq)>0, f"the input should not be empty, however {sub_seq=}\n {main_seq=}"
    main_len = len(main_seq)  
    sub_len = len(sub_seq)  
  
    # Early exit if sub_seq is longer than main_seq  
    if sub_len > main_len:  
        return -1  
  
    # Variable to keep track of the last index of a match  
    last_index = -1  
  
    # Iterate through main_seq to find sub_seq  
    for i in range(main_len - sub_len + 1):  
        # Check if the slice of main_seq matches sub_seq  
        if main_seq[i:i+sub_len] == sub_seq:  
            # Update the last_index to the current position  
            last_index = i  
  
    # Return the last index found or -1 if not found  
    return last_index  


def collator(        
        samples,
        task_type = None,
        llm_tokenizer = None,
        retriever_tokenizer = None,
        retrieval_context_length = 180,
    ):
    """
    collate tokenized input_ids and labels with left and right side padding supported
    
    Args:
        samples (dict): a dict contains input_ids, labels and maybe retrieval_text
        llm_tokenizer: tokenizer for llm
        retriever_tokenizer: tokenizer for retriever
        retrieval_context_length: max length for the retrieved passages
    
    Returns:
        xrag_input_ids: input_ids with xrag_token_id (xrag_labels,xrag_attention_mask)
        input_ids: input_ids for llm without xrag_token_id, vanilla rag (labels,attention_mask)
        retriever_input_ids: input_ids for retriever (retriever_attention_mask)

    """
    ret = dict()
    ids = [x['id'] for x in samples]
    ret["ids"] = ids
    if task_type == 'finetune':
        answers = [x['answers'] for x in samples]
        ret["answers"] = answers
    
        if 'random_background_id' in samples[0].keys() :
            random_background_ids = [x['random_background_id'] for x in samples]
            ret["random_background_ids"] = random_background_ids

        if 'hn_background_id' in samples[0].keys() :
            hn_background_ids = [x['hn_background_id'] for x in samples]
            ret["hn_background_ids"] = hn_background_ids

        if 'gt_background_id' in samples[0].keys() :
            gt_background_ids = [x['gt_background_id'] for x in samples]
            ret["gt_background_ids"] = gt_background_ids

    def padding(input_ids, labels=None, padding_side='right'):
        """
        batch padding
        """

        def _padding(ids,padding_value,padding_side='right'):
            if padding_side == 'right':
                return torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=padding_value)
            elif padding_side == 'left':
                flipped_ids = [torch.flip(x, dims=[0]) for x in ids]  
                return torch.flip(
                    torch.nn.utils.rnn.pad_sequence(flipped_ids, batch_first=True, padding_value=padding_value),
                    dims=[1],
                )
        
        input_ids = _padding(input_ids, padding_value=llm_tokenizer.pad_token_id, padding_side=padding_side)
        attention_mask = (input_ids != llm_tokenizer.pad_token_id).long()

        if labels is not None:
            labels = _padding(labels, padding_value=-100, padding_side=padding_side)

        return input_ids, attention_mask, labels

    xrag_input_ids, xrag_attention_mask, xrag_labels = padding(
        input_ids=[x['xrag_input_ids'] for x in samples],
        labels=[x['xrag_labels'] for x in samples] if 'xrag_labels' in samples[0].keys() else None,
        padding_side=llm_tokenizer.padding_side
    )
    ret["xrag_input_ids"] = xrag_input_ids
    ret["xrag_attention_mask"] = xrag_attention_mask
    ret["xrag_labels"] = xrag_labels

    #########################################################################################################

    if samples[0].get('random_xrag_input_ids') is not None:
        random_xrag_input_ids, random_xrag_attention_mask, random_xrag_labels = padding(
            input_ids=[x['random_xrag_input_ids'] for x in samples],
            labels=[x['random_xrag_labels'] for x in samples] if 'random_xrag_labels' in samples[0].keys() else None,
            padding_side=llm_tokenizer.padding_side
        )
        ret["random_xrag_input_ids"] = random_xrag_input_ids
        ret["random_xrag_attention_mask"] = random_xrag_attention_mask
        ret["random_xrag_labels"] = random_xrag_labels


    if samples[0].get('hn_xrag_input_ids') is not None:
        hn_xrag_input_ids, hn_xrag_attention_mask, hn_xrag_labels = padding(
            input_ids=[x['hn_xrag_input_ids'] for x in samples],
            labels=[x['hn_xrag_labels'] for x in samples] if 'hn_xrag_labels' in samples[0].keys() else None,
            padding_side=llm_tokenizer.padding_side
        )
        ret["hn_xrag_input_ids"] = hn_xrag_input_ids
        ret["hn_xrag_attention_mask"] = hn_xrag_attention_mask
        ret["hn_xrag_labels"] = hn_xrag_labels


    if samples[0].get('gt_xrag_input_ids') is not None:
        gt_xrag_input_ids, gt_xrag_attention_mask, gt_xrag_labels = padding(
            input_ids=[x['gt_xrag_input_ids'] for x in samples],
            labels=[x['gt_xrag_labels'] for x in samples] if 'gt_xrag_labels' in samples[0].keys() else None,
            padding_side=llm_tokenizer.padding_side
        )
        ret["gt_xrag_input_ids"] = gt_xrag_input_ids
        ret["gt_xrag_attention_mask"] = gt_xrag_attention_mask
        ret["gt_xrag_labels"] = gt_xrag_labels


    #########################################################################################################

    if 'retriever_input_text' in samples[0].keys():
        retriever_input_text = [x['retriever_input_text'] for x in samples]
        assert isinstance(retriever_input_text[0],list)
        retriever_input_text = [x for y in retriever_input_text for x in y]
        ## handling different retriever tokenization problem
        if retriever_tokenizer.name_or_path == "intfloat/e5-large-v2":
            retriever_input_text = ["passage: "+x for x in retriever_input_text]
        elif retriever_tokenizer.name_or_path == 'intfloat/e5-mistral-7b-instruct':
            retriever_input_text = [x + retriever_tokenizer.eos_token for x in retriever_input_text]

        tokenized_retrieval_text = retriever_tokenizer(
            retriever_input_text, 
            max_length=retrieval_context_length,
            padding=True, truncation=True, return_tensors="pt"
        )
        
        ret['retriever_input_ids'] = tokenized_retrieval_text['input_ids']
        ret['retriever_attention_mask'] = tokenized_retrieval_text['attention_mask']
    
    if samples[0].get('random_retriever_input_text') is not None:
        random_retriever_input_text = [x['random_retriever_input_text'] for x in samples]
        assert isinstance(random_retriever_input_text[0],list)
        random_retriever_input_text = [x for y in random_retriever_input_text for x in y]

        tokenized_random_retrieval_text = retriever_tokenizer(
            random_retriever_input_text, 
            max_length=retrieval_context_length,
            padding=True, truncation=True, return_tensors="pt"
        )
        
        ret['random_retriever_input_ids'] = tokenized_random_retrieval_text['input_ids']
        ret['random_retriever_attention_mask'] = tokenized_random_retrieval_text['attention_mask']

    if samples[0].get('hn_retriever_input_text') is not None:
        hn_retriever_input_text = [x['hn_retriever_input_text'] for x in samples]
        assert isinstance(hn_retriever_input_text[0],list)
        hn_retriever_input_text = [x for y in hn_retriever_input_text for x in y]

        tokenized_hn_retrieval_text = retriever_tokenizer(
            hn_retriever_input_text, 
            max_length=retrieval_context_length,
            padding=True, truncation=True, return_tensors="pt"
        )
        
        ret['hn_retriever_input_ids'] = tokenized_hn_retrieval_text['input_ids']
        ret['hn_retriever_attention_mask'] = tokenized_hn_retrieval_text['attention_mask']


    if samples[0].get('gt_retriever_input_text') is not None:
        gt_retriever_input_text = [x['gt_retriever_input_text'] for x in samples]
        assert isinstance(gt_retriever_input_text[0],list)
        gt_retriever_input_text = [x for y in gt_retriever_input_text for x in y]

        tokenized_gt_retrieval_text = retriever_tokenizer(
            gt_retriever_input_text, 
            max_length=retrieval_context_length,
            padding=True, truncation=True, return_tensors="pt"
        )
        
        ret['gt_retriever_input_ids'] = tokenized_gt_retrieval_text['input_ids']
        ret['gt_retriever_attention_mask'] = tokenized_gt_retrieval_text['attention_mask']

    if 'input_ids' in samples[0].keys():
        input_ids = [x['input_ids'] for x in samples]
        labels =    [x['labels'] for x in samples]

        input_ids, attention_mask, labels = padding(input_ids, labels, padding_side=llm_tokenizer.padding_side)

        ret['input_ids'] = input_ids
        ret['attention_mask'] = attention_mask
        ret['labels'] = labels

        #########################################################################################################

    if samples[0].get('random_input_ids') is not None:
    # if 'random_input_ids' in samples[0].keys():
        random_input_ids = [x['random_input_ids'] for x in samples]
        random_labels = [x['random_labels'] for x in samples]
        random_input_ids, random_attention_mask, random_labels = padding(random_input_ids, random_labels, padding_side=llm_tokenizer.padding_side)
        
        ret['random_input_ids'] = random_input_ids
        ret['random_attention_mask'] = random_attention_mask
        ret['random_labels'] = random_labels

    if samples[0].get('hn_input_ids') is not None:
    # if 'hn_input_ids' in samples[0].keys():
        hn_input_ids = [x['hn_input_ids'] for x in samples]
        hn_labels = [x['hn_labels'] for x in samples]
        hn_input_ids, hn_attention_mask, hn_labels = padding(hn_input_ids, hn_labels, padding_side=llm_tokenizer.padding_side)
        
        ret['hn_input_ids'] = hn_input_ids
        ret['hn_attention_mask'] = hn_attention_mask
        ret['hn_labels'] = hn_labels


    if samples[0].get('gt_input_ids') is not None:
        gt_input_ids = [x['gt_input_ids'] for x in samples]
        gt_labels = [x['gt_labels'] for x in samples]
        gt_input_ids, gt_attention_mask, gt_labels = padding(gt_input_ids, gt_labels, padding_side=llm_tokenizer.padding_side)
        
        ret['gt_input_ids'] = gt_input_ids
        ret['gt_attention_mask'] = gt_attention_mask
        ret['gt_labels'] = gt_labels


    if 'none_input_ids' in samples[0].keys():
        none_input_ids = [x['none_input_ids'] for x in samples]
        none_labels = [x['none_labels'] for x in samples]
        none_input_ids, none_attention_mask, none_labels = padding(none_input_ids, none_labels, padding_side=llm_tokenizer.padding_side)

        ret['none_input_ids'] = none_input_ids
        ret['none_attention_mask'] = none_attention_mask
        ret['none_labels'] = none_labels

        adaptive_input_ids = [
        x['none_input_ids'] if x['closed_book_correct']
        else x['gt_input_ids']
        for x in samples
        ]
        adaptive_labels = [
            x['none_labels'] if x['closed_book_correct']
            else x['gt_labels']
            for x in samples
        ]
        adaptive_input_ids, adaptive_attention_mask, adaptive_labels = padding(adaptive_input_ids, adaptive_labels, padding_side=llm_tokenizer.padding_side)

        ret['adaptive_input_ids'] = adaptive_input_ids
        ret['adaptive_attention_mask'] = adaptive_attention_mask
        ret['adaptive_labels'] = adaptive_labels

        #########################################################################################################

    return ret


@torch.no_grad()
def validate_during_pretrain(model, dataloader, device, vocab_size, accelerator=None):
    model.eval()
    total_loss = []

    for batch in tqdm(dataloader, desc="> validating", dynamic_ncols=True):
        retrieval_kwargs = {}
        if accelerator is not None:
            memory_slots = model.module.icae_forward(
                        input_ids = batch['retriever_input_ids'],
                        attention_mask = batch['retriever_attention_mask'],
                    )
        else:
            memory_slots = model.icae_forward(
                        input_ids = batch['retriever_input_ids'].to(device),
                        attention_mask = batch['retriever_attention_mask'].to(device),
                    )

            retrieval_kwargs['retrieval_embeds'] = memory_slots
        outputs = model(
            input_ids = batch['xrag_input_ids'].to(device),
            attention_mask = batch['xrag_attention_mask'].to(device),
            **retrieval_kwargs,
        )

        nll_loss = get_nll_loss(
            labels = batch['xrag_labels'],
            logits = outputs.logits,
            vocab_size = vocab_size,
        )
        total_loss.append(nll_loss.item())

    model.train()
    if accelerator is not None:
        if accelerator.use_distributed and accelerator.num_processes>1:
            all_ranks_objects = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(all_ranks_objects,total_loss)
            total_loss = [x for y in all_ranks_objects for x in y]
    ppl = torch.exp(torch.tensor(sum(total_loss)/len(total_loss)))
    return ppl


@torch.no_grad()
def validate_during_finetune(model, dataloader, accelerator ,retriever, tokenizer, output_dir, id2indices=None, id2embed_for_retriever=None, ret_embeddings=None, retriever_hidden_size=4096):
    model.eval()
    local_results = []
    for batch in tqdm(dataloader, desc="> validating", dynamic_ncols=True):
        retrieval_kwargs = {}
        base_model = model.module if hasattr(model, 'module') else model
        memory_slots = base_model.icae_forward(
                    input_ids = batch['retriever_input_ids'],
                    attention_mask = batch['retriever_attention_mask'],
                )
        
        retrieval_kwargs['retrieval_embeds'] = memory_slots

        input_ids = batch['xrag_input_ids'][:, :-1]
        attention_mask = batch['xrag_attention_mask'][:, :-1]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                max_new_tokens=30,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
                **retrieval_kwargs,
            )

        cur_results=[]
        results = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        input_texts = tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True
        )

        for idx, pred_text, input_text, answer in zip(batch['ids'], results, input_texts, batch['answers']):
            cur_results.append({
                'id': idx,
                'pred':  pred_text.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").strip(),
                'answer': answer,
                'prompt': input_text.strip(), 
            })

        local_results.extend(cur_results)

    accelerator.wait_for_everyone()
    all_results = [None] * accelerator.num_processes
    dist.all_gather_object(all_results, local_results)

    model.train()
    score = 0
    if accelerator.is_main_process:
        flat_results = sum(all_results, [])
        all_answers = [x['answer'] for x in flat_results]
        all_preds = [x['pred'] for x in flat_results]
        score, _= get_substring_match_score(all_preds, all_answers)  
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "validation_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(flat_results, f, ensure_ascii=False, indent=4)

    return score


def update_raw_datasets(criteria, ctx_select):
    def update_fn(example):
        if example[criteria]:
            example['background'] = example[f'{ctx_select}_background']
            example['id'] = example[f'{ctx_select}_background_id']
        return example
    return update_fn



def change_inference_form_batch_leftpad(
    xrag_input_ids: torch.LongTensor,        # (B, S)
    xrag_attention_mask: torch.LongTensor,   # (B, S)
    xrag_labels: torch.LongTensor,           # (B, S)
    pad_token_id: int = 0
):
    """
    - 각 배치 샘플에 대해 labels == -100 위치의 토큰들만 추출
    - 시퀀스 길이가 제각각이므로 왼쪽으로 패딩 후 반환

    Returns:
      context_ids:   (B, L_max) LongTensor  # left-padded
      context_mask:  (B, L_max) LongTensor  # left-padded
    """
    batch_ctx_ids   = []
    batch_ctx_masks = []

    # 1) 각 샘플마다 context 부분만 뽑아서 리스트에 저장
    for ids, mask, labels in zip(xrag_input_ids, xrag_attention_mask, xrag_labels):
        keep = labels == -100
        ctx_ids  = ids[keep]    # (n_i,)
        ctx_mask = mask[keep]   # (n_i,)
        batch_ctx_ids.append(ctx_ids)
        batch_ctx_masks.append(ctx_mask)

    # 2) 가장 긴 시퀀스 길이 구하기
    lengths = [seq.size(0) for seq in batch_ctx_ids]
    L_max    = max(lengths)

    # 3) 왼쪽 패딩 수 계산해서, 각 시퀀스 앞에 pad 채우기
    padded_ids = []
    padded_masks = []
    for seq_ids, seq_mask in zip(batch_ctx_ids, batch_ctx_masks):
        pad_len = L_max - seq_ids.size(0)

        # ids: pad_token_id로, mask: 0으로
        left_ids  = torch.full(
            (pad_len,), pad_token_id,
            dtype=seq_ids.dtype, device=seq_ids.device
        )
        left_mask = torch.zeros(
            (pad_len,), dtype=seq_mask.dtype, device=seq_mask.device
        )

        padded_ids.append(torch.cat([left_ids,  seq_ids],  dim=0))
        padded_masks.append(torch.cat([left_mask, seq_mask], dim=0))

    # 4) 배치 차원으로 쌓기
    context_ids_padded  = torch.stack(padded_ids,  dim=0)  # (B, L_max)
    context_mask_padded = torch.stack(padded_masks, dim=0)  # (B, L_max)

    return context_ids_padded, context_mask_padded