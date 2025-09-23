import torch
import random,copy

from .utils import ParaphraseInstructions, XRAG_TOKEN

def split_background(background,tokenizer,total_max_len,single_max_len,single_min_len=20):
    """
    split a long document into multiple smaller chunks between single_max_len and single_mini_len
    
    Args:
        background: string
    
    Return:
        background: a list of string
    """
    ids = tokenizer(background,add_special_tokens=False,max_length = total_max_len,truncation=True).input_ids
    background = [ids[idx:idx+single_max_len] for idx in range(0,len(ids),single_max_len)]
    assert len(background) >= 1, background
    if len(background[-1]) <= single_min_len and len(background)>1:
        background = background[:-1]
    background = [tokenizer.decode(x) for x in background]
    return background

def _concat_messages(messages, tokenizer, assist_content="content"):
    ## Mistral Chat Format
    message_text = ""
    for message in messages:
        if message["role"] == "user":
            message_text += "[INST] " + message["content"].strip() + " [/INST]"
        elif message["role"] == "assistant":
            message_text += message[assist_content].rstrip() + tokenizer.eos_token
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text

    return message_text

def _encode_chat_format(
        messages,
        tokenizer,
        max_seq_length,
        chat_format='mistral',
        assist_content="content"
    ):
    """
    encode messages to input_ids and make non-assistant part

    Args:
        messages (list): list of dict with 'role' and 'content' field
        tokenizer: llm tokenizer
        max_seq_lengh: maximun context length  
    
    Return:
        input_ids and labels
    """

    example_text = _concat_messages(messages, tokenizer, assist_content=assist_content).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx], tokenizer), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            
            messages_so_far = _concat_messages(messages[:message_idx+1], tokenizer)         

            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]

            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break
    
    # assert tokenizer.eos_token_id in input_ids, input_ids
    return {
        "input_ids":input_ids.flatten(),
        "labels":labels.flatten(),
    }


def encode_with_chat_format_pretrain(
        example,
        tokenizer,
        max_seq_length,
        icae_mem_size,
        chat_format='mistral',
        retrieval_input_text=True,
        ):
    """
    encode messages into input_ids and labels for paraphrase pretrain

    Args:
        example: data sample with 'text' filed
        tokenizer: llm_tokenizer
        max_seq_length: maximun context length
        icae_mem_size: number of tokens for ICAE memory
    
    Return:
        input_ids,labels and retriever_input_text
    """    
    # if tokenizer.eos_token_id not in tokenizer("this is good."+tokenizer.eos_token +'\n').input_ids:
    #     from transformers import AutoTokenizer
    #     new_tokenizer = AutoTokenizer.from_pretrained("allenai/tulu-2-7b")
    #     assert new_tokenizer.eos_token_id in new_tokenizer("this is good."+new_tokenizer.eos_token +'\n').input_ids, 'new_tokenizer'
    #     assert tokenizer.eos_token_id in tokenizer("this is good."+tokenizer.eos_token +'\n').input_ids, 'encode_with_chat_format_pretrain'    
    #     print(new_tokenizer)
    #     print(tokenizer)

    document = example['text'].strip()

    document_list = [document]

    xrag_token = "".join([XRAG_TOKEN] * icae_mem_size)

    instruction = random.choice(ParaphraseInstructions).format_map(dict(xrag_token=xrag_token))

    messages = [
        {"role":"user", "content":instruction},
        {"role":"assistant", "content":document},
    ]

    encoded = _encode_chat_format(messages, tokenizer, max_seq_length, chat_format)

    if retrieval_input_text:
        return {
            "xrag_input_ids": encoded['input_ids'],
            "xrag_labels": encoded['labels'],
            "retriever_input_text": document_list,
        }
    else:
        return {
            "xrag_input_ids": encoded['input_ids'],
            "xrag_labels": encoded['labels'],
        }

def encode_with_chat_format_finetune(
        example, 
        tokenizer,
        max_seq_length,
        icae_mem_size,
        retrieval_context_length = 180,
        use_rag_tuning = True,
        use_retriever_embed = False,
        retriever_tokenizer = None,
        chat_format = 'mistral',
        ctx_set = set(),
    ):

    messages, background, question, answers = example['messages'], example['background'], example['question'], example['answers']
    
    background_instruction = 'Refer to the background document:'
    if 'valid' in example['id']:
        background_instruction = 'Refer to the background document and answer the questions:'

    background_list = [background]

    ret = {'answers': answers}

    if use_rag_tuning and use_retriever_embed:
        sharded_background = []
        num_split = 0
        for _background  in background_list:
            _sharded_background = split_background(_background, retriever_tokenizer, total_max_len=max_seq_length, single_max_len=retrieval_context_length)
            sharded_background.extend(_sharded_background)
            num_split += len(_sharded_background)
        ret['retriever_input_text'] = sharded_background

    else:
        num_split = 1

    #########################################################################################################

    if 'random' in ctx_set:
        random_background = example['random_background']

        background_list = [random_background]

        if use_rag_tuning and use_retriever_embed:
            sharded_background = []
            random_num_split = 0
            for _background in background_list:
                _sharded_background = split_background(_background, retriever_tokenizer, total_max_len=max_seq_length, single_max_len=retrieval_context_length)
                sharded_background.extend(_sharded_background)
                random_num_split += len(_sharded_background)
            ret['random_retriever_input_text'] = sharded_background
        else:
            random_num_split = 1

    if 'hn' in ctx_set:
        hn_background = example['hn_background']

        background_list = [hn_background]

        if use_rag_tuning and use_retriever_embed:
            sharded_background = []
            hn_num_split = 0
            for _background in background_list:
                _sharded_background = split_background(_background, retriever_tokenizer, total_max_len=max_seq_length, single_max_len=retrieval_context_length)
                sharded_background.extend(_sharded_background)
                hn_num_split += len(_sharded_background)
            ret['hn_retriever_input_text'] = sharded_background
        else:
            hn_num_split = 1

    if 'gt' in ctx_set:
        gt_background = example['gt_background']

        background_list = [gt_background]

        if use_rag_tuning and use_retriever_embed:
            sharded_background = []
            gt_num_split = 0
            for _background in background_list:
                _sharded_background = split_background(_background, retriever_tokenizer, total_max_len=max_seq_length, single_max_len=retrieval_context_length)
                sharded_background.extend(_sharded_background)
                gt_num_split += len(_sharded_background)
            ret['gt_retriever_input_text'] = sharded_background
        else:
            gt_num_split = 1


    if use_rag_tuning:
        _messages = copy.deepcopy(messages)
        xrag_tokens = "".join([XRAG_TOKEN] * icae_mem_size * num_split)

        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                _messages[idx]['content'] = f"{background_instruction} {xrag_tokens}\n\n" + messages[idx]['content']
                break

        encoded = _encode_chat_format(_messages, tokenizer, max_seq_length, chat_format=chat_format)
        ret['xrag_input_ids'] = encoded['input_ids']
        ret['xrag_labels'] = encoded['labels']


        #########################################################################################################
        ## random xrag
        if 'random' in ctx_set:
            _messages = copy.deepcopy(messages)
            xrag_tokens_random = "".join([XRAG_TOKEN] * icae_mem_size * random_num_split)

            for idx in range(len(_messages)):
                if _messages[idx]['role'] == 'user':
                    _messages[idx]['content'] = f"{background_instruction} {xrag_tokens_random}\n\n" + messages[idx]['content']
                    break
            encoded = _encode_chat_format(_messages, tokenizer, max_seq_length, chat_format=chat_format)
            ret['random_xrag_input_ids'] = encoded['input_ids']
            ret['random_xrag_labels'] = encoded['labels']

        ## hn xrag
        if 'hn' in ctx_set:
            _messages = copy.deepcopy(messages)
            xrag_tokens_hn = "".join([XRAG_TOKEN] * icae_mem_size * hn_num_split)

            for idx in range(len(_messages)):
                if _messages[idx]['role'] == 'user':
                    _messages[idx]['content'] = f"{background_instruction} {xrag_tokens_hn}\n\n" + messages[idx]['content']
                    break
            encoded = _encode_chat_format(_messages, tokenizer, max_seq_length, chat_format=chat_format)
            ret['hn_xrag_input_ids'] = encoded['input_ids']
            ret['hn_xrag_labels'] = encoded['labels']

        ## gt xrag
        if 'gt' in ctx_set:
            _messages = copy.deepcopy(messages)
            xrag_tokens_gt = "".join([XRAG_TOKEN] * icae_mem_size * gt_num_split)

            for idx in range(len(_messages)):
                if _messages[idx]['role'] == 'user':
                    _messages[idx]['content'] = f"{background_instruction} {xrag_tokens_gt}\n\n" + messages[idx]['content']
                    break
            encoded = _encode_chat_format(_messages, tokenizer, max_seq_length, chat_format=chat_format)
            ret['gt_xrag_input_ids'] = encoded['input_ids']
            ret['gt_xrag_labels'] = encoded['labels']

        #########################################################################################################
        # rag
        background = example["background"]
        _messages = copy.deepcopy(messages)
        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                _messages[idx]['content'] = f"{background_instruction} {background}\n\n" + messages[idx]['content']
                break
        
        encoded = _encode_chat_format(_messages, tokenizer, max_seq_length, chat_format=chat_format)
        ret['input_ids'] = encoded['input_ids']
        ret['labels'] = encoded['labels']


        #########################################################################################################
        ## random rag
        # if 'random' in ctx_set: ## 무조건 함 일단.
        background_random = example['random_background']
        _messages = copy.deepcopy(messages)
        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                _messages[idx]['content'] = f"{background_instruction} {background_random}\n\n" + messages[idx]['content']
                break
        
        encoded = _encode_chat_format(_messages, tokenizer, max_seq_length, chat_format=chat_format)
        ret['random_input_ids'] = encoded['input_ids']
        ret['random_labels'] = encoded['labels']

        ## hn rag
        # if 'hn' in ctx_set: ## 무조건 함 일단.
        background_hn = example['hn_background']
        _messages = copy.deepcopy(messages)
        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                _messages[idx]['content'] = f"{background_instruction} {background_hn}\n\n" + messages[idx]['content']
                break
        
        encoded = _encode_chat_format(_messages, tokenizer, max_seq_length, chat_format=chat_format)
        ret['hn_input_ids'] = encoded['input_ids']
        ret['hn_labels'] = encoded['labels']

        ## gt rag
        # if 'gt' in ctx_set: ## 무조건 함 일단.
        background_gt = example['gt_background']
        _messages = copy.deepcopy(messages)
        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                _messages[idx]['content'] = f"{background_instruction} {background_gt}\n\n" + messages[idx]['content']
                break
        
        encoded = _encode_chat_format(_messages, tokenizer, max_seq_length, chat_format=chat_format)
        ret['gt_input_ids'] = encoded['input_ids']
        ret['gt_labels'] = encoded['labels']

        ## none rag (closed)
        if 'none' in ctx_set:
            _messages = copy.deepcopy(messages)
            encoded = _encode_chat_format(_messages, tokenizer, max_seq_length, chat_format=chat_format)
            ret['none_input_ids'] = encoded['input_ids']
            ret['none_labels'] = encoded['labels']

        #########################################################################################################

    return ret


QA_PROMPT = "Q: {question}?\nA: {answer}"
RAG_QA_PROMPT = "Background: {background}\n\n" + QA_PROMPT

FACT_CHECKING_PROPMT = "Claim: {question}\nAnswer: {answer}"
RAG_FACT_CHECKING_PROPMT = "Background: {background}\n\n" + FACT_CHECKING_PROPMT

MULTIPLE_CHOICE_PROMPT = "Question: {question}\nAnswer: {answer}"
RAG_MULTIPLE_CHOICE_PROMPT = "Background: {background}\n\n" + MULTIPLE_CHOICE_PROMPT


PROMPT_TEMPLATES = {
    "open_qa":{True:RAG_QA_PROMPT, False:QA_PROMPT},
    'fact_checking':{True:RAG_FACT_CHECKING_PROPMT, False:FACT_CHECKING_PROPMT},
    'multiple_choice':{True:RAG_MULTIPLE_CHOICE_PROMPT, False:MULTIPLE_CHOICE_PROMPT},
}

def get_start_prompt(task_type, include_retrieval):
    if task_type == 'open_qa':
        return {
            True: "Refer to the background document and answer the questions:",
            False:"Answer the questions:"
        }[include_retrieval]
    elif task_type == 'fact_checking':
        return {
            True: "Refer to the background document and verify the following claims with \"True\" or \"False\":",
            False:"Verify the following claims with \"True\" or \"False\":"
        }[include_retrieval]