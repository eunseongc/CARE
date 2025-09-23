from transformers import MistralForCausalLM, MistralConfig, LlamaForCausalLM, LlamaConfig, Qwen3ForCausalLM, Qwen3Config
import os
import torch
import torch.nn as nn
from peft import (
    get_peft_model,
    PeftModel,
)
import re

def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False  

class ICAEConfigMixin:
    def __init__(self, projector_type='mlp2x_gelu', mem_hidden_size=4096, icae_mem_size=16, **kwargs):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.mem_hidden_size = mem_hidden_size
        self.mem_size = icae_mem_size
                
    def init_train_args(self, args_ours):
        self.args_ours = vars(args_ours) ## convert to dictionary from Namespace

class MistralICAEConfig(ICAEConfigMixin, MistralConfig): pass
class LlamaICAEConfig(ICAEConfigMixin, LlamaConfig): pass
class QwenICAEConfig(ICAEConfigMixin, Qwen3Config): pass


def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


class Projector(nn.Module):
    def __init__(self,config):
        super().__init__()
        projector_type = config.projector_type
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mem_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.projector = nn.Sequential(*modules)
    
    def forward(self,context_embedding):
        return self.projector(context_embedding)

class ICAE(torch.nn.Module):
    def __init__(self, config, model_name_or_path, lora_config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name_or_path = model_name_or_path
        self.mem_size = config.mem_size
        self.mem_hidden_size = config.mem_hidden_size

        if self.model_name_or_path == "mistralai/mistral-7b-instruct-v0.2":
            self.icae = MistralForCausalLM.from_pretrained(self.model_name_or_path,
                                                           torch_dtype=config.torch_dtype,
                                                           use_flash_attention_2=True,
                                                           config=config)
        elif self.model_name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
            self.icae = LlamaForCausalLM.from_pretrained(self.model_name_or_path,
                                                         torch_dtype=config.torch_dtype,
                                                         use_flash_attention_2=True,
                                                         config=config)
        elif self.model_name_or_path == "Qwen/Qwen3-8B":
            self.icae = Qwen3ForCausalLM.from_pretrained(self.model_name_or_path,
                                                         torch_dtype=config.torch_dtype,
                                                         use_flash_attention_2=True,
                                                         config=config)
        else:
            raise ValueError(f"Unsupported model: {self.model_name_or_path}")

        self.vocab_size = self.icae.config.vocab_size

        # self.vocab_size = self.icae.config.vocab_size + 1    # [PAD] token
        # self.pad_token_id = self.vocab_size - 1
        self.mean_compression_rate = 1

        self.projector = Projector(config)

        # special tokens for Llama-2/Mistral tokenizer
        self.bos_id = 1
        self.eos_id = 2
        
        self.dim = self.icae.config.hidden_size
        self.icae = get_peft_model(self.icae, lora_config)
        
        self.memory_token_embed = nn.Embedding(self.mem_size, self.dim, padding_idx=None).to(config.torch_dtype)
        self.memory_indices = torch.arange(self.mem_size, device=self.device).unsqueeze(0)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        # self.append_sequence = torch.arange(self.vocab_size, self.vocab_size + self.mem_size, dtype=torch.long, device=device).unsqueeze(0) 
        
        if self.training:
            self.init()

    def init(self):
        print_trainable_parameters(self)
        
    ######################### ES ############################        
    def set_xrag_token_id(self, xrag_token_id):
        self.xrag_token_id = xrag_token_id

    def set_ae_token_id(self, ae_token_id):
        self.ae_token_id = ae_token_id
        
    def resize_token_embeddings(self, len_tokenizer, len_retriever_tokenizer=0):
        print(f"Resizing token embeddings from {self.vocab_size} to {len_tokenizer}")
        self.vocab_size = len_tokenizer
        
        
        self.icae.resize_token_embeddings(len_tokenizer)
    ##############################################################
    
    def icae_forward(self, input_ids, attention_mask):
        # add gen special token to the input_ids
        # seperate the last token <\s> from the input_ids
        # input_ids here: retrieval input tokens (like context w/o instruction)
        inputs_embeds = self.icae.get_base_model().model.embed_tokens(input_ids[:, :-1])
        
        # Special memory token embedding (생성으로 갈 거면 여기서 정답 codebook 입력하여 forcing, 그렇게하면 학습 시에만 사용 가능. 안해도 괜찮을듯?)
        batch_size, _ = input_ids.size()
        batch_memory_input_embeds = self.memory_token_embed(self.memory_indices.repeat(batch_size, 1))

        input_embeds_last = self.icae.get_base_model().model.embed_tokens(input_ids[:, -1].unsqueeze(1))

        ## COMP TOKEN 뺐음
        # ALL embeddings
        inputs_embeds = torch.cat([inputs_embeds, batch_memory_input_embeds, input_embeds_last], dim=1)
        attention_mask = torch.cat([attention_mask[:, :-1],
                                    torch.ones((input_ids.size(0), self.mem_size), device=attention_mask.device),
                                    torch.ones((input_ids.size(0), 1), device=attention_mask.device)], dim=1
                                    )

        outputs = self.icae(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            output_attentions=True,
                            use_cache=False,
        )

        memory_slots_outputs = outputs.hidden_states[-1][:, -(self.mem_size+1):-1] ## ICAE embeddings
        return memory_slots_outputs


    def prepare_inputs_embeds(self, input_ids, retrieval_embeds, hybrid_ret_embeds=None):
        inputs_embeds = self.icae.get_base_model().model.embed_tokens(input_ids)

        batch_size, seq_len, hidden_size = inputs_embeds.shape

        ## sanity check
        retrieval_embeds = retrieval_embeds.reshape(-1, self.mem_hidden_size)
        num_xrag_tokens = torch.sum(input_ids==self.xrag_token_id).item()
        num_retrieval_embeds = retrieval_embeds.shape[0]
        assert num_xrag_tokens == num_retrieval_embeds, (num_xrag_tokens, num_retrieval_embeds)

        retrieval_embeds = self.projector(retrieval_embeds.to(inputs_embeds.device))
        
        inputs_embeds[input_ids == self.xrag_token_id] = retrieval_embeds
        
        return inputs_embeds


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        retrieval_embeds=None,  # [-1, retrieval_hidden_size]
        **kwargs,
    ):  
        ## when inputs_embeds is passed, it means the model is doing generation
        ## and only the first round of generation would pass inputs_embeds
        ## https://github.com/huggingface/transformers/blob/79132d4cfe42eca5812e8c45ea1b075f04f907b6/src/transformers/models/llama/modeling_llama.py#L1250
        
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        at_the_beginning_of_generation = False
        if inputs_embeds is not None:
            assert not self.training
            assert retrieval_embeds is None
            at_the_beginning_of_generation = True

        if not at_the_beginning_of_generation:
            if retrieval_embeds is not None:
                inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
                input_ids = None

                # Sanity check
                if attention_mask is not None:
                    assert inputs_embeds.shape[1] == attention_mask.shape[1], (inputs_embeds.shape, attention_mask.shape)


        with self.icae.disable_adapter():  
            outputs = self.icae(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                **kwargs,
            )
        return outputs

    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        save_function = None,
        state_dict = None,
        safe_serialization: bool = False,
        tokenizer=None
    ):
        """
        Saves the ICAE model (base + LoRA adapters + any custom attributes) to `save_directory`.
        """
        # If we're not the main process, don't do any saving to avoid conflicts
        if not is_main_process:
            return

        # If no custom save_function is provided, default to torch.save
        if save_function is None:
            save_function = torch.save

        # 1. Make sure the directory exists
        os.makedirs(save_directory, exist_ok=True)

        # 2. Save model config, if needed
        #    (If self.config is a HuggingFace config, we can do:)
        self.config.save_pretrained(save_directory)

        # 3. Save the model (base + LoRA), handling both cases
        #    a) If it's a PEFT model, only LoRA adapters are saved
        #    b) Otherwise, save the full model weights
        if isinstance(self.icae, PeftModel):
            self.icae.save_pretrained(
                save_directory,
                is_main_process=is_main_process,
                save_function=save_function,
                state_dict=state_dict
            )
        else:
            self.icae.save_pretrained(
                save_directory,
                safe_serialization=safe_serialization,
                save_function=save_function,
                state_dict=state_dict
            )

        # 5. Optionally, save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)

        print(f"> [ICAE] Model saved to {save_directory}")


    @torch.no_grad()
    def generate(
        self,
        input_ids = None,
        retrieval_embeds = None,
        hybrid_ret_embeds = None,
        **kwargs,
    ):
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for generate")

        if retrieval_embeds is not None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
            input_ids = None

            # Sanity check
            if attention_mask is not None:
                assert inputs_embeds.shape[1] == attention_mask.shape[1], (inputs_embeds.shape, attention_mask.shape)
  

        with self.icae.disable_adapter():  
            outputs = self.icae.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )

        return outputs