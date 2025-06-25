import torch
from typing import List, Optional, Union
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.feature_extraction_utils import BatchFeature
# from dataclasses import dataclass
from .llada.modeling_llada import LLaDAModel, LLaDAModelLM, LLaDAConfig, create_model_config_from_pretrained_config
from .llada.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.generation.utils import GenerateOutput
import os
from accelerate.utils import reduce
ENFORCE_NUM_ITEMIN_BATCH = os.environ.get("ENFORCE_NUM_ITEMIN_BATCH", False)

eos_id = 126081 # hack
mask_id = 126336
fim_id = 126085

# @dataclass  # don't make it as dataclass, otherwise AutoConfig.from_pretrained() fails to modify attr within
class LlavaLladaConfig(LLaDAConfig):
    model_type = "llava_llada"
    
    
class LlavaLladaModel(LlavaMetaModel,LLaDAModel):
    config_class = LlavaLladaConfig
    dtype = torch.bfloat16 # hack

    def __init__(self, pretrained_config,llada_config,init_params=None,vision_kwargs=None):
        # breakpoint()
        
        LLaDAModel.__init__(self, llada_config)
        LlavaMetaModel.__init__(self, pretrained_config,vision_kwargs=vision_kwargs,skip_init=True)
        
    def embed_tokens(self, x):
        return self.transformer.wte(x)

def sample_t(b,device,policy='uniform',policy_args=None):
    if policy == 'uniform':
        return torch.rand(b, device=device)
    elif policy == 'logit_normal':
        if policy_args is None:
            policy_args = dict(logit_mean=0.0,logit_std=1.0)
        u = torch.normal(mean=policy_args['logit_mean'], std=policy_args['logit_std'], size=(b,), device="cpu")
        u = torch.nn.functional.sigmoid(u).to(device=device)
        return u
    elif policy == "mode":
        u = torch.rand(size=(b,), device="cpu")
        u = 1 - u - policy_args['mode_scale'] * (torch.cos(torch.pi * u / 2) ** 2 - 1 + u)
        return u
        
def forward_process(bsz,seq_len,device, eps=1e-3,policy='uniform',policy_args=None):
    b, l = bsz,seq_len
    t = sample_t(b,device,policy=policy,policy_args=policy_args)
    # t = torch.sigmoid(t)
    p_mask = (1 - eps) * t + eps
    
    p_mask = p_mask[:, None]#.repeat(1, l)
    
    masked_indices = torch.rand((b, l), device=device)
    mask_cutoff =  torch.max(p_mask,masked_indices.min(-1,keepdim=True).values)
    masked_indices = masked_indices <= mask_cutoff
    # mask at least one token
    # 126336 is used for [MASK] token
    #noisy_batch = torch.where(masked_indices, 126336, input_ids)
    
    return masked_indices, p_mask
import os
LOG_BATCH_LENGTH = os.environ.get('LOG_BATCH_LENGTH', False)
DEBUG_PRINT_IMAGE_RES = os.environ.get("DEBUG_PRINT_IMAGE_RES", False)

class DiffusionLMBackbone(LLaDAModelLM,LlavaMetaForCausalLM):
    
    config_class = LlavaLladaConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, 
    config: LLaDAConfig, 
    model: Optional[LLaDAModel] = None, init_params: bool = False,vision_kwargs=None,prefix_lm=False,**kwargs):
        LLaDAModelLM.__init__(self, config,model,init_params)

        # configure default generation settings
        config.model_type = "llava_llada"
        # config.rope_scaling = None
        self.prefix_lm = prefix_lm

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            self.model = LlavaLladaModel(config,model_config, init_params=init_params,vision_kwargs=vision_kwargs)
        else:
            self.model = model
        self.model.set_activation_checkpointing('whole_layer')
        
        self.post_init() # TODO
        
    def get_model(self):
        return self.model
    
    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)
    
    def forward(
        self,
        vl_input: BatchFeature,
    ) -> BatchFeature:
        # images tensor wrapped in a list for prepare_inputs_labels_for_multimodal, otherwise will be recognized as single batch image and cannot be correctly proc
        images = vl_input["pixel_values"]  # list already
        raw_image_sizes = vl_input.pop("raw_image_sizes", [(692, 704)])  # if no key, then takes the lavida default input dog pic size

        raw_input_ids = vl_input["input_ids"].clone()
        input_ids = vl_input["input_ids"]
        attention_mask_raw = vl_input["attention_mask"].clone()
        attention_mask = vl_input["attention_mask"]
        
        inputs_embeds = vl_input.pop("inputs_embeds", None)
        position_ids = vl_input.pop("position_ids", None)
        past_key_values = vl_input.pop("past_key_values", None)
        labels = vl_input.pop("labels", None)
        image_sizes = vl_input.pop("image_sizes", None)

        use_cache = vl_input.pop("use_cache", False)
        output_attentions = vl_input.pop("output_attentions", None)
        output_hidden_states = vl_input.pop("output_hidden_states", None)
        return_dict = vl_input.pop("return_dict", False)

        non_padding = ~(raw_input_ids==eos_id)
        attention_mask[raw_input_ids==eos_id] = True # no sequence attention mask per Sec B.1
        if labels is not None:
            labels[raw_input_ids==eos_id] = eos_id # revert target
            # fix attention mask FIXME is it a typo? will this take effect on input_ids?
            vl_input["input_ids"] == vl_input["input_ids"]

        if inputs_embeds is None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=raw_image_sizes)
        #prompt_lengths = 
        #breakpoint()
        # hack starts here
        # 1. Get the mask of trget tokens 
        # if we have labels, run forward process
        # prefix_length = 
        # 
        prompt_len = None
        if labels is not None:
            assert labels.min() == -100
            labels_mask = ~(labels == -100) # targets mask
            infill_token_pos = labels==fim_id
            # find index of the first non zero mask
            # labels_mask = labels_mask.cumsum(-1).eq(1)
            if self.prefix_lm:
                # breakpoint()
                prompt_len = labels_mask.float().argmax(dim=1)
                # print(prompt_len)
            noise_embeddings = self.get_model().transformer.wte(torch.tensor([mask_id]).to(raw_input_ids))
            # noise_embeddings is 1, 4096
            bsz,seq_len = labels_mask.shape
            noise_embeddings = noise_embeddings.view(1,1,-1)#.repeat(bsz,seq_len,1)
            # t = torch.rand(b, device=input_ids.device)
            masked_indices, p_mask = forward_process(bsz,seq_len,raw_input_ids.device,policy=policy,policy_args=policy_args)
            # torch.where()
            final_masked_indices = masked_indices&labels_mask & (~infill_token_pos)
            final_masked_indices_inv = (~masked_indices)&labels_mask & (~infill_token_pos)
            # breakpoint()
            # breakpoint()
            # boardcast goingon here
            # final_masked_indices: B X L X 1
            # noise_embeddings: 1 X 1 X D
            # inputs_embeds:  B X L X D
            inputs_embeds_inv = torch.where(final_masked_indices_inv.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)
            inputs_embeds = torch.where(final_masked_indices.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)
            # inputs_embeds_inv = torch.where(final_masked_indices_inv.view(bsz,seq_len,1),noise_embeddings,inputs_embeds)
            # print(final_masked_indices.float().mean(-1).cpu())
            # new_input_ids
            # breakpoint()
            
            labels_inv = labels.clone()
            labels_inv[~final_masked_indices_inv] = -100
            labels[~final_masked_indices] = -100
            labels[labels==fim_id] = -100 # kill infill token so we don't predict it
            
            inputs_embeds = torch.cat([inputs_embeds,inputs_embeds_inv])
            labels =  torch.cat([labels,labels_inv])
            if self.prefix_lm:
                prompt_len = prompt_len.repeat(2,1)
            final_masked_indices = torch.cat([final_masked_indices,final_masked_indices_inv])
            seq_len = labels.shape[-1]
            # print(seq_len)
            if LOG_BATCH_LENGTH:
                print("Batch Length",seq_len)
            CUFOFF=30720
            if seq_len > CUFOFF:
                print(seq_len,labels.shape)
                labels = labels[:,:CUFOFF]
                inputs_embeds = inputs_embeds[:,:CUFOFF]
                attention_mask = attention_mask[:,:CUFOFF]
                if position_ids is not None:
                    position_ids = position_ids[:,:CUFOFF]
                assert input_ids is None
                assert past_key_values is None
            elif seq_len < CUFOFF:
                pass
                # raise ValueError("Out of Length")
                # pad_len_max = 128 #torch.randint(0, 128, (1,)).item()
                # if pad_len_max > 0:
                #     pad_len = torch.randint(0,pad_len_max,(1,)).item()
                #     padding = torch.full((bsz,pad_len),eos_id,dtype=labels.dtype,device=labels.device) 
                #     labels = torch.cat([labels,padding],dim=-1)
                #     new_input_ids  = torch.cat([new_input_ids,padding],dim=-1)
                #     padding = torch.full((bsz,pad_len,inputs_embeds.shape[-1]),0,dtype=inputs_embeds.dtype,device=inputs_embeds.device)
                #     inputs_embeds = torch.cat([inputs_embeds,padding],dim=-2)
                #     padding = torch.full((bsz,pad_len),1,dtype=attention_mask.dtype,device=attention_mask.device)
                #     attention_mask = torch.cat([attention_mask,padding],dim=-1)
                #     if position_ids is not None:
                #         padding = torch.full((bsz,padding),0,dtype=position_ids.dtype,device=position_ids.device)
                #         position_ids = torch.cat([position_ids,padding],dim=-1)
        #assert attention_mask is None or torch.all(attention_mask)
        attention_mask = None
        num_items_in_batch = None
        if ENFORCE_NUM_ITEMIN_BATCH:
            num_items_in_batch = labels.ne(-100).float().sum()
            num_items_in_batch = reduce(num_items_in_batch)
            num_items_in_batch = num_items_in_batch.long()

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prompt_len=prompt_len,
            num_items_in_batch=num_items_in_batch,
        )
        # output['new_input_ids']=new_input_ids
        # output['labels'] = labels
        # output['final_masked_indices']=final_masked_indices
        # output['p_mask'] = p_mask
        output = BatchFeature(
            data={
                "backbone_features": output[0],  # return_dict=False, and no label passed in, thus the first tensor is the outptu logits
                "backbone_attention_mask": attention_mask_raw,
            }
        )
        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            # breakpoint()
            inputs_embeds = self.get_model().embed_tokens(inputs)
        # if DEBUG_PRINT_IMAGE_RES:
        #     print("Seq len:",inputs_embeds.shape[1])

        #return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        return llada_generate(self.get_model(),inputs_embeds=inputs_embeds,position_ids=position_ids,attention_mask=attention_mask,**kwargs)
    
    
    @torch.no_grad()
    def log_likelyhood_inference(
        self,
        inputs: Optional[torch.Tensor] = None,
        answer: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        mc_num=128,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        max_seq_len = 5000
        #if inputs_embeds.shape[1] > max_seq_len:
        max_seq_len = max_seq_len[:,-max_seq_len:]
        answer = answer[:300]
        return get_log_likelihood(self.get_model(), None,inputs_embeds=inputs_embeds, answer=answer, mc_num=mc_num,**kwargs)
        #return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        return llada_generate(self.get_model(),inputs_embeds=inputs_embeds,position_ids=position_ids,attention_mask=attention_mask,**kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_llada", LlavaLladaConfig)
AutoModelForCausalLM.register(LlavaLladaConfig, DiffusionLMBackbone)