"""minimized script for runnging policy inference, aiming to build improved SYSTEM#2 upon it."""

import os
os.environ["HF_HOME"] = "/data0/fredhong/hf_models/"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
sys.path.append("/home/fredhong/gr00t4gp-origin/") # gr00t env pip install, need to specify otherwise it goes into the 4gp-latest repo
import torch
import gr00t

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP, Gr1ArmsOnlyDataConfig
from gr00t.model.backbone.eagle2_hg_model.inference_eagle_repo import EagleProcessor, ModelSpecificValues
# from gr00t.model.backbone.llada.mm_utils import process_images
# from gr00t.model.backbone.diffusion_backbone import DiffusionLMBackbone

# selecting backbone vlm model 
# VLM_MODEL = "lavida"  # if not specified, model will go with eagle2 by default
MODEL_PATH = "/data0/fredhong/hf_models/hub/gr00t/"
# let the hf.tr handle the downloading to hf_home, if internet broken dunno this good
# MODEL_PATH = "nvidia/GR00T-N1-2B"

import json
with open(os.path.join(MODEL_PATH, "config.json")) as json_file:
    VLM_MODEL = json.load(json_file)['vlm_model_type']


# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
EMBODIMENT_TAG = "gr1"
device = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "/home/fredhong/gr00t4gp-origin/demo_data/robot_sim.PickNPlace"


# data_config = DATA_CONFIG_MAP["gr1_arms_only"]  # this by default goes to raw eagleprocessor, below we init a eagleprocessor instance but with pretrainedtokenizerfast
# TODO ablate whether to fix num img token minimum
model_spec = ModelSpecificValues(
    template="qwen2-chat",
    num_image_token=1,   # change from 64 to 1 compared with Eagle2
) if VLM_MODEL=="lavida" else None

# # I wanna use eagle2 tokenizer while using llavida model here
# VLM_MODEL = "eagle2"  # while making config json model type lavida

# TODO whether to use lavida tokenzier or stay with eagle2 tokenizer in the 0-shot case, before transfer learning, now 0-shot
vlm_processor = EagleProcessor(model_spec=model_spec, 
                                use_lavida_tokenizer=False,
                            #    use_lavida_tokenizer=VLM_MODEL=="lavida"
                               )
# if not pass in lavida vlm proc then by default eagleprocessor
data_config = Gr1ArmsOnlyDataConfig(vlm_processor)

modality_config = data_config.modality_config()
modality_transform = data_config.transform()

# Create the dataset
dataset = LeRobotSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="decord",
    video_backend_kwargs=None,
    transforms=None,  # We'll handle transforms separately through the policy
    embodiment_tag=EMBODIMENT_TAG,
)
step_data = dataset[0]
IMG_SIZE = step_data["video.ego_view"].shape[-2]


# # HERE SWITCH THE RAW PickNPlace into the llada demo sample frame for sanity check
# from PIL import Image
# from torchvision.transforms.functional import to_tensor
# import numpy as np
# image = Image.open('/home/fredhong/diffusion_lm/LaViDa/images/dog.png').convert('RGB')

# lavida_kwargs = {'vision_kwargs': {'mm_vision_tower': 'google/siglip-so400m-patch14-384', 'mm_resampler_type': None, 'mm_projector_type': 'mlp2x_gelu', 'mm_hidden_size': 1152, 'use_mm_proj': True}, 'device_map': 'cuda:0', 'torch_dtype': torch.bfloat16}
# lavida_model = DiffusionLMBackbone.from_pretrained('jacklishufan/lavida-llada-v1.0-instruct', low_cpu_mem_usage=True, local_files_only=True, attn_implementation='eager', **lavida_kwargs)
# siglip_image_processor = lavida_model.get_vision_tower().image_processor
# image_tensor = process_images([image], siglip_image_processor, lavida_model.config)
# image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
# step_data["video.lavida_proc_frame"] = image_tensor
# del lavida_model, siglip_image_processor # free up memory
# torch.cuda.empty_cache()

# .resize((256, 256))  # same res as in PickNPlace
# step_data["video.ego_view"] = to_tensor(image).permute(1, 2, 0)[None, ...]

# # here is to put doge demo img into the same uint8 before transforming (same proc) as in eagle, but turns out not the case, no need to transform when porting lavida as the VLM here
# step_data["video.ego_view"] = (np.array(image)[None, ...] * 255).astype(np.uint8)

policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
    # model_type="lavida",
    model_type=VLM_MODEL  # comment out this line then go for default
)

# print out the policy model architecture
# print(policy.model)

# """_summary_
# testing if the finetuned eagle2 vlm inside gr00t-n1 still has the general lingustic capability.
# """
# if VLM_MODEL == "eagle2":
#     from gr00t.model.backbone.eagle2_hg_model.inference_eagle_repo import EagleProcessor
#     llm = policy.model.backbone.model.language_model # type: ignore
#     tokenizer = EagleProcessor().tokenizer

#     # NOTE simply put in text tokens as below won't work on the SYSTEM#2-eagle2-vlm, as VLM here in gr00t is only trained with both visual-lingual tokens input tgt, but NO individual input
#     inp = tokenizer(
#         "what is the color of the apple?",
#         # "pick the pear from the counter and place it in the plate",
#         return_tensors="pt"
#     ).to("cuda")
#     print(inp)
#     outputs = llm.generate(**inp, do_sample=True, max_new_tokens=100)
#     print(outputs)
#     answer = tokenizer.decode(outputs[0].cpu().tolist())
#     print(answer)


predicted_action = policy.get_action(step_data)
for key, value in predicted_action.items():
    print(key, value.shape)
print(f"we have pred: \n{predicted_action['action.left_arm'][:,0]}")