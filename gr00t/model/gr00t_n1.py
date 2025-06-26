# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone
from .backbone import DiffusionLMBackbone
from .backbone.llada.mm_utils import process_images

from accelerate import load_checkpoint_and_dispatch
import torchvision.transforms.functional as F


BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00T_N1Config(PretrainedConfig):
    model_type = "gr00t_n1"
    vlm_model_type = "eagle2"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1Config,
        local_model_path: str,
        # vlm_model_type: str  # this a flag helps us switch from the vanilla version to DVLM
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        if config.vlm_model_type != "lavida":
            self.backbone = EagleBackbone(**config.backbone_cfg)
        else:
            lavida_kwargs = {
                'vision_kwargs': {'mm_vision_tower': 'google/siglip-so400m-patch14-384', 'mm_resampler_type': None, 'mm_projector_type': 'mlp2x_gelu', 'mm_hidden_size': 1152, 'use_mm_proj': True}, 
                # 'device_map': 'auto',   # by default == 'meta'
                'torch_dtype': torch.bfloat16
            }
            self.backbone = DiffusionLMBackbone.from_pretrained(
                "/data0/fredhong/hf_models/hub/models--jacklishufan--lavida-llada-v1.0-instruct/snapshots/814b2e364e82390f03df451bdf4e81e8ba8eab37/",
                # "jacklishufan/lavida-llada-v1.0-instruct",
                low_cpu_mem_usage=True, local_files_only=True, 
                attn_implementation='eager', 
                **lavida_kwargs
                # "/data1/fredhong/hf_models/hub/models--hbXNov--lavida-llada-reason/snapshots/246b218b0444bc9edfd09ab0e05b072db96e5e2a/"
                # "hbXNov/lavida-llada-reason"
            )
            self.backbone.tie_weights()
            self.backbone = load_checkpoint_and_dispatch(
                self.backbone, 
                "/data0/fredhong/hf_models/hub/models--jacklishufan--lavida-llada-v1.0-instruct/snapshots/814b2e364e82390f03df451bdf4e81e8ba8eab37/", 
                device_map='auto'
            )
            # [to be removed here] as mentioned in get_action(), a img with size smaller than vision_tower.img_size, cannot be patchified and proc by lavida vision tower, therefore don't shrink the ratio of image grid pin point here
            # raw_lavida_wid, raw_lavida_hei, img_size = 692, 704, 256
            # self.backbone.config.image_grid_pinpoints = \
            # list(map(lambda l: [int(l[0] / raw_lavida_wid * img_size), int(l[1] / raw_lavida_hei * img_size)], self.backbone.config.image_grid_pinpoints))
            # self.backbone.config.scale_ratio = raw_lavida_wid / img_size  # this for tuning the patch division in process_anyres_image()
        self.backbone_type = config.vlm_model_type
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        # # for lavida input w/o eagle2 proc, uncomment below and commetn the inputs assign
        # raw_image = inputs.pop("video.lavida_proc_frame", None)
        # image_to_process = raw_image
        # image_to_process = inputs["pixel_values"]
        # backbone_inputs["pixel_values"] = raw_image

        # get universal backbone inputs
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # for lavida input w/o eagle2 proc, commetn the inputs assign
        # using the eagle2 robot input to be processed by the lavida processor
        if self.backbone_type == "lavida":
            # img transform: originally for PIL, now scale pixel_values back to (0,1) for the sake of image proc siglip encoder
            image_to_process = ((inputs["pixel_values"] + 1) * 255 / 2).to(torch.uint8)

            # TODO move this transform to gr00t.transform
            if image_to_process.shape[-1] < 384:  # a img with size smaller than vision_tower.img_size, cannot be patchified and proc by lavida vision tower
                image_to_process = F.resize(image_to_process, [704])

            # lavida img processing discarding the llava patchifying, to follow eagle2 siglip enc fashion
            # TODO lavida processor now not patchifying the input frames, but scale the input img to correct lavida/llava res
            image_processor = self.backbone.get_vision_tower().image_processor
            _config = self.backbone.config
            # _config.image_aspect_ratio = "scale2_384"  # this line deactivates patchifying
            image_tensor = process_images(image_to_process, image_processor, _config)
            image_tensor = [_image.to(dtype=torch.bfloat16, device="cuda") for _image in image_tensor]
            if image_tensor[0].dim() == 3:  # the image tensor must be wrapped in a list, accord with prepare_inputs_labels_for_multimodal
                print("rewrapping image tensor into a list of tensor")
                image_tensor = [image_tensor[0][None]]  # it's one img with chw shape, needs to be flattened to accord with siglip
            backbone_inputs["pixel_values"] = image_tensor
            backbone_inputs["raw_image_sizes"] = [image_to_process.shape[-2:]]

            logits = self.backbone.generate(
                backbone_inputs.input_ids,
                return_logits=True,
                images=image_tensor,
                # image_sizes=backbone_inputs.raw_image_sizes,
                image_sizes=[(704, 704)],
                do_sample=False,
                temperature=0.1,
                max_new_tokens=64,
                block_length=64,
                step_ratio=0.5, # 32 steps
                tokenizer=None,
                prefix_lm=True,
                verbose=True,
                schedule='shift',
            )
            # TODO here needs a re-TRAINED linear layer to act as projector of between embedding dims. See Eagle2 backbone
            #   and for DiffusionBackbone, this linear should be `Linear(infeat=126464, outfeat=1536)`
            #   rather than `Linear(infeat=2048, outfeat=1536)` as in Eagle2
            # logits = self.linear(logits)
            logits = logits[..., -1536:]

            backbone_outputs = BatchFeature(
                data={
                    "backbone_features": logits,  # return_dict=False, and no label passed in, thus the first tensor is the outptu logits
                }
            )
        else:  # for eagle2
            # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
            backbone_outputs = self.backbone(backbone_inputs)
        
        # backbone output feat requires shape of torch.Size([1, n_id, 1536])
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str,
        model_type: str,
        **kwargs
    ):
        # import pdb; pdb.set_trace()
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        # kwargs["vlm_model_type"] = model_type  # by default is eagle2, this kwarg is to enable lavida dllm as backbone
        print(kwargs)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        # TODO now the model_type arg cannot control config.json
        #   as modifying the json file in the model_path looks awkward
        #   need to solve this and pass backbone model control northernbound back to policy.py
        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        if model_type == "eagle2":
            pretrained_model.backbone.set_trainable_parameters(
                tune_visual=tune_visual, tune_llm=tune_llm
            )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model


# register
AutoConfig.register("gr00t_n1", GR00T_N1Config)
AutoModel.register(GR00T_N1Config, GR00T_N1)
