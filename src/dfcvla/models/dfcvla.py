import os
import dataclasses
import torch
from torch import nn
import safetensors.torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models.pi0_config import Pi0Config
from vggt.models.vggt import VGGT
from dfcvla.config import VGGTConfig


class DFCVLA(nn.Module):
    def __init__(self, base_model_config: Pi0Config, vggt_config: VGGTConfig, dfvla_config = None):
        super().__init__()
        self.base_model_config = base_model_config
        self.vggt_config = vggt_config

        self.base_model = PI0Pytorch(base_model_config)
        self.vggt = VGGT(img_size=vggt_config.img_size,
                         patch_size=vggt_config.patch_size,
                         embed_dim=vggt_config.embed_dim,
                         enable_camera=vggt_config.enable_camera,
                         enable_point=vggt_config.enable_point,
                         enable_depth=vggt_config.enable_depth,
                         enable_track=vggt_config.enable_track,
                         feature_only=vggt_config.feature_only)
        
    def forward(self, observation, actions, noise=None, time=None):
        return self.base_model(observation, actions, noise, time)