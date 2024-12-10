# models/clip_wrapper.py

import torch
import clip
from PIL import Image
from typing import Tuple

class CLIPWrapper:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.model, self.preprocess = clip.load(
            config.CLIP_MODEL, 
            device=self.device,
            jit=False
        )
        
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """编码图像"""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        return image_input
    
    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本"""
        text_input = clip.tokenize([text]).to(self.device)
        return self.model.encode_text(text_input).float()
    
    def get_visual_model(self):
        """获取视觉模型部分"""
        return self.model.visual
    
    def get_target_layer(self):
        """获取目标层"""
        return getattr(self.model.visual, self.config.SALIENCY_LAYER)

