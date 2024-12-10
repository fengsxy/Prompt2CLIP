# models/gradcam.py

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from scipy.ndimage import filters

class GradCAM:
    def __init__(self, config):
        self.config = config
    
    def compute_heatmap(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target: torch.Tensor,
        layer: nn.Module
    ) -> np.ndarray:
        """Compute GradCAM attention heatmap
        
        Args:
            model: Visual model
            input_tensor: Input image tensor
            target: Target text embedding
            layer: Target layer for GradCAM
            
        Returns:
            np.ndarray: Normalized attention heatmap
        """
        # Record original input size
        original_size = input_tensor.shape[2:]
        
        # Zero out existing gradients
        if input_tensor.grad is not None:
            input_tensor.grad.data.zero_()
            
        # Store gradient states
        requires_grad = {}
        for name, param in model.named_parameters():
            requires_grad[name] = param.requires_grad
            param.requires_grad_(False)
            
        # Get intermediate activations and gradients
        with Hook(layer) as hook:        
            output = model(input_tensor)
            output.backward(target)

            grad = hook.gradient.float()
            act = hook.activation.float()
            
            # Global average pool gradients
            alpha = grad.mean(dim=(2, 3), keepdim=True)
            
            # Weight combination of activation maps
            gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
            
            # ReLU to keep only positive influences
            gradcam = torch.clamp(gradcam, min=0)

        # Resize to original input size
        gradcam = F.interpolate(
            gradcam,
            original_size,
            mode='bicubic',
            align_corners=False
        )
        
        # Restore gradient settings
        for name, param in model.named_parameters():
            param.requires_grad_(requires_grad[name])
            
        # Convert to numpy and normalize
        heatmap = gradcam.squeeze().detach().cpu().numpy()
        
        # Apply Gaussian blur if configured
        if self.config.BLUR_HEATMAP:
            heatmap = filters.gaussian_filter(
                heatmap,
                sigma=0.02 * max(original_size)
            )
            
        # Normalize to [0,1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap

class Hook:
    """Hook class to get intermediate layer activations"""
    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad