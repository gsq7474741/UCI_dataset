"""1D Grad-CAM implementation for time series classification.

Based on: "1D Grad-CAM" paper methodology
- Weight map: feature maps × gradients
- Importance map: weight map × purity (for qualitative evaluation)
- Contribution map: feature scores (for quantitative evaluation)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GradCAM1D:
    """1D Grad-CAM for temporal CNN models.
    
    Computes gradient-weighted class activation maps for 1D time series.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: CNN model
            target_layer: Target conv layer for Grad-CAM (default: last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        # Find target layer if not specified
        if self.target_layer is None:
            self.target_layer = self._find_last_conv_layer()
        
        # Register hooks
        self._register_hooks()
    
    def _find_last_conv_layer(self) -> nn.Module:
        """Find the last Conv1d layer in the model."""
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv1d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv1d layer found in model")
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute Grad-CAM for input.
        
        Args:
            x: Input tensor [B, C, T]
            target_class: Target class index (default: predicted class)
            
        Returns:
            Dict with:
            - 'weight_map': [B, T] weight map
            - 'cam': [B, T] class activation map (normalized)
            - 'logits': [B, num_classes] model output
            - 'pred': [B] predicted class
        """
        self.model.eval()
        
        # Forward pass
        x.requires_grad_(True)
        logits = self.model(x)
        pred = logits.argmax(dim=-1)
        
        # Use predicted class if not specified
        if target_class is None:
            target_idx = pred
        else:
            target_idx = torch.full_like(pred, target_class)
        
        # Backward pass for target class
        self.model.zero_grad()
        
        # One-hot encode target
        one_hot = torch.zeros_like(logits)
        for i in range(len(target_idx)):
            one_hot[i, target_idx[i]] = 1
        
        # Backward
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Get activations and gradients
        activations = self.activations  # [B, C_feat, T_feat]
        gradients = self.gradients      # [B, C_feat, T_feat]
        
        # Compute weights (global average of gradients)
        # Paper: weight = mean(gradients, axis=time)
        weights = gradients.mean(dim=-1, keepdim=True)  # [B, C_feat, 1]
        
        # Compute weight map: sum(weights * activations)
        # This gives importance of each time step
        weight_map = (weights * activations).sum(dim=1)  # [B, T_feat]
        
        # Upsample to original time dimension
        T_orig = x.shape[-1]
        T_feat = weight_map.shape[-1]
        
        if T_feat != T_orig:
            weight_map = F.interpolate(
                weight_map.unsqueeze(1),
                size=T_orig,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        # Normalize CAM to [0, 1]
        cam = weight_map.clone()
        for i in range(cam.shape[0]):
            cam_i = cam[i]
            cam_min = cam_i.min()
            cam_max = cam_i.max()
            if cam_max - cam_min > 1e-8:
                cam[i] = (cam_i - cam_min) / (cam_max - cam_min)
            else:
                cam[i] = torch.zeros_like(cam_i)
        
        return {
            'weight_map': weight_map,
            'cam': cam,
            'logits': logits.detach(),
            'pred': pred,
            'target': target_idx,
        }
    
    def compute_contribution(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute feature contribution map.
        
        The contribution is the percentage of each feature's score
        in the final CNN output.
        
        Args:
            x: Input tensor [B, C, T]
            target_class: Target class index
            
        Returns:
            Dict with contribution map and other results
        """
        result = self(x, target_class)
        
        # Get softmax probabilities
        probs = F.softmax(result['logits'], dim=-1)
        
        # Contribution = weight_map * probability of target class
        target_probs = probs.gather(1, result['target'].unsqueeze(1)).squeeze(1)  # [B]
        
        # Scale weight map by class probability
        contribution = result['weight_map'] * target_probs.unsqueeze(1)  # [B, T]
        
        # Normalize
        contribution_norm = contribution.clone()
        for i in range(contribution.shape[0]):
            c = contribution[i]
            c_abs_sum = c.abs().sum()
            if c_abs_sum > 1e-8:
                contribution_norm[i] = c / c_abs_sum
        
        result['contribution'] = contribution_norm
        result['probs'] = probs
        
        return result


def compute_purity(
    data: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute purity (class separability) at each time step.
    
    Purity measures how well each time step separates classes.
    Higher purity = better class separation = more important feature.
    
    Args:
        data: [N, T] time series data (single channel)
        labels: [N] class labels
        num_classes: Number of classes
        
    Returns:
        [T] purity scores
    """
    N, T = data.shape
    purity = torch.zeros(T, device=data.device)
    
    for t in range(T):
        # Get values at this time step for each class
        class_means = []
        class_vars = []
        
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                vals = data[mask, t]
                class_means.append(vals.mean())
                class_vars.append(vals.var() + 1e-8)
        
        if len(class_means) >= 2:
            # Between-class variance
            overall_mean = data[:, t].mean()
            between_var = sum((m - overall_mean) ** 2 for m in class_means) / len(class_means)
            
            # Within-class variance
            within_var = sum(class_vars) / len(class_vars)
            
            # Purity = between / within (Fisher criterion)
            purity[t] = between_var / within_var
    
    # Normalize to [0, 1]
    purity_min = purity.min()
    purity_max = purity.max()
    if purity_max - purity_min > 1e-8:
        purity = (purity - purity_min) / (purity_max - purity_min)
    
    return purity


def compute_importance(
    weight_map: torch.Tensor,
    purity: torch.Tensor,
) -> torch.Tensor:
    """Compute importance map = weight_map × purity.
    
    Args:
        weight_map: [B, T] or [T] weight map from Grad-CAM
        purity: [T] purity scores
        
    Returns:
        [B, T] or [T] importance map
    """
    return weight_map * purity
