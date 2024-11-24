# evaluation/metrics.py

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import cv2

class EvaluationMetrics:
    def __init__(self):
        pass
        
    def compute_metrics(self, heatmap: np.ndarray, gt_mask: np.ndarray) -> dict:
        """Compute evaluation metrics between heatmap and ground truth mask
        
        Args:
            heatmap (np.ndarray): Attention heatmap
            gt_mask (np.ndarray): Ground truth mask
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Ensure both arrays have the same shape
        if heatmap.shape != gt_mask.shape:
            print(f"Resizing arrays - Heatmap: {heatmap.shape}, GT Mask: {gt_mask.shape}")
            # Resize heatmap to match ground truth mask size
            heatmap = cv2.resize(heatmap, (gt_mask.shape[1], gt_mask.shape[0]), 
                               interpolation=cv2.INTER_LINEAR)
        
        # Flatten arrays
        heatmap_flat = heatmap.flatten()
        mask_flat = gt_mask.flatten()
        
        # Normalize heatmap to [0,1] if needed
        if heatmap_flat.max() > 1 or heatmap_flat.min() < 0:
            heatmap_flat = (heatmap_flat - heatmap_flat.min()) / (heatmap_flat.max() - heatmap_flat.min())
        
        # Ensure mask is binary
        mask_flat = mask_flat > 0.5
        
        try:
            # Compute AUC-ROC
            auc_score = roc_auc_score(mask_flat, heatmap_flat)
            
            # Compute PR-AUC
            precision, recall, _ = precision_recall_curve(mask_flat, heatmap_flat)
            pr_auc = auc(recall, precision)
            
            # Compute IoU (using threshold of 0.5 for heatmap)
            heatmap_binary = heatmap_flat > 0.5
            intersection = np.logical_and(heatmap_binary, mask_flat).sum()
            union = np.logical_or(heatmap_binary, mask_flat).sum()
            iou = intersection / (union + 1e-8)
            
            # Compute coverage
            coverage = np.sum(np.logical_and(heatmap_binary, mask_flat)) / (np.sum(mask_flat) + 1e-8)
            
        except Exception as e:
            print(f"Error computing metrics: {str(e)}")
            print(f"Heatmap stats - min: {heatmap_flat.min()}, max: {heatmap_flat.max()}")
            print(f"Mask stats - min: {mask_flat.min()}, max: {mask_flat.max()}")
            print(f"Shapes - Heatmap: {heatmap.shape}, Mask: {gt_mask.shape}")
            return {
                'auc_roc': 0.0,
                'pr_auc': 0.0,
                'iou': 0.0,
                'coverage': 0.0
            }
            
        return {
            'auc_roc': auc_score,
            'pr_auc': pr_auc,
            'iou': iou,
            'coverage': coverage
        }