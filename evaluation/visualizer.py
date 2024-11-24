# evaluation/visualizer.py

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict
from skimage.transform import resize

class Visualizer:
    def __init__(self, config):
        self.config = config
        
    def resize_to_match(
        self,
        source: np.ndarray,
        target_shape: tuple
    ) -> np.ndarray:
        """Resize source array to match target shape
        
        Args:
            source: Source array to resize
            target_shape: Shape to match (H, W)
            
        Returns:
            np.ndarray: Resized array
        """
        if source.shape[:2] != target_shape[:2]:
            # Keep the number of channels if present
            output_shape = target_shape[:2]
            if source.ndim == 3:
                output_shape = output_shape + (source.shape[2],)
            
            source = resize(source,
                          output_shape,
                          mode='reflect',
                          anti_aliasing=True,
                          preserve_range=True)
            
            # Ensure the values are in the correct range
            if source.max() > 1.0:
                source = source / 255.0
                
        return source
        
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """Overlay heatmap on image
        
        Args:
            image: Original image (H,W,3)
            heatmap: Attention heatmap (H,W)
            alpha: Transparency value
            
        Returns:
            np.ndarray: Overlayed visualization
        """
        # Ensure image is in correct format
        if image.max() > 1.0:
            image = image / 255.0
            
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
            
        # Resize heatmap to match image size
        heatmap = self.resize_to_match(heatmap, image.shape)
        
        # Create colormap
        cmap = plt.get_cmap('jet')
        heatmap_colored = cmap(heatmap)[..., :3]
        
        # Overlay
        overlayed = (1 - alpha) * image + alpha * heatmap_colored
        
        # Clip values
        overlayed = np.clip(overlayed, 0, 1)
        return overlayed
    
    def plot_results(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """Plot visualization results
        
        Args:
            image: Original image
            heatmap: Attention heatmap
            ground_truth: Ground truth mask
            metrics: Evaluation metrics
            title: Plot title
            save_path: Path to save visualization
        """
        try:
            # Print original shapes for debugging
            print(f"\nDebug shapes before processing:")
            print(f"Image shape: {image.shape}")
            print(f"Heatmap shape: {heatmap.shape}")
            if ground_truth is not None:
                print(f"Ground truth shape: {ground_truth.shape}")
            
            # Ensure all arrays are float and in range [0,1]
            image = image.astype(float)
            if image.max() > 1.0:
                image = image / 255.0
                
            # Create figure
            fig, axes = plt.subplots(
                1, 
                3 if ground_truth is not None else 2,
                figsize=self.config.PLOT_SIZE
            )
            
            # Plot original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Plot heatmap overlay
            overlayed = self.overlay_heatmap(image, heatmap)
            axes[1].imshow(overlayed)
            axes[1].set_title('Attention Map')
            axes[1].axis('off')
            
            # Plot ground truth if available
            if ground_truth is not None:
                # Resize ground truth to match image size
                gt_resized = self.resize_to_match(ground_truth, image.shape)
                axes[2].imshow(gt_resized, cmap='gray')
                axes[2].set_title('Ground Truth')
                axes[2].axis('off')
            
            # Add title and metrics
            if title:
                fig.suptitle(title[:200], fontsize=12)
                
            if metrics:
                metric_text = '\n'.join(
                    f"{k}: {v:.3f}" for k, v in metrics.items()
                )
                fig.text(0.02, 0.02, metric_text, fontsize=8)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if path provided
            if save_path and self.config.SAVE_RESULTS:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            plt.close(fig)  # Close to prevent memory leaks
            
        except Exception as e:
            print(f"Error in plot_results: {str(e)}")
            print(f"Image shape: {image.shape}")
            print(f"Heatmap shape: {heatmap.shape}")
            if ground_truth is not None:
                print(f"Ground truth shape: {ground_truth.shape}")
            plt.close('all')  # Clean up any open figures