# main.py

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from typing import List, Dict, Tuple

from config.config import Config
from data.coco_loader import COCOLoader
from models.clip_wrapper import CLIPWrapper
from models.gradcam import GradCAM
from evaluation.metrics import EvaluationMetrics
from evaluation.visualizer import Visualizer

class ExperimentRunner:
    def __init__(self):
        self.config = Config()
        self.coco_loader = COCOLoader(self.config)
        self.clip_wrapper = CLIPWrapper(self.config)
        self.gradcam = GradCAM(self.config)
        self.metrics = EvaluationMetrics()
        self.visualizer = Visualizer(self.config)
        
    def generate_prompt(
        self,
        object_name: str,
        context_info: str = None,
        prompt_type: str = 'baseline'
    ) -> str:
        """Generate prompts based on different experiment types
        
        Args:
            object_name (str): Name of the target object
            context_info (str, optional): Additional context information
            prompt_type (str): Type of prompt to generate ('baseline', 'context', or 'caption')
            
        Returns:
            str: Generated prompt text
        """
        if prompt_type == 'baseline':
            return object_name
        elif prompt_type == 'context':
            return f"{context_info} {object_name}"
        elif prompt_type == 'caption':
            return f"a photo of {context_info} with a {object_name}"
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
    def process_single_image(
        self,
        image_info: Dict,
        object_name: str,
        prompt: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        """Process a single image
        
        Args:
            image_info: Image metadata
            object_name: Target object name
            prompt: Text prompt
            
        Returns:
            Tuple containing:
            - normalized image
            - attention heatmap
            - ground truth mask
            - evaluation metrics
        """
        try:
            # Load image
            image = self.coco_loader.load_image(image_info)
            if image is None:
                print(f"Failed to load image: {image_info['id']}")
                return None
                
            # Keep original image for visualization
            original_image = image.copy()
                
            # Convert to RGB PIL image and resize for CLIP
            image_pil = Image.fromarray(image).convert('RGB')
            image_pil = image_pil.resize((224, 224))  # CLIP's expected input size
            
            # Get ground truth mask
            gt_mask = self.coco_loader.get_object_masks(image_info, object_name)
            if gt_mask is None:
                print(f"Failed to get mask for object {object_name} in image {image_info['id']}")
                return None
            
            # Print shapes for debugging
            print(f"\nProcessing image {image_info['id']}:")
            print(f"Original image shape: {original_image.shape}")
            print(f"Ground truth mask shape: {gt_mask.shape}")
            
            # Prepare inputs
            image_input = self.clip_wrapper.encode_image(image_pil)
            text_target = self.clip_wrapper.encode_text(prompt)
            
            # Compute attention map
            heatmap = self.gradcam.compute_heatmap(
                self.clip_wrapper.get_visual_model(),
                image_input,
                text_target,
                self.clip_wrapper.get_target_layer()
            )
            
            print(f"Generated heatmap shape: {heatmap.shape}")
            
            # Compute evaluation metrics
            metrics = self.metrics.compute_metrics(heatmap, gt_mask)
            
            # Normalize image for visualization
            image_norm = original_image.astype(float) / 255.0
            
            return image_norm, heatmap, gt_mask, metrics
            
        except Exception as e:
            print(f"Error processing image {image_info['id']}: {str(e)}")
            return None
    
    def run_experiment(
        self,
        prompt_types: List[str] = ['baseline', 'context', 'caption'],
        save_dir: str = 'results'
    ):
        """Run the complete experiment
        
        Args:
            prompt_types: List of prompt types to test
            save_dir: Directory to save results
            
        Returns:
            Dict: Average metrics for each prompt type
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Get experimental image set
        print("Selecting diverse images...")
        selected_images = self.coco_loader.select_diverse_images()
        
        # Store results
        all_results = {prompt_type: [] for prompt_type in prompt_types}
        
        # Process each image
        for idx, (img_info, objects, captions) in enumerate(tqdm(selected_images)):
            try:
                # Process each object
                for obj in objects:
                    # Test each prompt type
                    for prompt_type in prompt_types:
                        # Generate prompt
                        context = captions[0] if prompt_type == 'caption' else None
                        prompt = self.generate_prompt(obj, context, prompt_type)
                        
                        # Process image
                        result = self.process_single_image(img_info, obj, prompt)
                        if result is None:
                            print(f"Skipping image {img_info['id']} for object {obj}")
                            continue
                            
                        image, heatmap, gt_mask, metrics = result
                        
                        # Save path
                        save_path = os.path.join(
                            save_dir,
                            f"img_{idx}_{obj}_{prompt_type}.png"
                        )
                        
                        # Visualize and save
                        self.visualizer.plot_results(
                            image,
                            heatmap,
                            gt_mask,
                            metrics,
                            f"Object: {obj}, Prompt: {prompt}",
                            save_path
                        )
                        
                        # Record metrics
                        all_results[prompt_type].append(metrics)
                        
            except Exception as e:
                print(f"Error processing image {idx}: {str(e)}")
                continue
                
        # Calculate average metrics
        avg_results = {}
        for prompt_type in prompt_types:
            if not all_results[prompt_type]:
                print(f"No valid results for prompt type: {prompt_type}")
                continue
                
            metrics_list = all_results[prompt_type]
            avg_metrics = {
                metric: np.mean([m[metric] for m in metrics_list])
                for metric in metrics_list[0].keys()
            }
            avg_results[prompt_type] = avg_metrics
            
        return avg_results

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Run experiment
    print("Starting experiment...")
    results = runner.run_experiment()
    
    # Print results
    print("\nAverage Results:")
    for prompt_type, metrics in results.items():
        print(f"\n{prompt_type.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")

if __name__ == "__main__":
    main()