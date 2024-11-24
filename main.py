import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from typing import List, Dict, Tuple
import json

from config.config import Config
from data.coco_loader import COCOLoader
from models.clip_wrapper import CLIPWrapper
from models.gradcam import GradCAM
from evaluation.metrics import EvaluationMetrics
from evaluation.visualizer import Visualizer
from tools.prompt_generator import PromptGenerator

class ExperimentRunner:
    def __init__(self, prompt_file: str):
        """Initialize experiment runner
        
        Args:
            prompt_file: Path to JSON file containing prompts
        """
        self.config = Config()
        self.coco_loader = COCOLoader(self.config)
        self.clip_wrapper = CLIPWrapper(self.config)
        self.gradcam = GradCAM(self.config)
        self.metrics = EvaluationMetrics()
        self.visualizer = Visualizer(self.config)
        
        # Load prompts
        self.prompt_data = PromptGenerator.load_prompts(prompt_file)
        
    def process_single_image(
        self,
        image_info: Dict,
        object_name: str,
        prompt: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        """Process a single image with given prompt"""
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
            image_pil = image_pil.resize((224, 224))
            
            # Get ground truth mask
            gt_mask = self.coco_loader.get_object_masks(image_info, object_name)
            if gt_mask is None:
                print(f"Failed to get mask for object {object_name}")
                return None
            
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
            
            # Compute evaluation metrics
            metrics = self.metrics.compute_metrics(heatmap, gt_mask)
            
            # Normalize image for visualization
            image_norm = original_image.astype(float) / 255.0
            
            return image_norm, heatmap, gt_mask, metrics
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def run_experiment(self, save_dir: str = 'results'):
        """Run experiment using loaded prompts"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Store results for each prompt type
        all_results = {}
        
        # Process each image
        for image_key, image_data in tqdm(self.prompt_data.items()):
            img_info = self.coco_loader.coco.loadImgs(int(image_data['image_id']))[0]
            
            # Process each object with different prompt types
            for obj in image_data['objects']:
                for prompt_type, prompts in image_data['prompts'].items():
                    if prompt_type not in all_results:
                        all_results[prompt_type] = []
                        
                    # Get specific prompt for this object
                    prompt = prompts[obj]
                    
                    # Process image
                    result = self.process_single_image(img_info, obj, prompt)
                    if result is None:
                        continue
                        
                    image, heatmap, gt_mask, metrics = result
                    
                    # Save visualization
                    save_path = os.path.join(
                        save_dir,
                        f"{image_key}_{obj}_{prompt_type}.png"
                    )
                    
                    self.visualizer.plot_results(
                        image,
                        heatmap,
                        gt_mask,
                        metrics,
                        f"Object: {obj}, Prompt: {prompt}",
                        save_path
                    )
                    
                    # Record metrics
                    all_results[prompt_type].append({
                        'image_id': image_data['image_id'],
                        'object': obj,
                        'prompt': prompt,
                        'metrics': metrics
                    })
        
        # Calculate average metrics
        avg_results = {}
        for prompt_type, results in all_results.items():
            if not results:
                continue
                
            avg_metrics = {
                metric: np.mean([r['metrics'][metric] for r in results])
                for metric in results[0]['metrics'].keys()
            }
            avg_results[prompt_type] = avg_metrics
            
        # Save detailed results
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump({
                'average_metrics': avg_results,
                'detailed_results': all_results
            }, f, indent=4)
            
        return avg_results

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create experiment runner with prompt file
    runner = ExperimentRunner('prompts_template.json')
    
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