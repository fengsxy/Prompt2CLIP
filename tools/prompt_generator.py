import json
from typing import List, Dict
from pycocotools.coco import COCO
import os
import random
from tqdm import tqdm

class SimpleConfig:
    # Get project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    COCO_ROOT = os.path.join(PROJECT_ROOT, 'coco')
    DATA_TYPE = 'val2017'
    
    INSTANCE_ANN_FILE = os.path.join(COCO_ROOT, 'annotations', f'instances_{DATA_TYPE}.json')
    CAPTION_ANN_FILE = os.path.join(COCO_ROOT, 'annotations', f'captions_{DATA_TYPE}.json')
    
    # Number of images to process
    NUM_IMAGES = 5
    MIN_OBJECTS = 2
    MAX_OBJECTS = 3

class PromptGenerator:
    def __init__(self):
        """Initialize prompt generator"""
        self.config = SimpleConfig()
        self.coco = COCO(self.config.INSTANCE_ANN_FILE)
        self.coco_caps = COCO(self.config.CAPTION_ANN_FILE)
    
    def get_filtered_images(self) -> List[Dict]:
        """Get images that contain 2-3 objects
        
        Returns:
            List[Dict]: Filtered image information
        """
        all_images = []
        # Get all image ids
        img_ids = self.coco.getImgIds()
        
        print("Filtering images with 2-3 objects...")
        for img_id in tqdm(img_ids):
            # Get annotations for this image
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Get unique object categories in this image
            objects = set(self.coco.loadCats(cat_id)[0]['name'] 
                         for cat_id in [ann['category_id'] for ann in anns])
            selected_obj = "skis"
            if not (selected_obj in objects):
                continue
            # Check if number of objects is in our range
            if self.config.MIN_OBJECTS <= len(objects) <= self.config.MAX_OBJECTS:
                img_info = self.coco.loadImgs(img_id)[0]
                all_images.append((img_info, list(objects)))
        
        print(f"Found {len(all_images)} images with {self.config.MIN_OBJECTS}-{self.config.MAX_OBJECTS} objects")
        
        # Randomly select images if we have more than we need
        if len(all_images) > self.config.NUM_IMAGES:
            selected_images = random.sample(all_images, self.config.NUM_IMAGES)
        else:
            selected_images = all_images
            
        return [img_info for img_info, _ in selected_images]
        
    def generate_prompt_file(
        self,
        output_path: str = 'prompts.json'
    ):
        """Generate a JSON file containing prompts for filtered images
        
        Args:
            output_path: Path to save the prompt file
        """
        # Get filtered images
        selected_images = self.get_filtered_images()
        prompt_data = {}
        
        print("Generating prompts...")
        for img_info in tqdm(selected_images):
            img_id = str(img_info['id'])
            
            # Get image path
            relative_path = os.path.join('coco', self.config.DATA_TYPE, img_info['file_name'])
            
            # Get captions
            ann_ids = self.coco_caps.getAnnIds(imgIds=img_info['id'])
            captions = [ann['caption'] for ann in self.coco_caps.loadAnns(ann_ids)]
            
            # Get objects
            ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
            anns = self.coco.loadAnns(ann_ids)
            objects = list(set(self.coco.loadCats(cat_id)[0]['name'] 
                            for cat_id in [ann['category_id'] for ann in anns]))
            
            # Skip if somehow we got an image with wrong number of objects
            if not (self.config.MIN_OBJECTS <= len(objects) <= self.config.MAX_OBJECTS):
                continue
            
            # Create empty prompt templates
            prompt_data[f"image_{img_id}"] = {
                "image_id": img_id,
                "image_path": relative_path,
                "captions": captions,
                "objects": objects,
                "prompts": {
                    "baseline": {
                        obj: obj for obj in objects
                    },
                    "context": {
                        obj: f"{obj} in the scene" for obj in objects
                    },
                    "complex_context": {
                        obj: f"a detailed view of the {obj} in its environment" for obj in objects
                    },
                    "hallucination": {
                        obj: f"an imaginary {obj} with supernatural features" for obj in objects
                    }
                }
            }
        
        print(f"\nGenerated prompts for {len(prompt_data)} images")
        print(f"Sample objects distribution:")
        obj_counts = {}
        for data in prompt_data.values():
            num_objs = len(data['objects'])
            obj_counts[num_objs] = obj_counts.get(num_objs, 0) + 1
        for num_objs, count in sorted(obj_counts.items()):
            print(f"Images with {num_objs} objects: {count}")
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, indent=4)
            
        print(f"\nPrompt file saved to: {os.path.abspath(output_path)}")
            
    @staticmethod
    def load_prompts(prompt_file: str) -> Dict:
        """Load prompts from JSON file
        
        Args:
            prompt_file: Path to prompt JSON file
            
        Returns:
            Dict: Loaded prompt data
        """
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return json.load(f)

def main():
    """Generate template prompt file"""
    random.seed(42)  # For reproducibility
    
    generator = PromptGenerator()
    
    # Generate prompt file
    generator.generate_prompt_file('prompts_template.json')
    
if __name__ == "__main__":
    main()