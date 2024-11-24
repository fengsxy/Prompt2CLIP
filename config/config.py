# config/config.py

import os

class Config:
    # Get the absolute path of the project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    COCO_ROOT = os.path.join(PROJECT_ROOT, 'coco')  # COCO dataset root directory
    DATA_TYPE = 'val2017'  # Dataset split type
    
    # Annotation file paths
    INSTANCE_ANN_FILE = os.path.join(COCO_ROOT, 'annotations', f'instances_{DATA_TYPE}.json')
    CAPTION_ANN_FILE = os.path.join(COCO_ROOT, 'annotations', f'captions_{DATA_TYPE}.json')
    
    # Image directory
    IMAGE_DIR = os.path.join(COCO_ROOT, DATA_TYPE)
    
    # Model settings
    CLIP_MODEL = 'RN50'  # CLIP model variant
    DEVICE = 'cuda'  # Device for model inference, fallback to 'cpu' if GPU not available
    
    # Experiment settings
    NUM_IMAGES = 500  # Number of images to process
    MIN_OBJECTS = 2   # Minimum number of objects per image
    MAX_OBJECTS = 3   # Maximum number of objects per image
    
    # Evaluation settings
    SALIENCY_LAYER = 'layer4'  # Layer to extract gradients for GradCAM
    BLUR_HEATMAP = True        # Whether to apply Gaussian blur to heatmaps
    
    # Visualization settings
    PLOT_SIZE = (10, 5)        # Figure size for plot outputs
    SAVE_RESULTS = True        # Whether to save visualization results
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')  # Directory to save results
    
    @classmethod
    def get_instance_annfile(cls) -> str:
        """Get the full path to instance annotation file"""
        return cls.INSTANCE_ANN_FILE
    
    @classmethod
    def get_caption_annfile(cls) -> str:
        """Get the full path to caption annotation file"""
        return cls.CAPTION_ANN_FILE
    
    @classmethod
    def get_image_path(cls, image_id: str) -> str:
        """Get the full path for a specific image
        
        Args:
            image_id (str): The ID of the image
            
        Returns:
            str: Full path to the image file
        """
        return os.path.join(cls.IMAGE_DIR, f'{image_id}.jpg')