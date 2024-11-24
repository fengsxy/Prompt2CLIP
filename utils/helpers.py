# utils/helpers.py

import os

def validate_coco_setup(config):
    """Validate the COCO dataset setup configuration
    
    This function checks if all required files and directories exist
    and are accessible. It provides detailed feedback about any missing
    components and the current configuration.
    
    Args:
        config: Configuration object containing path settings
        
    Returns:
        bool: True if setup is valid, False otherwise
    """
    errors = []
    
    # Check COCO root directory
    if not os.path.exists(config.COCO_ROOT):
        errors.append(f"COCO root directory not found: {config.COCO_ROOT}")
    
    # Check annotation files
    if not os.path.exists(config.get_instance_annfile()):
        errors.append(f"Instance annotation file not found: {config.get_instance_annfile()}")
    
    if not os.path.exists(config.get_caption_annfile()):
        errors.append(f"Caption annotation file not found: {config.get_caption_annfile()}")
    
    # Check image directory
    if not os.path.exists(config.IMAGE_DIR):
        errors.append(f"Image directory not found: {config.IMAGE_DIR}")
    
    if errors:
        print("COCO Dataset Setup Errors:")
        for error in errors:
            print(f"- {error}")
        print("\nCurrent Configuration:")
        print(f"Project Root: {config.PROJECT_ROOT}")
        print(f"COCO Root: {config.COCO_ROOT}")
        print(f"Annotations Directory: {os.path.dirname(config.get_instance_annfile())}")
        print(f"Image Directory: {config.IMAGE_DIR}")
        return False
    
    # Print successful configuration
    print("COCO Dataset Setup Validated Successfully!")
    print("\nCurrent Path Configuration:")
    print(f"Project Root: {config.PROJECT_ROOT}")
    print(f"COCO Root: {config.COCO_ROOT}")
    print(f"Instance Annotations: {config.get_instance_annfile()}")
    print(f"Caption Annotations: {config.get_caption_annfile()}")
    print(f"Image Directory: {config.IMAGE_DIR}")
    return True

def print_file_structure(config, max_files: int = 5):
    """Print the actual file structure for debugging purposes
    
    Args:
        config: Configuration object containing path settings
        max_files (int): Maximum number of files to list in each directory
    """
    print("\nFile Structure Check:")
    
    # Check annotations
    ann_dir = os.path.dirname(config.get_instance_annfile())
    if os.path.exists(ann_dir):
        print("\nAnnotations Directory Contents:")
        for file in sorted(os.listdir(ann_dir))[:max_files]:
            print(f"  - {file}")
            
    # Check image directory
    if os.path.exists(config.IMAGE_DIR):
        print("\nImage Directory Contents:")
        for file in sorted(os.listdir(config.IMAGE_DIR))[:max_files]:
            print(f"  - {file}")
    
    # Check results directory
    if os.path.exists(config.RESULTS_DIR):
        print("\nResults Directory Contents:")
        for file in sorted(os.listdir(config.RESULTS_DIR))[:max_files]:
            print(f"  - {file}")
            
    if max_files < len(os.listdir(config.IMAGE_DIR)):
        print("  ...")  # Indicate there are more files