# data/coco_loader.py

import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image
from typing import List, Dict, Tuple

class COCOLoader:
    def __init__(self, config):
        self.config = config
        self.coco = COCO(config.get_instance_annfile())
        self.coco_caps = COCO(config.get_caption_annfile())
        
    def get_categories(self) -> List[str]:
        """获取所有类别名称"""
        cats = self.coco.loadCats(self.coco.getCatIds())
        return [cat['name'] for cat in cats]
    
    def get_images_with_objects(self, object_names: List[str]) -> List[Dict]:
        """获取包含指定物体的图片"""
        cat_ids = self.coco.getCatIds(catNms=object_names)
        img_ids = self.coco.getImgIds(catIds=cat_ids)
        return self.coco.loadImgs(img_ids)
    
    def load_image(self, img_info: Dict) -> np.ndarray:
        """加载图片"""
        I = io.imread(img_info['coco_url'])
        return I
    
    def get_annotations(self, img_id: int, cat_ids: List[int]) -> List[Dict]:
        """获取图片的标注信息"""
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
        return self.coco.loadAnns(ann_ids)
    
    def get_captions(self, img_id: int) -> List[str]:
        """获取图片的描述文本"""
        ann_ids = self.coco_caps.getAnnIds(imgIds=img_id)
        anns = self.coco_caps.loadAnns(ann_ids)
        return [ann['caption'] for ann in anns]
    
    def select_diverse_images(self) -> List[Tuple[Dict, List[str], List[str]]]:
        """选择多样化的图片集合"""
        categories = self.get_categories()
        selected_images = []
        
        while len(selected_images) < self.config.NUM_IMAGES:
            # 随机选择2-3个物体类别
            num_objects = np.random.randint(
                self.config.MIN_OBJECTS, 
                self.config.MAX_OBJECTS + 1
            )
            selected_cats = np.random.choice(
                categories, 
                size=num_objects, 
                replace=False
            )
            
            # 获取包含这些物体的图片
            images = self.get_images_with_objects(selected_cats)
            if not images:
                continue
                
            # 随机选择一张图片
            img = np.random.choice(images)
            captions = self.get_captions(img['id'])
            
            selected_images.append((img, list(selected_cats), captions))
            
        return selected_images
    
    def get_object_masks(self, img_info: Dict, object_name: str) -> np.ndarray:
        """获取指定物体的掩码"""
        cat_ids = self.coco.getCatIds(catNms=[object_name])
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        
        mask = np.zeros((img_info['height'], img_info['width']))
        for ann in anns:
            curr_mask = self.coco.annToMask(ann)
            mask = np.maximum(mask, curr_mask)
            
        return mask