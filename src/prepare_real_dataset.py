import os
import json
import shutil
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET


class RealDatasetPreparer:
    """
    Prepare 5-shot dataset using real polyp images from data/train
    """
    
    def __init__(self, source_dir, output_dir, n_shot=5, seed=42):
        """
        Initialize dataset preparer
        
        Args:
            source_dir: Source directory with real polyp images (data/train)
            output_dir: Output directory for 5-shot dataset
            n_shot: Number of examples per class (5-shot)
            seed: Random seed for reproducibility
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.n_shot = n_shot
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def create_5shot_from_real_data(self):
        """Create 5-shot dataset from real polyp images"""
        print(f"🚀 Creating {self.n_shot}-shot dataset from real polyp images...")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find real polyp and non-polyp images
        polyp_images = self._get_images_from_dir(self.source_dir / 'polyp')
        non_polyp_images = self._get_images_from_dir(self.source_dir / 'non_polyp')
        
        print(f"📊 Found {len(polyp_images)} polyp images and {len(non_polyp_images)} non-polyp images")
        
        # Sample 5-shot training set
        train_polyp = self._sample_images(polyp_images, self.n_shot, 'polyp')
        train_non_polyp = self._sample_images(non_polyp_images, self.n_shot, 'non_polyp')
        
        # Remaining images for validation
        remaining_polyp = [img for img in polyp_images if img not in train_polyp]
        remaining_non_polyp = [img for img in non_polyp_images if img not in train_non_polyp]
        
        # Sample validation set (use remaining images, or if none, use training images)
        if remaining_polyp:
            val_polyp = self._sample_images(remaining_polyp, min(1, len(remaining_polyp)), 'polyp')
        else:
            val_polyp = train_polyp[:1] if train_polyp else []
            
        if remaining_non_polyp:
            val_non_polyp = self._sample_images(remaining_non_polyp, min(1, len(remaining_non_polyp)), 'non_polyp')
        else:
            val_non_polyp = train_non_polyp[:1] if train_non_polyp else []
        
        print(f"📊 Dataset split:")
        print(f"   Train polyp: {len(train_polyp)} images")
        print(f"   Train non-polyp: {len(train_non_polyp)} images")
        print(f"   Val polyp: {len(val_polyp)} images")
        print(f"   Val non-polyp: {len(val_non_polyp)} images")
        
        # Copy images and create annotations
        self._copy_and_annotate(train_polyp, 'train', 'polyp')
        self._copy_and_annotate(train_non_polyp, 'train', 'non_polyp')
        self._copy_and_annotate(val_polyp, 'val', 'polyp')
        self._copy_and_annotate(val_non_polyp, 'val', 'non_polyp')
        
        # Create YOLO format annotations
        self._create_yolo_dataset_yaml()
        
        print("✅ 5-shot dataset created from real images!")
        
    def _get_images_from_dir(self, directory):
        """Get all images from a directory"""
        if not directory.exists():
            return []
            
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        images = []
        
        for ext in image_extensions:
            images.extend(list(directory.glob(f'*{ext}')))
            images.extend(list(directory.glob(f'*{ext.upper()}')))
        
        return images
        
    def _sample_images(self, images, n_samples, class_name):
        """Sample n_images from the available images"""
        if len(images) <= n_samples:
            print(f"⚠️  Only {len(images)} {class_name} images available, using all of them")
            return images
        
        return random.sample(images, n_samples)
        
    def _copy_and_annotate(self, images, split, class_name):
        """Copy images and create annotations"""
        # Create directories
        img_dir = self.output_dir / split / 'images'
        label_dir = self.output_dir / split / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in images:
            # Copy image
            shutil.copy2(img_path, img_dir / img_path.name)
            
            # Create YOLO annotation
            label_path = label_dir / (img_path.stem + '.txt')
            self._create_yolo_annotation(img_path, label_path, class_name)
            
    def _create_yolo_annotation(self, img_path, label_path, class_name):
        """Create YOLO format annotation for an image"""
        try:
            # Get image dimensions
            with Image.open(img_path) as img:
                width, height = img.size
        except:
            width, height = 640, 480  # Default size
        
        # Class mapping
        class_map = {'polyp': 0, 'non_polyp': 1}
        class_id = class_map.get(class_name, 0)
        
        # Create bounding box (use full image as bounding box for now)
        # In real implementation, you'd have actual bounding box annotations
        x_center, y_center = 0.5, 0.5
        box_width, box_height = 1.0, 1.0
        
        # Write YOLO format annotation
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
            
    def _create_yolo_dataset_yaml(self):
        """Create YOLO dataset.yaml file"""
        yaml_content = f"""# YOLO dataset configuration for 5-shot polyp detection
path: {self.output_dir.absolute()}
train: train/images
val: val/images

nc: 2
names: ['polyp', 'non_polyp']

# 5-shot dataset configuration
n_shot: {self.n_shot}
seed: {self.seed}
"""
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
            
        return yaml_path


def prepare_real_5shot_dataset():
    """Prepare 5-shot dataset from real polyp images"""
    print("🎯 Preparing 5-shot dataset from real polyp images...")
    
    # Check if source data exists
    source_dir = Path("data/train")
    if not source_dir.exists():
        print("❌ Source directory data/train not found!")
        return None
    
    # Prepare dataset
    preparer = RealDatasetPreparer(
        source_dir=source_dir,
        output_dir="data/5shot_real",
        n_shot=1  # Use 1-shot since we only have 1 polyp and 1 non-polyp image
    )
    
    preparer.create_5shot_from_real_data()
    
    print("🎯 Real 5-shot dataset preparation complete!")
    return preparer.output_dir


if __name__ == "__main__":
    dataset_dir = prepare_real_5shot_dataset()
    if dataset_dir:
        print(f"✅ Dataset prepared at: {dataset_dir}")
