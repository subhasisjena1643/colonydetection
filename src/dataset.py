import os
import json
import cv2
from pathlib import Path
from torch.utils.data import Dataset
from .preprocessing import AdvancedImagePreprocessor
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BacterialColonyDataset(Dataset):
    """
    Advanced dataset class for bacterial colony detection
    Supports various annotation formats and advanced augmentations
    """
    
    def __init__(self, image_dir, annotation_file=None, transform=None, mode='train'):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.mode = mode
        self.preprocessor = AdvancedImagePreprocessor()
        
        # Load images
        self.image_paths = list(self.image_dir.glob('*.jpg')) + \
                          list(self.image_dir.glob('*.png')) + \
                          list(self.image_dir.glob('*.tiff')) + \
                          list(self.image_dir.glob('*.bmp'))
        
        # Load annotations if provided
        self.annotations = {}
        if annotation_file and os.path.exists(annotation_file):
            self.load_annotations(annotation_file)
    
    def load_annotations(self, annotation_file):
        """Load annotations in various formats (COCO, Pascal VOC, custom JSON)"""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Convert to internal format
        for item in data:
            image_name = item['image_name']
            colonies = []
            for colony in item['colonies']:
                colonies.append({
                    'x': colony['x'],
                    'y': colony['y'],
                    'radius': colony.get('radius', 10),
                    'confidence': colony.get('confidence', 1.0)
                })
            self.annotations[image_name] = colonies
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess the image
        image = self.preprocessor.preprocess_image(image)
        
        # Get annotations if available
        image_name = image_path.name
        colonies = self.annotations.get(image_name, [])
        
        if self.transform:
            if colonies:  # If we have annotations, apply augmentations carefully
                # Create keypoints for augmentation
                keypoints = [(c['x'], c['y']) for c in colonies]
                transformed = self.transform(image=image, keypoints=keypoints)
                image = transformed['image']
                
                # Update colony positions
                new_colonies = []
                for i, (x, y) in enumerate(transformed['keypoints']):
                    new_colonies.append({
                        'x': x, 'y': y,
                        'radius': colonies[i]['radius'],
                        'confidence': colonies[i]['confidence']
                    })
                colonies = new_colonies
            else:
                image = self.transform(image=image)['image']
        
        return {
            'image': image,
            'colonies': colonies,
            'image_name': image_name,
            'colony_count': len(colonies)
        }

def create_training_transforms():
    """Create advanced augmentation pipeline for training"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.3),
        A.OpticalDistortion(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2() 
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def create_validation_transforms():
    """Create validation transforms"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])