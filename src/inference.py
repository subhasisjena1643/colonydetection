import torch
import cv2
import time
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from .model import BacterialColonyDetector
from .preprocessing import AdvancedImagePreprocessor

class InferenceEngine:
    """
    Optimized inference engine for bacterial colony detection
    """
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = BacterialColonyDetector()
        self.load_model(model_path)
        self.preprocessor = AdvancedImagePreprocessor()
        
        # TensorRT optimization (if available)
        self.optimize_model()
    
    def load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def optimize_model(self):
        """Optimize model for inference"""
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Compile model (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model)
        except:
            pass  # Fallback for older PyTorch versions
    
    def preprocess_batch(self, images):
        """Preprocess batch of images"""
        processed_images = []
        
        for image in images:
            # Preprocess
            processed = self.preprocessor.preprocess_image(image)
            
            # Convert to tensor
            if len(processed.shape) == 3:
                processed = torch.from_numpy(processed).permute(2, 0, 1)
            else:
                processed = torch.from_numpy(processed).unsqueeze(0)
            
            # Normalize
            processed = processed.float() / 255.0
            processed_images.append(processed)
        
        return torch.stack(processed_images)
    
    def detect_colonies(self, image_path_or_array, return_visualization=True):
        """Detect bacterial colonies in image"""
        # Load image
        if isinstance(image_path_or_array, (str, Path)):
            image = cv2.imread(str(image_path_or_array))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path_or_array
        
        original_image = image.copy()
        
        # Preprocess
        processed_batch = self.preprocess_batch([image])
        processed_batch = processed_batch.to(self.device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            detections = self.model.predict(processed_batch, self.device)
        inference_time = time.time() - start_time
        
        # Post-process results
        results = {
            'colony_count': len(detections),
            'colonies': detections,
            'inference_time': inference_time,
            'image_shape': image.shape
        }
        
        if return_visualization:
            visualization = self.visualize_detections(original_image, detections)
            results['visualization'] = visualization
        
        return results
    
    def visualize_detections(self, image, detections):
        """Create visualization of detected colonies"""
        vis_image = image.copy()
        
        for i, detection in enumerate(detections):
            x, y = int(detection['x']), int(detection['y'])
            confidence = detection['confidence']
            
            # Draw circle for colony
            cv2.circle(vis_image, (x, y), 8, (0, 255, 0), 2)
            
            # Draw confidence score
            cv2.putText(vis_image, f'{confidence:.2f}', (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw colony number
            cv2.putText(vis_image, str(i+1), (x-5, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add summary text
        cv2.putText(vis_image, f'Total Colonies: {len(detections)}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return vis_image
    
    def batch_process_directory(self, input_dir, output_dir, batch_size=4):
        """Process entire directory of images"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(input_dir.glob('*.jpg')) + \
                     list(input_dir.glob('*.png')) + \
                     list(input_dir.glob('*.tiff')) + \
                     list(input_dir.glob('*.bmp'))
        
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_files), batch_size), desc='Processing images'):
            batch_files = image_files[i:i+batch_size]
            batch_images = []
            
            # Load batch
            for file_path in batch_files:
                image = cv2.imread(str(file_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                batch_images.append(image)
            
            # Process batch
            processed_batch = self.preprocess_batch(batch_images)
            processed_batch = processed_batch.to(self.device)
            
            with torch.no_grad():
                batch_detections = self.model.predict(processed_batch, self.device)
            
            # Save results
            for j, (file_path, detections) in enumerate(zip(batch_files, batch_detections)):
                result = {
                    'image_name': file_path.name,
                    'colony_count': len(detections),
                    'colonies': detections
                }
                results.append(result)
                
                # Save visualization
                vis_image = self.visualize_detections(batch_images[j], detections)
                vis_path = output_dir / f'detected_{file_path.name}'
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Save summary
        summary_path = output_dir / 'detection_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results