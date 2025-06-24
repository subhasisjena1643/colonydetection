import numpy as np
import cv2
import json
from tqdm import tqdm
from pathlib import Path

class DatasetGenerator:
    """
    Generate synthetic bacterial colony datasets for training
    Useful when annotated data is limited
    """
    
    def __init__(self, output_dir, num_images=1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_images = num_images
    
    def generate_synthetic_colony(self, image, x, y, radius, colony_type='circular'):
        """Generate a synthetic bacterial colony"""
        if colony_type == 'circular':
            # Create circular colony
            cv2.circle(image, (x, y), radius, (180, 150, 120), -1)
            # Add texture
            cv2.circle(image, (x, y), radius-2, (160, 130, 100), -1)
            # Add highlight
            cv2.circle(image, (x-2, y-2), radius//3, (200, 170, 140), -1)
        
        elif colony_type == 'irregular':
            # Create irregular colony using ellipse
            angle = np.random.randint(0, 180)
            axes = (radius, int(radius * np.random.uniform(0.7, 1.3)))
            cv2.ellipse(image, (x, y), axes, angle, 0, 360, (180, 150, 120), -1)
    
    def generate_background(self, width=1024, height=1024):
        """Generate realistic agar plate background"""
        # Create base agar color
        background = np.ones((height, width, 3), dtype=np.uint8) * 240
        background[:, :, 0] = 235  # Slight yellowish tint
        background[:, :, 1] = 240
        background[:, :, 2] = 245
        
        # Add texture and variation
        noise = np.random.normal(0, 10, (height, width, 3))
        background = np.clip(background + noise, 0, 255).astype(np.uint8)
        
        # Add scratches and imperfections
        for _ in range(np.random.randint(5, 15)):
            start_point = (np.random.randint(0, width), np.random.randint(0, height))
            end_point = (np.random.randint(0, width), np.random.randint(0, height))
            cv2.line(background, start_point, end_point, (220, 220, 220), 1)
        
        return background
    
    def generate_dataset(self):
        """Generate complete synthetic dataset"""
        annotations = []
        
        for i in tqdm(range(self.num_images), desc='Generating synthetic data'):
            # Generate background
            image = self.generate_background()
            height, width = image.shape[:2]
            
            # Generate random number of colonies
            num_colonies = np.random.randint(10, 100)
            colonies = []
            
            for _ in range(num_colonies):
                # Random position (avoid edges)
                x = np.random.randint(50, width - 50)
                y = np.random.randint(50, height - 50)
                
                # Random size
                radius = np.random.randint(8, 25)
                
                # Check for overlap with existing colonies
                overlap = False
                for existing in colonies:
                    dist = np.sqrt((x - existing['x'])**2 + (y - existing['y'])**2)
                    if dist < (radius + existing['radius'] + 10):
                        overlap = True
                        break
                
                if not overlap:
                    # Generate colony
                    colony_type = np.random.choice(['circular', 'irregular'], p=[0.7, 0.3])
                    self.generate_synthetic_colony(image, x, y, radius, colony_type)
                    
                    colonies.append({
                        'x': x, 'y': y, 'radius': radius, 'confidence': 1.0
                    })
            
            # Save image
            image_name = f'synthetic_{i:05d}.jpg'
            image_path = self.output_dir / image_name
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Save annotation
            annotations.append({
                'image_name': image_name,
                'colonies': colonies
            })
        
        # Save annotations file
        annotations_path = self.output_dir / 'annotations.json'
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Generated {self.num_images} synthetic images with annotations")
        return annotations