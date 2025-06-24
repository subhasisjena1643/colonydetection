from src.generator import DatasetGenerator
from src.trainer import TrainingManager
from src.dataset import BacterialColonyDataset, create_training_transforms, create_validation_transforms
from src.inference import InferenceEngine
from src.model import BacterialColonyDetector

from torch.utils.data import DataLoader
from pathlib import Path
import torch
import numpy as np

def train_model(config):
    # Dataset and Dataloaders
    train_dataset = BacterialColonyDataset(
        image_dir=config['train_dir'],
        annotation_file=config['train_annotations'],
        transform=create_training_transforms()
    )
    val_dataset = BacterialColonyDataset(
        image_dir=config['val_dir'],
        annotation_file=config['val_annotations'],
        transform=create_validation_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Model & Trainer
    model = BacterialColonyDetector()
    trainer = TrainingManager(model, device='cuda')

    best_map = 0.0
    for epoch in range(1, config['num_epochs'] + 1):
        trainer.train_epoch(train_loader, epoch)

        if epoch % config['val_frequency'] == 0:
            metrics = trainer.validate(val_loader)
            map_score = metrics.get('mAP', 0.0)

            if map_score > best_map:
                best_map = map_score
                trainer.save_checkpoint(f"{config['output_dir']}/best_model.pth", epoch, metrics)
    
    print("Training completed.")
    return trainer

def main():
    config = {
        'train_dir': 'data/train/images',
        'val_dir': 'data/val/images',
        'train_annotations': 'data/train/annotations.json',
        'val_annotations': 'data/val/annotations.json',
        'output_dir': 'outputs',
        'batch_size': 8,
        'num_epochs': 100,
        'val_frequency': 5,
        'learning_rate': 1e-4
    }

    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)

    if not Path(config['train_dir']).exists():
        print("[INFO] Synthetic dataset not found. Generating now...")
        generator = DatasetGenerator('data/train', num_images=800)
        generator.generate_dataset()

        generator = DatasetGenerator('data/val', num_images=200)
        generator.generate_dataset()

    trainer = train_model(config)

    model_path = Path(config['output_dir']) / 'best_model.pth'
    if model_path.exists():
        inference_engine = InferenceEngine(model_path)
        test_img = Path("data/test/sample.jpg")
        if test_img.exists():
            results = inference_engine.detect_colonies(str(test_img))
            print(f"[INFERENCE] Detected: {results['colony_count']} colonies in {results['inference_time']:.3f}s")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()