import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from tqdm import tqdm

from .loss import FocalLoss, SmoothL1Loss

class TrainingManager:
    """
    Advanced training manager with optimization for RTX 3050 6GB
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.smooth_l1_loss = SmoothL1Loss(beta=1.0)
        
        # Optimizer with gradient clipping
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Mixed precision training for memory efficiency
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.training_history = defaultdict(list)
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with memory optimization"""
        self.model.train()
        epoch_losses = defaultdict(float)
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Memory management: clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            images = batch['image'].to(self.device)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                
                # Calculate losses
                total_loss = 0.0
                cls_loss = 0.0
                reg_loss = 0.0
                
                for scale_output in outputs:
                    # Generate targets for this scale
                    cls_targets, reg_targets, reg_mask = self.generate_targets(
                        batch, scale_output['classification'].shape
                    )
                    
                    # Classification loss
                    scale_cls_loss = self.focal_loss(
                        scale_output['classification'], cls_targets
                    )
                    cls_loss += scale_cls_loss
                    
                    # Regression loss (only for positive samples)
                    scale_reg_loss = self.smooth_l1_loss(
                        scale_output['regression'], reg_targets, reg_mask
                    )
                    reg_loss += scale_reg_loss
                
                total_loss = cls_loss + reg_loss
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update progress
            epoch_losses['total'] += total_loss.item()
            epoch_losses['classification'] += cls_loss.item()
            epoch_losses['regression'] += reg_loss.item()
            
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Cls': f'{cls_loss.item():.4f}',
                'Reg': f'{reg_loss.item():.4f}'
            })
        
        # Update learning rate
        self.scheduler.step()
        
        # Record epoch statistics
        for key, value in epoch_losses.items():
            self.training_history[key].append(value / len(dataloader))
        
        return epoch_losses
    
    def generate_targets(self, batch, output_shape):
        """Generate training targets for detection"""
        batch_size, num_anchors, height, width = output_shape
        
        cls_targets = torch.zeros(output_shape, device=self.device)
        reg_targets = torch.zeros(batch_size, num_anchors * 4, height, width, device=self.device)
        reg_mask = torch.zeros_like(reg_targets, device=self.device)
        
        # This is a simplified target generation
        # In practice, you'd implement anchor matching based on IoU
        
        return cls_targets, reg_targets, reg_mask
    
    def validate(self, dataloader):
        """Validation with metrics calculation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                images = batch['image'].to(self.device)
                
                # Get predictions
                detections = self.model.predict(images, self.device)
                
                # Store for metrics
                all_predictions.extend(detections)
                all_targets.extend(batch['colonies'])
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets)
        return metrics
    
    def calculate_metrics(self, predictions, targets):
        """Calculate detection metrics"""
        # This would implement mAP, precision, recall calculations
        # Simplified for brevity
        metrics = {
            'mAP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        return metrics
    
    def save_checkpoint(self, filepath, epoch, metrics):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'training_history': dict(self.training_history)
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.training_history = defaultdict(list, checkpoint['training_history'])
        return checkpoint['epoch'], checkpoint['metrics']