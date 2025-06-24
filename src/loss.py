import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in detection
    Focuses training on hard examples
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss for regression targets"""
    
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred, target, mask=None):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta,
                          0.5 * diff ** 2 / self.beta,
                          diff - 0.5 * self.beta)
        
        if mask is not None:
            loss = loss * mask
        
        return loss.mean()