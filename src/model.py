import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extractor using EfficientNet backbone
    Optimized for RTX 3050 6GB memory constraints
    """
    
    def __init__(self):
        super().__init__()
        
        # Use EfficientNet-B4 as backbone (good balance of accuracy/memory)
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        
        # Remove classifier layers
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Multi-scale feature fusion
        self.fpn_conv1 = nn.Conv2d(1792, 256, 1)  # EfficientNet-B4 final feature dim
        self.fpn_conv2 = nn.Conv2d(448, 256, 1)   # From layer 6
        self.fpn_conv3 = nn.Conv2d(160, 256, 1)   # From layer 4
        self.fpn_conv4 = nn.Conv2d(56, 256, 1)    # From layer 2
        
        # Upsampling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Feature refinement
        self.refine_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.batch_norm = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        features = []
        
        # Extract multi-scale features
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [2, 4, 6, 8]:  # Collect features at different scales
                features.append(x)
        
        # Feature Pyramid Network (FPN) style fusion
        p5 = self.fpn_conv1(features[3])  # Highest level
        p4 = self.fpn_conv2(features[2]) + self.upsample(p5)
        p3 = self.fpn_conv3(features[1]) + self.upsample(p4)
        p2 = self.fpn_conv4(features[0]) + self.upsample(p3)
        
        # Refine features
        p2 = self.relu(self.batch_norm(self.refine_conv(p2)))
        p3 = self.relu(self.batch_norm(self.refine_conv(p3)))
        p4 = self.relu(self.batch_norm(self.refine_conv(p4)))
        p5 = self.relu(self.batch_norm(self.refine_conv(p5)))
        
        return [p2, p3, p4, p5]

class ColonyDetectionHead(nn.Module):
    """
    Advanced detection head for bacterial colonies
    Uses both classification and regression for precise localization
    """
    
    def __init__(self, feature_dim=256, num_anchors=9):
        super().__init__()
        
        # Classification branch (colony/background)
        self.cls_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, num_anchors, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Regression branch (precise location and size)
        self.reg_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, num_anchors * 4, 3, padding=1)  # x, y, w, h
        )
        
        # Attention mechanism for hard examples
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Classification and regression
        cls_output = self.cls_conv(attended_features)
        reg_output = self.reg_conv(attended_features)
        
        return cls_output, reg_output, attention_weights

class BacterialColonyDetector(nn.Module):
    """
    Complete bacterial colony detection model
    Optimized for high precision detection of blurry/cluttered colonies
    """
    
    def __init__(self, num_anchors=9):
        super().__init__()
        
        self.feature_extractor = MultiScaleFeatureExtractor()
        self.detection_heads = nn.ModuleList([
            ColonyDetectionHead(256, num_anchors) for _ in range(4)
        ])
        
        # Post-processing parameters
        self.nms_threshold = 0.3
        self.confidence_threshold = 0.5
        
    def forward(self, x):
        # Extract multi-scale features
        features = self.feature_extractor(x)
        
        # Apply detection heads
        outputs = []
        for i, head in enumerate(self.detection_heads):
            cls_out, reg_out, att_out = head(features[i])
            outputs.append({
                'classification': cls_out,
                'regression': reg_out,
                'attention': att_out,
                'scale': i
            })
        
        return outputs
    
    def predict(self, x, device='cuda'):
        """Inference with post-processing"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            detections = self.post_process_detections(outputs, x.shape)
        return detections
    
    def post_process_detections(self, outputs, input_shape):
        """Advanced post-processing with NMS and filtering"""
        all_detections = []
        
        for scale_output in outputs:
            cls_pred = scale_output['classification']
            reg_pred = scale_output['regression']
            
            # Convert predictions to detections
            detections = self.decode_predictions(cls_pred, reg_pred, input_shape)
            all_detections.extend(detections)
        
        # Apply Non-Maximum Suppression
        final_detections = self.apply_nms(all_detections)
        
        return final_detections
    
    def decode_predictions(self, cls_pred, reg_pred, input_shape):
        """Convert network predictions to detection boxes"""
        batch_size, _, height, width = cls_pred.shape
        detections = []
        
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    for a in range(cls_pred.shape[1]):
                        confidence = cls_pred[b, a, h, w].item()
                        
                        if confidence > self.confidence_threshold:
                            # Decode regression predictions
                            dx = reg_pred[b, a*4, h, w].item()
                            dy = reg_pred[b, a*4+1, h, w].item()
                            dw = reg_pred[b, a*4+2, h, w].item()
                            dh = reg_pred[b, a*4+3, h, w].item()
                            
                            # Convert to image coordinates
                            x = (w + dx) * (input_shape[3] / width)
                            y = (h + dy) * (input_shape[2] / height)
                            
                            detections.append({
                                'x': x, 'y': y,
                                'width': dw, 'height': dh,
                                'confidence': confidence
                            })
        
        return detections
    
    def apply_nms(self, detections):
        """Non-Maximum Suppression to remove duplicate detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Apply NMS
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            detections = [d for d in detections 
                         if self.calculate_iou(best, d) < self.nms_threshold]
        
        return keep
    
    def calculate_iou(self, det1, det2):
        """Calculate Intersection over Union"""
        x1_min = det1['x'] - det1['width'] / 2
        y1_min = det1['y'] - det1['height'] / 2
        x1_max = det1['x'] + det1['width'] / 2
        y1_max = det1['y'] + det1['height'] / 2
        
        x2_min = det2['x'] - det2['width'] / 2
        y2_min = det2['y'] - det2['height'] / 2
        x2_max = det2['x'] + det2['width'] / 2
        y2_max = det2['y'] + det2['height'] / 2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0