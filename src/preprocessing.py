import cv2
import numpy as np

class AdvancedImagePreprocessor:
    """
    State-of-the-art image preprocessing pipeline for bacterial colony detection
    Handles blurry, distorted, and cluttered microscope images
    """
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
    def enhance_contrast_adaptive(self, image):
        """Advanced contrast enhancement using CLAHE"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            return self.clahe.apply(image)
    
    def denoise_advanced(self, image):
        """Multi-scale denoising for microscope images"""
        # Non-local means denoising
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        # Additional bilateral filtering for edge preservation
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        return denoised
    
    def sharpen_unsharp_mask(self, image):
        """Unsharp masking for enhancing colony boundaries"""
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    
    def normalize_illumination(self, image):
        """Correct uneven illumination common in microscope images"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Estimate background using morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Smooth background
        background = cv2.GaussianBlur(background, (51, 51), 0)
        
        # Normalize
        normalized = cv2.divide(gray, background, scale=255)
        
        if len(image.shape) == 3:
            # Apply correction to all channels
            correction_factor = normalized.astype(np.float32) / gray.astype(np.float32)
            corrected = image.astype(np.float32)
            for i in range(3):
                corrected[:, :, i] = corrected[:, :, i] * correction_factor
            return np.clip(corrected, 0, 255).astype(np.uint8)
        else:
            return normalized
    
    def preprocess_image(self, image):
        """Complete preprocessing pipeline"""
        # Step 1: Illumination correction
        processed = self.normalize_illumination(image)
        
        # Step 2: Denoising
        processed = self.denoise_advanced(processed)
        
        # Step 3: Contrast enhancement
        processed = self.enhance_contrast_adaptive(processed)
        
        # Step 4: Sharpening
        processed = self.sharpen_unsharp_mask(processed)
        
        return processed