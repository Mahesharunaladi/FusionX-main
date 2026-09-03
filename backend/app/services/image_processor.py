"""
Image Processor: Handles image loading, preprocessing, and cloud masking
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image preprocessing for satellite imagery"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    TARGET_SIZE = (224, 224)
    
    def __init__(self, target_size: tuple = (224, 224)):
        """
        Initialize image processor
        
        Args:
            target_size: Target image size for model input
        """
        self.target_size = target_size
        logger.info(f"Image processor initialized with target size: {target_size}")
    
    @staticmethod
    def is_supported_format(file_path: Union[str, Path]) -> bool:
        """Check if file is a supported image format"""
        return Path(file_path).suffix.lower() in ImageProcessor.SUPPORTED_FORMATS
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[Image.Image]:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object or None if loading failed
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                logger.warning(f"Image file not found: {image_path}")
                return None
            
            if not self.is_supported_format(image_path):
                logger.warning(f"Unsupported format: {image_path}")
                return None
            
            # Open image
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to target size
        if image.size != self.target_size:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        return image
    
    def normalize(self, image_array: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            Normalized image array
        """
        if image_array.dtype == np.uint8:
            return image_array.astype(np.float32) / 255.0
        return image_array.astype(np.float32)
    
    def cloud_mask(self, image: Image.Image) -> np.ndarray:
        """
        Simple cloud detection based on brightness
        Returns mask where 1 = cloud, 0 = clear
        
        Args:
            image: PIL Image object
            
        Returns:
            Binary mask array
        """
        img_array = np.array(image)
        
        # Simple heuristic: very bright pixels are likely clouds
        brightness = np.mean(img_array, axis=2)
        cloud_threshold = np.percentile(brightness, 95)
        
        mask = (brightness > cloud_threshold).astype(np.uint8)
        return mask
    
    def get_ndvi(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Calculate NDVI (Normalized Difference Vegetation Index)
        Useful for vegetation/land cover analysis
        
        Args:
            image: PIL Image (should have NIR band for accurate NDVI)
            
        Returns:
            NDVI array or None if not possible
        """
        img_array = np.array(image, dtype=np.float32)
        
        # For RGB images, use approximation
        # NDVI ≈ (R - G) / (R + G)
        if img_array.shape[2] >= 3:
            red = img_array[:, :, 0]
            green = img_array[:, :, 1]
            
            ndvi = (red - green) / (red + green + 1e-8)
            return ndvi
        
        return None
    
    def get_image_stats(self, image: Image.Image) -> dict:
        """
        Get basic statistics about image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with image statistics
        """
        img_array = np.array(image, dtype=np.float32)
        
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "mean_brightness": float(np.mean(img_array)),
            "std_brightness": float(np.std(img_array)),
            "min_pixel": float(np.min(img_array)),
            "max_pixel": float(np.max(img_array))
        }
    
    def create_thumbnail(self, image: Image.Image, size: tuple = (128, 128)) -> Image.Image:
        """
        Create thumbnail of image
        
        Args:
            image: PIL Image object
            size: Thumbnail size
            
        Returns:
            Thumbnail image
        """
        img_copy = image.copy()
        img_copy.thumbnail(size, Image.Resampling.LANCZOS)
        return img_copy
