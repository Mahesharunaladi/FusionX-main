"""
Dataset Loader: Discovers and manages satellite imagery dataset
"""

from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Manages satellite imagery dataset"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    
    def __init__(self, dataset_path: Path):
        """
        Initialize dataset loader
        
        Args:
            dataset_path: Root path to dataset
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        self._images = None
        logger.info(f"Dataset loader initialized: {self.dataset_path}")
    
    @classmethod
    def is_image_file(cls, path: Path) -> bool:
        """Check if file is a supported image"""
        return path.suffix.lower() in cls.SUPPORTED_FORMATS
    
    def discover_images(self) -> List[Path]:
        """
        Recursively discover all images in dataset
        
        Returns:
            List of Path objects for all discovered images
        """
        images = []
        
        for item in self.dataset_path.rglob("*"):
            if item.is_file() and self.is_image_file(item):
                images.append(item)
        
        images.sort()
        logger.info(f"Discovered {len(images)} images in {self.dataset_path}")
        return images
    
    def get_all_images(self) -> List[Path]:
        """
        Get all images in dataset (cached)
        
        Returns:
            List of Path objects for all images
        """
        if self._images is None:
            self._images = self.discover_images()
        return self._images
    
    def refresh(self):
        """Refresh the image list (discover again)"""
        self._images = None
        return self.get_all_images()
    
    def count_images(self) -> int:
        """Get total number of images"""
        return len(self.get_all_images())
    
    def get_image_by_name(self, name: str) -> Path:
        """
        Get image by filename
        
        Args:
            name: Image filename
            
        Returns:
            Path object or None if not found
        """
        for img_path in self.get_all_images():
            if img_path.name == name or img_path.stem == name:
                return img_path
        return None
    
    def get_subdirectory_images(self, subdir: str) -> List[Path]:
        """
        Get all images in a specific subdirectory
        
        Args:
            subdir: Subdirectory name
            
        Returns:
            List of Path objects in that subdirectory
        """
        subdir_path = self.dataset_path / subdir
        if not subdir_path.exists():
            return []
        
        images = []
        for item in subdir_path.rglob("*"):
            if item.is_file() and self.is_image_file(item):
                images.append(item)
        
        images.sort()
        return images
    
    def get_metadata(self) -> dict:
        """Get dataset metadata"""
        images = self.get_all_images()
        
        total_size = sum(img.stat().st_size for img in images) / (1024 * 1024)  # MB
        
        return {
            "dataset_path": str(self.dataset_path),
            "total_images": len(images),
            "total_size_mb": round(total_size, 2),
            "subdirectories": [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
        }
