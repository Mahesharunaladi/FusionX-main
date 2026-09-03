"""
Embedding Service: Handles CLIP/SigLIP encoding for images and text
"""

import numpy as np
import open_clip
import torch
from PIL import Image
from typing import Union
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages vision-language model embeddings (CLIP/SigLIP)"""
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu"
    ):
        """
        Initialize the embedding service
        
        Args:
            model_name: CLIP model architecture (e.g., "ViT-B-32", "ViT-L-14")
            pretrained: Pretrained weights (e.g., "laion2b_s34b_b79k", "openai")
            device: Compute device ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.embedding_dim = None
        
        logger.info(f"Initializing embedding service with {model_name} on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model and preprocessing pipeline"""
        try:
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
            
            model.eval()
            self.model = model
            self.preprocess = preprocess
            
            # Determine embedding dimension
            self.embedding_dim = int(self.model.text_projection.shape[-1])
            logger.info(f"✓ Model loaded: {self.model_name}, embedding dim: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dim
    
    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings to unit length"""
        return features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    
    def encode_text(self, text: Union[str, list]) -> np.ndarray:
        """
        Encode text query to embedding vector
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Float32 numpy array of shape (1, embedding_dim) or (n, embedding_dim)
        """
        if isinstance(text, str):
            text = [text]
        
        tokens = self.tokenizer(text).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = self._normalize(features)
        
        return features.cpu().numpy().astype("float32")
    
    def encode_image(self, image: Union[Image.Image, str]) -> np.ndarray:
        """
        Encode image to embedding vector
        
        Args:
            image: PIL Image or file path string
            
        Returns:
            Float32 numpy array of shape (embedding_dim,)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image or file path")
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Preprocess and add batch dimension
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features = self._normalize(features)
        
        return features.cpu().numpy().astype("float32")[0]  # Return single embedding
    
    def encode_images_batch(self, images: list, batch_size: int = 16) -> np.ndarray:
        """
        Encode multiple images efficiently
        
        Args:
            images: List of PIL Images or file paths
            batch_size: Number of images to process at once
            
        Returns:
            Float32 numpy array of shape (n_images, embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Convert batch to tensors
            tensors = []
            for img in batch:
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                if img.mode != "RGB":
                    img = img.convert("RGB")
                tensors.append(self.preprocess(img))
            
            batch_tensor = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                features = self._normalize(features)
            
            all_embeddings.append(features.cpu().numpy())
        
        return np.vstack(all_embeddings).astype("float32")
    
    def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1
        """
        # Normalize embeddings if not already normalized
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        return float(np.dot(e1, e2))
