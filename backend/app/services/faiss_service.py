"""
FAISS Service: Manages vector indexing and semantic search
"""

import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging
import json

logger = logging.getLogger(__name__)


class FAISSService:
    """Manages FAISS vector index for efficient semantic search"""
    
    def __init__(self, embedding_dim: int = 512, index_path: Path = None):
        """
        Initialize FAISS service
        
        Args:
            embedding_dim: Dimension of embedding vectors
            index_path: Path to save/load index files
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path or Path("./indices")
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.image_paths = []
        self.path_to_id = {}
        
        self.index_file = self.index_path / "embeddings.index"
        self.metadata_file = self.index_path / "metadata.json"
        
        logger.info(f"Initialized FAISS service with dim={embedding_dim}")
    
    def _create_index(self):
        """Create a new HNSW index for efficient similarity search"""
        # Use HNSW (Hierarchical Navigable Small World) for fast ANN search
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexHNSWFlat(quantizer, self.embedding_dim)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 64
        logger.info("✓ Created new HNSW index")
    
    def add_embeddings(self, embeddings: np.ndarray, paths: List[Path]):
        """
        Add embeddings to the index
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            paths: List of Path objects corresponding to each embedding
        """
        if self.index is None:
            self._create_index()
        
        # Ensure embeddings are float32
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        # Add to index
        start_idx = self.index.ntotal if self.index else 0
        self.index.add(embeddings)
        
        # Update path mappings
        for i, path in enumerate(paths):
            idx = start_idx + i
            self.image_paths.append(path)
            self.path_to_id[str(path)] = idx
        
        logger.info(f"Added {len(paths)} embeddings to index (total: {self.index.ntotal})")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
        """
        Search for top-k similar embeddings
        
        Args:
            query_embedding: Query embedding of shape (1, embedding_dim) or (embedding_dim,)
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, paths)
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return np.array([]), np.array([]), []
        
        # Reshape if necessary
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        k = min(k, self.index.ntotal)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert distances to similarity scores (1 / (1 + distance))
        similarities = 1.0 / (1.0 + distances[0])
        
        # Get paths
        result_paths = [self.image_paths[int(idx)] for idx in indices[0]]
        
        return similarities, indices[0], result_paths
    
    def batch_search(self, query_embeddings: np.ndarray, k: int = 5) -> List[Tuple[np.ndarray, np.ndarray, List[Path]]]:
        """
        Search for multiple queries
        
        Args:
            query_embeddings: Array of shape (n_queries, embedding_dim)
            k: Number of results per query
            
        Returns:
            List of (similarities, indices, paths) tuples
        """
        results = []
        for i in range(len(query_embeddings)):
            sims, inds, paths = self.search(query_embeddings[i:i+1], k)
            results.append((sims, inds, paths))
        return results
    
    def get_index_size(self) -> int:
        """Get number of vectors in index"""
        return self.index.ntotal if self.index else 0
    
    def get_index_size_mb(self) -> float:
        """Get approximate index size in megabytes"""
        if self.index is None:
            return 0.0
        # Rough estimate: each float32 is 4 bytes
        size_bytes = self.index.ntotal * self.embedding_dim * 4
        return size_bytes / (1024 * 1024)
    
    def index_exists(self) -> bool:
        """Check if index files exist"""
        return self.index_file.exists() and self.metadata_file.exists()
    
    def save_index(self):
        """Save index to disk"""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            metadata = {
                "embedding_dim": self.embedding_dim,
                "n_vectors": self.index.ntotal,
                "image_paths": [str(p) for p in self.image_paths],
                "path_to_id": self.path_to_id
            }
            
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Index saved to {self.index_file}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load_index(self):
        """Load index from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)
            
            self.image_paths = [Path(p) for p in metadata["image_paths"]]
            self.path_to_id = metadata["path_to_id"]
            
            logger.info(f"✓ Index loaded: {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def clear(self):
        """Clear the index"""
        self._create_index()
        self.image_paths = []
        self.path_to_id = {}
        logger.info("Index cleared")
    
    def remove_embedding(self, path: Path):
        """Remove embedding for a specific image (not efficient with HNSW)"""
        if str(path) in self.path_to_id:
            del self.path_to_id[str(path)]
            self.image_paths = [p for p in self.image_paths if p != path]
            logger.info(f"Removed: {path}")
