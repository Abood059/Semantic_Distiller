import logging
import numpy as np
import faiss
import time

class FPSSampler:
    def __init__(self, metric: str = "cosine"):
        """Initialize Farthest Point Sampling sampler.
        
        Args:
            metric: Distance metric to use (currently only "cosine" supported)
        """
        self.logger = logging.getLogger(__name__)
        self.metric = metric
        if metric != "cosine":
            self.logger.warning(f"Metric '{metric}' not fully supported, using cosine")
    
    def select(self, embeddings: list[list[float]], k: int) -> list[int]:
        """
        Select k diverse embeddings using Farthest Point Sampling with FAISS.
        
        Args:
            embeddings: List of embedding vectors (each as list of floats)
            k: Number of samples to select
            
        Returns:
            List of indices of selected embeddings
            
        Raises:
            ValueError: If k > len(embeddings)
        """
        if k > len(embeddings):
            raise ValueError(f"k ({k}) cannot exceed number of embeddings ({len(embeddings)})")
        
        if k == 0:
            return []
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Build FAISS index for inner product (equivalent to cosine for normalized vectors)
        index = faiss.IndexFlatIP(embeddings_array.shape[1])
        index.add(embeddings_array)
        
        # Random first point
        np.random.seed(int(time.time() * 1000) % 2**32)
        selected_indices = [np.random.randint(len(embeddings))]        
        # Iteratively select farthest points
        for _ in range(k - 1):
            # Get similarities of all points to last selected point
            last_selected = np.array([embeddings_array[selected_indices[-1]]])
            similarities, _ = index.search(last_selected, len(embeddings))
            similarities = similarities[0]
            
            # For already selected points, set similarity to 1 (minimum distance)
            similarities[selected_indices] = 1.0
            
            # Select point with minimum similarity (maximum distance)
            next_idx = np.argmin(similarities)
            selected_indices.append(int(next_idx))
        
        self.logger.info(f"Selected {len(selected_indices)} diverse samples from {len(embeddings)} embeddings")
        return selected_indices
