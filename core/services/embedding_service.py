import logging
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str, device: str = None):
        """Load SentenceTransformer model for embedding generation."""
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Loading embedding model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate normalized embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each as list of floats)
        """
        with torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        return embeddings.tolist()
    
    def cleanup(self) -> None:
        """Optional model cleanup to free memory."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Embedding service cleaned up")
