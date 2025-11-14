import logging

import numpy as np
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generate embeddings using BGE-M3 model.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = False):
        """
        Initialize BGE-M3 embedding model.
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.dimension = 1024
        logger.info(f"Model loaded (dimension={self.dimension})")

    def embed_documents(self, texts: list[str], batch_size: int = 12) -> np.ndarray:
        """
        Embed document texts (for indexing).
        """
        logger.info(f"Embedding {len(texts)} documents (batch_size={batch_size})")

        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,  # BGE-M3 supports up to 8192 tokens
        )

        dense_embeddings: np.ndarray = np.array(output['dense_vecs'])
        dense_embeddings = dense_embeddings / np.linalg.norm(
            dense_embeddings, axis=1, keepdims=True
        )

        logger.info(f"✓ Generated embeddings: shape={dense_embeddings.shape}")
        return dense_embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query (for retrieval).
        """
        output = self.model.encode([query], batch_size=1, max_length=8192)

        # Extract dense embedding
        embedding: np.ndarray = np.array(output['dense_vecs'][0])
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
