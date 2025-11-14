import logging
import pickle
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from src.indexing.embedder import Embedder

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combine BM25 (lexical) and vector (semantic) search with RRF fusion.
    Uses Reciprocal Rank Fusion (RRF) to merge results without score normalization.
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_model: str,
        collection_name: str = "boeing_737",
    ):
        """
        Initialize hybrid retriever.
        """
        self.persist_dir = Path(persist_dir)
        self.embedder = Embedder(embedding_model, use_fp16=False)

        # Load ChromaDB
        logger.info(f"Loading ChromaDB from {self.persist_dir}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(collection_name)

        # Load BM25
        logger.info("Loading BM25 index")
        self._load_bm25()

        logger.info("✓ Hybrid retriever ready")

    def _load_bm25(self) -> None:
        """Load BM25 index and metadata from disk."""
        bm25_path = self.persist_dir / "bm25_index.pkl"

        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {bm25_path}")

        with open(bm25_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
        self.texts = data["texts"]
        self.page_numbers = data["page_numbers"]
        self.original_texts = data["original_texts"]

        logger.info(f"✓ BM25 index loaded ({len(self.chunk_ids)} chunks)")

    def search(self, query: str, top_k: int = 100) -> list[dict]:
        """
        Perform hybrid search with RRF fusion.
        """
        logger.info(f"Hybrid search: '{query[:50]}...' (top_k={top_k})")

        # Vector search
        vector_results = self._vector_search(query, top_k)

        # BM25 search
        bm25_results = self._bm25_search(query, top_k)

        # RRF Fusion
        fused_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            k=60,  # Standard RRF constant
        )

        # Format and return top_k results
        formatted = self._format_results(fused_results[:top_k])

        logger.info(f"✓ Retrieved {len(formatted)} results")
        return formatted

    def _vector_search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        """
        Perform vector similarity search.
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )

        # Return as (chunk_id, rank) pairs
        chunk_ids = results["ids"][0]
        return [(chunk_id, rank) for rank, chunk_id in enumerate(chunk_ids)]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        """
        Perform BM25 lexical search.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[idx], rank) for rank, idx in enumerate(top_indices)]

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[str, int]],
        bm25_results: list[tuple[str, int]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """
        Fuse rankings using Reciprocal Rank Fusion.
        RRF formula: score(chunk) = sum(1 / (k + rank_i)) for all retrievers
        """
        rrf_scores: dict[str, float] = {}

        for chunk_id, rank in vector_results:
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        for chunk_id, rank in bm25_results:
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results

    def _format_results(self, fused_results: list[tuple[str, float]]) -> list[dict]:
        """
        Format fused results with full chunk metadata.
        """
        formatted = []

        for chunk_id, rrf_score in fused_results:
            # Get index of chunk
            idx = self.chunk_ids.index(chunk_id)

            formatted.append(
                {
                    "chunk_id": chunk_id,
                    "text": self.texts[idx],  # Contextualized text
                    "original_text": self.original_texts[idx],  # Non-contextualized
                    "page_number": self.page_numbers[idx],
                    "rrf_score": float(rrf_score),
                }
            )

        return formatted
