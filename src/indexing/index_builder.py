import pickle
import logging
from pathlib import Path
from typing import List
import chromadb
import numpy as np
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.indexing.embedder import Embedder

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Build dual indices: ChromaDB (vector) + BM25 (lexical).
    """
    
    def __init__(self, persist_dir: str, embedding_model: str, collection_name: str = "boeing_737"):
        """
        Initialize index builder.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedder = Embedder(embedding_model, use_fp16=False)
        
        # Initialize ChromaDB with persistent storage
        logger.info(f"Initializing ChromaDB at {self.persist_dir}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Match normalized embeddings
        )
        
        logger.info(f"Collection '{collection_name}' ready")
    
    def build_indices(self, chunks: List[Chunk]) -> None:
        """
        Build both vector (ChromaDB) and BM25 indices from chunks.
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        logger.info(f"Building indices for {len(chunks)} chunks")
        
        # Prepare data
        chunk_ids = [c.chunk_id for c in chunks]
        # Use contextualized text for richer semantic matching
        texts = [c.contextualized_text for c in chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.embed_documents(texts)
        
        # Build vector index (ChromaDB)
        logger.info("Adding to ChromaDB...")
        self._add_to_chromadb(chunk_ids, texts, embeddings, chunks)
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        self._build_bm25_index(chunk_ids, texts, chunks)
        
        logger.info(f"✓ Indices built successfully")
    
    def _add_to_chromadb(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        chunks: List[Chunk]
    ) -> None:
        """Add embeddings and metadata to ChromaDB in batches."""
        
        # ChromaDB performs better with batch insertions
        batch_size = 100
        
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            
            self.collection.add(
                ids=chunk_ids[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                documents=texts[i:batch_end],
                metadatas=[{
                    "page_number": c.page_number,
                    "chunk_id": c.chunk_id,
                    "original_text": c.text  # Store non-contextualized for display
                } for c in chunks[i:batch_end]]
            )
            
            if (i + batch_size) % 500 == 0:
                logger.info(f"  Added {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
    
    def _build_bm25_index(
        self,
        chunk_ids: List[str],
        texts: List[str],
        chunks: List[Chunk]
    ) -> None:
        """Build and persist BM25 index."""
        
        # Simple whitespace tokenization (BM25 standard)
        tokenized_texts = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_texts)
        
        # Save BM25 index and metadata
        bm25_path = self.persist_dir / "bm25_index.pkl"
        
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'bm25': bm25,
                'chunk_ids': chunk_ids,
                'texts': texts,
                'page_numbers': [c.page_number for c in chunks],
                'original_texts': [c.text for c in chunks]
            }, f)
        
        logger.info(f"✓ BM25 index saved to {bm25_path}")
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the indexed collection."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name,
            "persist_dir": str(self.persist_dir)
        }