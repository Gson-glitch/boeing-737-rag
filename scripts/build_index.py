import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ingestion.chunker import Chunker
from src.indexing.index_builder import IndexBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Build indices from processed chunks."""
    
    # Load processed chunks
    chunks_path = Path(settings.processed_chunks_path)
    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        logger.error("Run 'python scripts/process_manual.py' first")
        sys.exit(1)
    
    logger.info(f"Loading chunks from {chunks_path}")
    chunks = Chunker.load(str(chunks_path))
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Build indices
    builder = IndexBuilder(
        persist_dir=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model
    )
    
    builder.build_indices(chunks)
    
    # Print stats
    stats = builder.get_collection_stats()
    logger.info("\n" + "="*50)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("="*50)
    logger.info(f"Total chunks indexed: {stats['total_chunks']}")
    logger.info(f"Collection: {stats['collection_name']}")
    logger.info(f"Location: {stats['persist_dir']}")
    logger.info("="*50)


if __name__ == "__main__":
    main()