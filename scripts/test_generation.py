import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.answer_generator import AnswerGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_end_to_end():
    """Test complete RAG pipeline."""
    
    test_questions = [
        "What is the first action after positive rate of climb?",
        "What does the amber STAIRS OPER light indicate?",
        "Where is the ISOLATION VALVE switch set during After Start Procedure?",
    ]
    
    # Initialize components
    logger.info("Initializing pipeline...")
    retriever = HybridRetriever(
        persist_dir=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model
    )
    reranker = Reranker(model_name=settings.reranker_model)
    generator = AnswerGenerator(api_key=settings.gemini_api_key)
    
    # Test each question
    for question in test_questions:
        logger.info("\n" + "="*80)
        logger.info(f"QUESTION: {question}")
        logger.info("="*80)
        
        # Retrieve
        results = retriever.search(question, top_k=settings.hybrid_top_k)
        logger.info(f"Retrieved {len(results)} results")
        
        # Rerank
        reranked = reranker.rerank(question, results, top_k=settings.rerank_top_k)
        logger.info(f"Reranked to {len(reranked)} results")
        
        # Generate
        answer, pages = generator.generate(question, reranked, max_chunks=5)
        
        # Display
        logger.info(f"\nANSWER:\n{answer}")
        logger.info(f"\nPAGES: {pages}")
        logger.info("="*80 + "\n")


if __name__ == "__main__":
    test_end_to_end()