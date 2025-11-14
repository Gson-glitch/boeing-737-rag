import logging
from fastapi import APIRouter, HTTPException
from src.api.models import QueryRequest, QueryResponse
from src.config import settings
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize components (singleton pattern)
_retriever = None
_reranker = None
_generator = None


def get_retriever() -> HybridRetriever:
    """Lazy initialization of retriever."""
    global _retriever
    if _retriever is None:
        logger.info("Initializing retriever...")
        _retriever = HybridRetriever(
            persist_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )
    return _retriever


def get_reranker() -> Reranker:
    """Lazy initialization of reranker."""
    global _reranker
    if _reranker is None:
        logger.info("Initializing reranker...")
        _reranker = Reranker(model_name=settings.reranker_model)
    return _reranker


def get_generator() -> AnswerGenerator:
    """Lazy initialization of generator."""
    global _generator
    if _generator is None:
        logger.info("Initializing generator...")
        _generator = AnswerGenerator(api_key=settings.gemini_api_key)
    return _generator


@router.post("/query", response_model=QueryResponse)
async def query_manual(request: QueryRequest) -> QueryResponse:
    """
    Query the Boeing 737 Operations Manual.
    """
    try:
        question = request.question
        logger.info(f"Query received: '{question[:100]}...'")
        
        # Retrieve
        retriever = get_retriever()
        results = retriever.search(question, top_k=settings.hybrid_top_k)
        
        if not results:
            return QueryResponse(
                answer="No relevant information found in the manual.",
                pages=[]
            )
        
        # Rerank
        reranker = get_reranker()
        reranked = reranker.rerank(
            question,
            results,
            top_k=settings.rerank_top_k
        )
        
        # Generate answer
        generator = get_generator()
        answer, pages = generator.generate(
            question,
            reranked,
            max_chunks=5
        )
        
        logger.info(f"Query processed successfully. Pages: {pages}")
        
        return QueryResponse(answer=answer, pages=pages)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "boeing-737-rag"}