import logging
import time

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


def log_retry(retry_state):
    """Log tenacity retry attempts."""
    logger.warning(
        f"Rate limit or service error hit. "
        f"Retrying {retry_state.fn.__name__} in {retry_state.next_action.sleep:.1f}s... "
        f"(Attempt {retry_state.attempt_number})"
    )


class Contextualizer:
    """Add contextual information to chunks using Gemini."""

    def __init__(self, api_key: str, requests_per_minute: int = 50):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro")
        # Proactive throttle: add buffer to 60 RPM limit
        self.delay_seconds = 60.0 / requests_per_minute

    def add_context(self, chunks: list[Chunk], batch_size: int = 10) -> list[Chunk]:
        """Add context to chunks for better retrieval."""
        logger.info(f"Adding context to {len(chunks)} chunks")

        for chunk in tqdm(chunks, desc="Adding context", unit="chunk"):
            time.sleep(self.delay_seconds)

            context = self._generate_context(chunk)
            chunk.contextualized_text = f"{context}\n\n{chunk.text}"

        logger.info("Context generation complete")
        return chunks

    @retry(
        retry=retry_if_exception_type(
            (
                google_exceptions.ResourceExhausted,  # 429
                google_exceptions.ServiceUnavailable,  # 503
            )
        ),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(5),
        before_sleep=log_retry,
    )
    def _generate_context(self, chunk: Chunk) -> str:
        """Generate 2-3 sentence context for a chunk."""
        prompt = (
            f"This is a chunk from Boeing 737 Operations Manual, "
            f"Page {chunk.page_number}.\n\n"
            f"Page context (first 500 chars):\n"
            f"{chunk.parent_page_text[:500]}...\n\n"
            f"Chunk:\n"
            f"{chunk.text}\n\n"
            "Provide 2-3 sentences of context explaining:\n"
            "1. What procedure/section this relates to\n"
            "2. Key technical terms or components\n"
            "Keep it concise and technical. Context only, no preamble."
        )

        try:
            response = self.model.generate_content(prompt)
            return str(response.text.strip())
        except Exception as e:
            logger.error(
                f"Context generation FAILED permanently for {chunk.chunk_id}: {e}"
            )
            return f"Boeing 737 manual content from page {chunk.page_number}"
