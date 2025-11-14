import logging

logger = logging.getLogger(__name__)


class PageAggregator:
    """
    Aggregate page numbers from retrieved chunks.
    """

    @staticmethod
    def extract_pages(
        results: list[dict], max_pages: int = 5, score_key: str = "rerank_score"
    ) -> list[int]:
        """
        Extract unique page numbers from results.
        """
        if not results:
            return []

        page_scores: dict[int, float] = {}
        for result in results:
            page_num = result["page_number"]
            score = result.get(score_key, 0.0)

            # Keep highest score for each page
            if page_num not in page_scores or score > page_scores[page_num]:
                page_scores[page_num] = score

        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        pages = [page for page, score in sorted_pages[:max_pages]]

        logger.info(f"Extracted {len(pages)} unique pages from {len(results)} results")
        return pages

    @staticmethod
    def extract_pages_with_confidence(
        results: list[dict],
        confidence_threshold: float = 0.5,
        max_pages: int = 10,
        score_key: str = "rerank_score",
    ) -> list[int]:
        """
        Extract pages that meet a confidence threshold.
        """
        if not results:
            return []

        # Filter by confidence
        confident_results = [
            r for r in results if r.get(score_key, 0.0) >= confidence_threshold
        ]

        if not confident_results:
            logger.warning(
                f"No results above threshold {confidence_threshold}, "
                f"falling back to top {max_pages}"
            )
            return PageAggregator.extract_pages(results, max_pages, score_key)

        # Extract pages
        return PageAggregator.extract_pages(confident_results, max_pages, score_key)
