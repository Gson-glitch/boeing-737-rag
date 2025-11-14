import logging
import re
import textwrap

import google.generativeai as genai

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Generate answers from retrieved context using LLM.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize answer generator.
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Answer generator ready (model={model_name})")

    def generate(
        self, query: str, retrieved_chunks: list[dict], max_chunks: int = 5
    ) -> tuple[str, list[int]]:
        """
        Generate answer from retrieved context.
        """
        if not retrieved_chunks:
            return self._no_results_response(query)

        # Select top chunks
        top_chunks = retrieved_chunks[:max_chunks]

        # Build prompt
        prompt = self._build_prompt(query, top_chunks)

        # Generate answer
        logger.info(f"Generating answer for: '{query[:50]}...'")
        response = self.model.generate_content(prompt)
        answer = response.text.strip()

        # Extract cited pages
        cited_pages = self._extract_cited_pages(answer, top_chunks)

        logger.info(f"Generated answer with {len(cited_pages)} page citations")
        return answer, cited_pages

    def _build_prompt(self, query: str, chunks: list[dict]) -> str:
        """
        Build prompt with context and instructions.
        """
        # Format context documents
        context_docs = []
        for i, chunk in enumerate(chunks, 1):
            page = chunk["page_number"]
            text = chunk["original_text"]
            context_docs.append(f"[Document {i} - Page {page}]\n{text}")

        context_section = "\n\n".join(context_docs)

        # Build full prompt with proper alignment
        prompt = textwrap.dedent(f"""\
            You are an expert assistant for the Boeing 737 Operations Manual.

            CONTEXT DOCUMENTS:
            {context_section}

            INSTRUCTIONS:
            1. Answer the user's question using ONLY the information in the context documents above
            2. Be precise and technical - this is for flight operations
            3. When you reference information, cite the document number in brackets like [Document 1]
            4. If you use information from multiple documents, cite all of them like [Document 1, Document 2]
            5. If the answer cannot be found in the provided context, clearly state: "This information is not available in the provided manual sections."
            6. Do not add information from your general knowledge
            7. Keep answers concise but complete

            USER QUESTION:
            {query}

            ANSWER:
        """)

        return prompt

    def _extract_cited_pages(self, answer: str, chunks: list[dict]) -> list[int]:
        """
        Extract page numbers from citations in the answer.
        """
        citation_pattern = r"\[Document\s+\d+(?:(?:,\s*(?:Document\s+)?\d+)*)\]"
        matches = re.findall(citation_pattern, answer)

        cited_doc_indices: set[int] = set()

        for match in matches:
            numbers = re.findall(r"\d+", match)
            cited_doc_indices.update(int(n) for n in numbers)

        # Map document indices to page numbers
        cited_pages = []
        for doc_idx in sorted(cited_doc_indices):
            if 1 <= doc_idx <= len(chunks):
                page = chunks[doc_idx - 1]["page_number"]
                if page not in cited_pages:
                    cited_pages.append(page)

        # Fallback: if no citations found, use top 3 chunks
        if not cited_pages:
            logger.warning("No citations found in answer, using top 3 context pages")
            cited_pages = [chunk["page_number"] for chunk in chunks[:3]]
            cited_pages = sorted(list(set(cited_pages)))

        return cited_pages

    def _no_results_response(self, query: str) -> tuple[str, list[int]]:
        """
        Generate response when no relevant context is found.
        """
        answer = (
            "I could not find relevant information in the Boeing 737 Operations Manual "
            "to answer your question. Please verify the question or consult the full manual."
        )
        return answer, []
