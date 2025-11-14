import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)


@dataclass
class ParsedElement:
    """Document element with page tracking."""

    text: str
    page_number: int
    element_type: str


class PDFParser:
    """Parse PDF maintaining page numbers and structure."""

    NOISE_PATTERNS = [
        r"Copyright © The Boeing Company",
        r"^DO NOT USE FOR FLIGHT$",
        r"^Boeing 737 Operations Manual$",
        r"See title page for details",
        r"D6-27370-TBC",
        r"FCOM Template",
        r"^\[Option.*\]$",
        r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}$",
        r"^Normal Procedures Chapter NP",
        r"^Table of Contents",
        r"^Normal Procedures\s*-\s*$",
        r"^Introduction\s*$",
    ]

    # Images to skip (header logos, decorative elements)
    SKIP_IMAGE_TEXTS = [
        "DO NOT USE FOR FLIGHT",
        "Boeing 737 Operations Manual",
    ]

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def parse(self) -> list[ParsedElement]:
        """Extract elements with page numbers from PDF."""
        logger.info(f"Parsing {self.pdf_path}")

        elements = partition_pdf(
            filename=str(self.pdf_path),
            strategy="hi_res",
            infer_table_structure=True,
        )

        parsed = []

        for elem in elements:
            text = str(elem).strip()

            if not text or len(text) < 10 or self._is_noise(text):
                continue

            page_num = (
                elem.metadata.page_number
                if hasattr(elem.metadata, "page_number")
                else 1
            )
            elem_type = self._get_type(elem)

            # Skip header/logo images
            if elem_type == "image" and self._is_header_image(text):
                continue

            # Keep only meaningful diagrams (longer descriptions)
            if elem_type == "image":
                if len(text) < 20:  # Skip small/simple images
                    continue

            parsed.append(
                ParsedElement(
                    text=text, 
                    page_number=page_num if page_num is not None else 0,
                    element_type=elem_type)
            )

        logger.info(
            f"Parsed {len(parsed)} elements from {len(set(e.page_number for e in parsed))} pages"
        )
        return parsed

    def _get_type(self, element) -> str:
        """Determine element type."""
        name = type(element).__name__
        if "Image" in name:
            return "image"
        elif "Table" in name:
            return "table"
        elif "List" in name:
            return "list"
        elif "Title" in name:
            return "title"
        return "text"

    def _is_noise(self, text: str) -> bool:
        """Check if text is noise."""
        for pattern in self.NOISE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        if ". . . . ." in text:
            return True

        return False

    def _is_header_image(self, text: str) -> bool:
        """Check if image is a header/logo."""
        for skip_text in self.SKIP_IMAGE_TEXTS:
            if skip_text.lower() in text.lower():
                return True
        return False

    def save(self, elements: list[ParsedElement], path: str):
        """Save elements to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(e) for e in elements], f, indent=2)
        logger.info(f"Saved to {path}")


def group_by_page(elements: list[ParsedElement]) -> dict[int, str]:
    """Group elements by page and concatenate text."""
    pages: dict[int, list[str]] = {}
    for elem in elements:
        if elem.page_number not in pages:
            pages[elem.page_number] = []
        pages[elem.page_number].append(elem.text)

    return {page: "\n\n".join(texts) for page, texts in pages.items()}
