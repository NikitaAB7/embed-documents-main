"""Document fetcher for retrieving PDFs from the API."""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional

from docling.datamodel.base_models import DocumentStream

from rag.schemas.rag_schemas import DocumentMetadata
from utils.api_utils import DefineEdgeFundamentalsAPI

logger = logging.getLogger(__name__)


class DocumentFetcher:
    """Fetch and cache corporate documents from the API."""

    def __init__(
        self,
        max_pdf_retries: int = 3,
        initial_backoff_seconds: float = 1.0,
    ):
        """Initialize document fetcher with API client."""
        self.api = DefineEdgeFundamentalsAPI()
        self.max_pdf_retries = max(1, max_pdf_retries)
        self.initial_backoff_seconds = max(0.1, initial_backoff_seconds)

    async def get_available_documents(
        self,
        fincode: Optional[int] = None,
        category: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[DocumentMetadata]:
        """Get metadata for available documents matching filters.

        Args:
            fincode: Filter by company financial code
            category: Filter by document category
            start_date: Filter by start date (YYYY-MM-DD)
            end_date: Filter by end date (YYYY-MM-DD)

        Returns:
            List of DocumentMetadata objects
        """
        # Get metadata from API (uses cached response)
        if not start_date or not end_date:
            # Default to last 2 years if not specified

            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        logger.info(
            f"Fetching document metadata: fincode={fincode}, category={category}, "
            f"date_range={start_date} to {end_date}"
        )

        meta_data = await self.api.get_meta_data_related_to_file(
            start_date=start_date, end_date=end_date
        )

        if not isinstance(meta_data, list):
            logger.warning("No documents found or invalid response from API")
            return []

        # Filter by fincode and category
        filtered_docs = []
        for item in meta_data:
            # Apply filters
            if fincode and item.get("fincode") != fincode:
                continue
            if category and item.get("subcatname") != category:
                continue

            # Create DocumentMetadata
            doc = DocumentMetadata(
                filename=item.get("attachmentname", ""),
                fincode=item.get("fincode", 0),
                category=item.get("subcatname", ""),
                document_date=item.get("newsDt", "")[:10],  # Extract YYYY-MM-DD
                is_embedded=False,  # Will be updated by vector store check
            )
            filtered_docs.append(doc)

        logger.info(f"Found {len(filtered_docs)} documents matching filters")
        return filtered_docs

    async def get_pdf_doc_stream(
        self,
        filename: str,
        category: str,
    ) -> DocumentStream:
        """Get PDF content as a byte stream.

        Args:
            filename: Document filename
            category: Document category
        Returns:
            PDF content as bytes
        """
        attempt = 1
        delay = self.initial_backoff_seconds

        while True:
            try:
                response = await self.api.get_file_by_category_and_attachment_name(
                    sub_cat_name=category, attachment_ame=filename
                )

                pdf_content = response.content
                buf = BytesIO(pdf_content)

                return DocumentStream(name=filename, stream=buf)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                if attempt >= self.max_pdf_retries:
                    logger.error(
                        "Failed to fetch document %s after %s attempts: %s",
                        filename,
                        attempt,
                        e,
                    )
                    raise

                logger.warning(
                    "Attempt %s/%s failed for %s (%s). Retrying in %.1fs",
                    attempt,
                    self.max_pdf_retries,
                    filename,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
                attempt += 1
                delay *= 2

    async def save_document_to_disk(
        self, filename: str, category: str, output_dir: str = ".cache/rag_pdfs_disk"
    ) -> str:
        """Fetch and save document to disk for processing.

        Args:
            filename: Document filename
            category: Document category
            output_dir: Directory to save PDFs

        Returns:
            Path to saved PDF file
        """
        # Create output directory (async)
        await asyncio.to_thread(os.makedirs, output_dir, exist_ok=True)

        # Fetch document
        response = await self.api.get_file_by_category_and_attachment_name(
            sub_cat_name=category, attachment_ame=filename
        )

        pdf_content = response.content

        # Save to disk (async)
        output_path = os.path.join(output_dir, filename)

        def write_file():
            with open(output_path, "wb") as f:
                f.write(pdf_content)

        await asyncio.to_thread(write_file)

        logger.info(f"Saved PDF to disk: {output_path}")
        return output_path
