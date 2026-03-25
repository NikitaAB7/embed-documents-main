"""DeepSeek OCR adapter for the unified parser module.

This adapter uses DeepSeek's vision-language models for document OCR and parsing.
Supports multiple inference modes:

1. API mode: Use DeepSeek cloud API
2. Local mode: Run DeepSeek-OCR with transformers (recommended for accuracy)
3. VLLM mode: High-throughput inference (~2500 tokens/s on A100-40G)

DeepSeek-OCR Local Inference:
    Uses deepseek-ai/DeepSeek-OCR model with flash_attention_2
    - Supports different image size modes: Tiny, Small, Base, Large, Gundam
    - Crop mode for high-resolution documents
    - Markdown output with grounding

Size Presets:
    Tiny:  base_size=512,  image_size=512,  crop_mode=False
    Small: base_size=640,  image_size=640,  crop_mode=False
    Base:  base_size=1024, image_size=1024, crop_mode=False
    Large: base_size=1280, image_size=1280, crop_mode=False
    Gundam: base_size=1024, image_size=640, crop_mode=True (default)

DeepSeek-OCR excels at:
- High-accuracy OCR for printed and handwritten text
- Table structure recognition
- Multi-language document support
- Layout-aware text extraction
- Mathematical formula recognition
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from parser.base import BaseDocumentParser, DocumentInput
from parser.schemas import (
    BoundingBox,
    ChunkType,
    DocumentMetadata,
    ParsedChunk,
    ParsedDocument,
    ParserBackend,
)

logger = logging.getLogger(__name__)


# Size presets for DeepSeek-OCR
DEEPSEEK_OCR_PRESETS = {
    "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},  # Default
}


class DeepSeekAdapter(BaseDocumentParser):
    """Adapter for DeepSeek-OCR document parsing.

    Uses DeepSeek's vision-language model for OCR and document understanding.
    Supports API, local transformers, and VLLM inference modes.
    
    Local mode uses deepseek-ai/DeepSeek-OCR with flash_attention_2 for
    high-accuracy OCR with markdown output.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-ai/DeepSeek-OCR",
        use_local: bool = False,
        use_vllm: bool = False,
        local_model_path: Optional[str] = None,
        vllm_tensor_parallel_size: int = 1,
        vllm_max_model_len: int = 4096,
        extract_tables: bool = True,
        extract_layout: bool = True,
        language: str = "auto",
        # DeepSeek-OCR specific settings
        size_preset: str = "gundam",  # tiny, small, base, large, gundam
        base_size: Optional[int] = None,  # Override preset
        image_size: Optional[int] = None,  # Override preset
        crop_mode: Optional[bool] = None,  # Override preset
        ocr_prompt: str = "<image>\n<|grounding|>Convert the document to markdown. ",
        cuda_device: str = "0",
        save_results: bool = False,
        output_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize DeepSeek adapter.

        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            base_url: API base URL
            model: Model name (default: deepseek-ai/DeepSeek-OCR)
            use_local: Use local inference with transformers (recommended)
            use_vllm: Use VLLM for high-throughput inference
            local_model_path: Path to local model weights
            vllm_tensor_parallel_size: Number of GPUs for tensor parallelism
            vllm_max_model_len: Maximum sequence length for VLLM
            extract_tables: Extract tables with structure
            extract_layout: Preserve document layout
            language: Target language for OCR ('auto', 'en', 'zh', etc.)
            size_preset: Image size preset (tiny/small/base/large/gundam)
            base_size: Override base_size from preset
            image_size: Override image_size from preset
            crop_mode: Override crop_mode from preset
            ocr_prompt: Prompt for DeepSeek-OCR (default uses grounding)
            cuda_device: CUDA device to use (default: '0')
            save_results: Save OCR results to disk
            output_dir: Directory for saved results
            **kwargs: Additional BaseDocumentParser arguments
        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.model = model
        self.use_local = use_local
        self.use_vllm = use_vllm
        self.local_model_path = local_model_path or os.getenv(
            "DEEPSEEK_MODEL_PATH", "deepseek-ai/DeepSeek-OCR"
        )
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_max_model_len = vllm_max_model_len
        self.extract_tables = extract_tables
        self.extract_layout = extract_layout
        self.language = language
        
        # DeepSeek-OCR specific settings
        preset = DEEPSEEK_OCR_PRESETS.get(size_preset, DEEPSEEK_OCR_PRESETS["gundam"])
        self.base_size = base_size if base_size is not None else preset["base_size"]
        self.image_size = image_size if image_size is not None else preset["image_size"]
        self.crop_mode = crop_mode if crop_mode is not None else preset["crop_mode"]
        self.ocr_prompt = ocr_prompt
        self.cuda_device = cuda_device
        self.save_results = save_results
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="deepseek_ocr_")
        
        self._local_model = None
        self._tokenizer = None
        self._vllm_engine = None
        self._vllm_processor = None

    @property
    def backend(self) -> ParserBackend:
        return ParserBackend.DEEPSEEK

    async def _parse_impl(
        self,
        document: DocumentInput,
        source_name: str,
        **kwargs: Any,
    ) -> ParsedDocument:
        """Parse document using DeepSeek VL2."""
        # Convert document to images (one per page)
        images = await self._document_to_images(document)

        chunks = []
        page_count = len(images)

        if self.use_vllm:
            # VLLM batch processing for all pages
            all_results = await self._parse_pages_vllm(images)
            for page_idx, page_result in enumerate(all_results):
                for chunk in page_result:
                    chunk.page = page_idx + 1
                chunks.extend(page_result)
        else:
            # Sequential processing
            for page_idx, image_bytes in enumerate(images):
                page_num = page_idx + 1

                if self.use_local:
                    page_result = await self._parse_page_local(image_bytes, page_num)
                else:
                    page_result = await self._parse_page_api(image_bytes, page_num)

                chunks.extend(page_result)

        return ParsedDocument(
            source=source_name,
            chunks=chunks,
            metadata=DocumentMetadata(
                page_count=page_count,
                parser_backend=self.backend.value,
                extra={
                    "model": self.model,
                    "language": self.language,
                    "inference_mode": "vllm" if self.use_vllm else ("local" if self.use_local else "api"),
                },
            ),
        )

    async def _parse_pages_vllm(
        self,
        images: list[bytes],
    ) -> list[list[ParsedChunk]]:
        """Parse multiple pages using VLLM for high throughput.
        
        Based on DeepSeek-OCR VLLM implementation.
        Achieves ~2500 tokens/s on A100-40G.
        """
        try:
            from vllm import LLM, SamplingParams
            from PIL import Image
        except ImportError:
            raise ImportError(
                "VLLM inference requires: pip install vllm pillow"
            )

        # Lazy load VLLM engine
        if self._vllm_engine is None:
            logger.info(f"Loading DeepSeek-VL2 with VLLM: {self.local_model_path}")
            self._vllm_engine = LLM(
                model=self.local_model_path,
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                max_model_len=self.vllm_max_model_len,
                trust_remote_code=True,
                dtype="bfloat16",
            )

        # Convert images to PIL
        pil_images = []
        for img_bytes in images:
            pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

        # Build prompts for all pages
        prompt = self._build_extraction_prompt()
        
        # Prepare inputs for VLLM
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=4096,
            stop=["<|endoftext|>"],
        )

        # Process in batch using VLLM
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self._run_vllm_batch,
            pil_images,
            prompt,
            sampling_params,
        )

        # Parse outputs into chunks
        all_chunks = []
        for page_idx, result_text in enumerate(results):
            page_chunks = self._parse_model_output(result_text, page_idx + 1)
            all_chunks.append(page_chunks)

        return all_chunks

    def _run_vllm_batch(
        self,
        images: list,
        prompt: str,
        sampling_params,
    ) -> list[str]:
        """Run VLLM batch inference (blocking)."""
        from vllm import TextPrompt
        from vllm.multimodal import MultiModalDataDict

        # Build requests with images
        requests = []
        for img in images:
            requests.append({
                "prompt": prompt,
                "multi_modal_data": {"image": img},
            })

        # Generate outputs
        outputs = self._vllm_engine.generate(
            requests,
            sampling_params,
        )

        return [output.outputs[0].text for output in outputs]

    async def _parse_page_api(
        self,
        image_bytes: bytes,
        page_num: int,
    ) -> list[ParsedChunk]:
        """Parse a single page using DeepSeek API."""
        import httpx

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not set. "
                "Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )

        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Build the prompt for structured extraction
        prompt = self._build_extraction_prompt()

        # Call DeepSeek API
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}"
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
            result = response.json()

        content = result["choices"][0]["message"]["content"]
        return self._parse_model_output(content, page_num)

    async def _parse_page_local(
        self,
        image_bytes: bytes,
        page_num: int,
    ) -> list[ParsedChunk]:
        """Parse a single page using local DeepSeek-OCR model.
        
        Uses deepseek-ai/DeepSeek-OCR with auto dtype for
        high-accuracy OCR with markdown output.
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Local inference requires: pip install transformers torch pillow"
            )

        # Lazy load model
        if self._local_model is None:
            model_path = self.local_model_path
            logger.info(f"Loading DeepSeek-OCR model: {model_path}")
            
            # Set CUDA device
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            
            # Load with eager attention to avoid FlashAttention2 dependency
            self._local_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                attn_implementation="eager",
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self._local_model = self._local_model.eval().cuda()
                logger.info(f"DeepSeek-OCR loaded on CUDA device {self.cuda_device}")
            else:
                self._local_model = self._local_model.eval()
                logger.info("DeepSeek-OCR loaded on CPU (no CUDA available)")

        # Save image temporarily for model.infer()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None,
                self._run_deepseek_ocr_inference,
                tmp_path,
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

        return self._parse_model_output(content, page_num)

    def _run_deepseek_ocr_inference(self, image_path: str) -> str:
        """Run DeepSeek-OCR model.infer() (blocking).
        
        Uses the model.infer() method from DeepSeek-OCR with configurable
        base_size, image_size, and crop_mode parameters.
        """
        logger.debug(
            f"Running DeepSeek-OCR: base_size={self.base_size}, "
            f"image_size={self.image_size}, crop_mode={self.crop_mode}"
        )
        
        result = self._local_model.infer(
            self._tokenizer,
            prompt=self.ocr_prompt,
            image_file=image_path,
            output_path=self.output_dir,
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=self.crop_mode,
            save_results=self.save_results,
            test_compress=False,
        )
        
        return result if isinstance(result, str) else str(result)

    def _run_local_inference_legacy(self, image, prompt: str) -> str:
        """Legacy local model inference for DeepSeek-VL2 (blocking)."""
        inputs = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": [image, prompt]}],
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = inputs.to(self._local_model.device)

        outputs = self._local_model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _build_extraction_prompt(self) -> str:
        """Build the extraction prompt for the model."""
        base_prompt = """Extract ALL text content from this document page with high accuracy.

Output Format:
1. Preserve the original reading order (top to bottom, left to right)
2. For tables, use markdown table format with | separators
3. For headings, prefix with appropriate # symbols
4. For lists, use - or numbered format
5. Preserve paragraph breaks with blank lines

"""
        if self.extract_tables:
            base_prompt += """For tables:
- Detect all table structures
- Maintain column alignment
- Include headers if present
- Mark table start with [TABLE_START] and end with [TABLE_END]

"""
        if self.extract_layout:
            base_prompt += """For layout:
- Mark section headings with [HEADING]...[/HEADING]
- Mark figure captions with [CAPTION]...[/CAPTION]

"""
        base_prompt += "Now extract the text from the image:"
        return base_prompt

    def _parse_model_output(
        self,
        content: str,
        page_num: int,
    ) -> list[ParsedChunk]:
        """Parse model output into structured chunks."""
        chunks = []
        chunk_id = 0

        # Split into sections based on markers and structure
        lines = content.split("\n")
        current_text = []
        current_type = ChunkType.TEXT
        in_table = False

        for line in lines:
            # Check for table markers
            if "[TABLE_START]" in line:
                # Flush current text
                if current_text:
                    chunks.append(self._create_chunk(
                        "\n".join(current_text),
                        current_type,
                        page_num,
                        chunk_id,
                    ))
                    chunk_id += 1
                    current_text = []
                in_table = True
                current_type = ChunkType.TABLE
                continue
            elif "[TABLE_END]" in line:
                # Flush table
                if current_text:
                    chunks.append(self._create_chunk(
                        "\n".join(current_text),
                        ChunkType.TABLE,
                        page_num,
                        chunk_id,
                    ))
                    chunk_id += 1
                    current_text = []
                in_table = False
                current_type = ChunkType.TEXT
                continue

            # Check for heading markers
            heading_match = re.match(r"\[HEADING\](.*?)\[/HEADING\]", line)
            if heading_match:
                if current_text:
                    chunks.append(self._create_chunk(
                        "\n".join(current_text),
                        current_type,
                        page_num,
                        chunk_id,
                    ))
                    chunk_id += 1
                    current_text = []
                chunks.append(self._create_chunk(
                    heading_match.group(1).strip(),
                    ChunkType.HEADING,
                    page_num,
                    chunk_id,
                ))
                chunk_id += 1
                continue

            # Check for caption markers
            caption_match = re.match(r"\[CAPTION\](.*?)\[/CAPTION\]", line)
            if caption_match:
                if current_text:
                    chunks.append(self._create_chunk(
                        "\n".join(current_text),
                        current_type,
                        page_num,
                        chunk_id,
                    ))
                    chunk_id += 1
                    current_text = []
                chunks.append(self._create_chunk(
                    caption_match.group(1).strip(),
                    ChunkType.CAPTION,
                    page_num,
                    chunk_id,
                ))
                chunk_id += 1
                continue

            # Check for markdown headings
            if line.startswith("#"):
                if current_text:
                    chunks.append(self._create_chunk(
                        "\n".join(current_text),
                        current_type,
                        page_num,
                        chunk_id,
                    ))
                    chunk_id += 1
                    current_text = []
                chunks.append(self._create_chunk(
                    line.lstrip("#").strip(),
                    ChunkType.HEADING,
                    page_num,
                    chunk_id,
                ))
                chunk_id += 1
                continue

            # Detect table by pipe characters
            if "|" in line and not in_table:
                if self._looks_like_table_row(line):
                    if current_text:
                        chunks.append(self._create_chunk(
                            "\n".join(current_text),
                            current_type,
                            page_num,
                            chunk_id,
                        ))
                        chunk_id += 1
                        current_text = []
                    current_type = ChunkType.TABLE

            current_text.append(line)

        # Flush remaining text
        if current_text:
            text = "\n".join(current_text).strip()
            if text:
                chunks.append(self._create_chunk(
                    text,
                    current_type,
                    page_num,
                    chunk_id,
                ))

        return chunks

    def _looks_like_table_row(self, line: str) -> bool:
        """Check if a line looks like a markdown table row."""
        return line.count("|") >= 2 and not line.strip().startswith("|--")

    def _create_chunk(
        self,
        text: str,
        chunk_type: ChunkType,
        page: int,
        index: int,
    ) -> ParsedChunk:
        """Create a ParsedChunk with standard metadata."""
        return ParsedChunk(
            text=text,
            chunk_type=chunk_type,
            page=page,
            index=index,
            chunk_id=f"deepseek-p{page}-c{index}",
            metadata={
                "parser": "deepseek",
                "model": self.model,
            },
        )

    async def _document_to_images(
        self,
        document: DocumentInput,
    ) -> list[bytes]:
        """Convert document pages to images."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PDF to image conversion requires: pip install pymupdf")

        # Get document bytes
        if isinstance(document, (str, Path)):
            with open(document, "rb") as f:
                doc_bytes = f.read()
        elif isinstance(document, bytes):
            doc_bytes = document
        else:
            doc_bytes = document.read()

        # Open PDF and convert to images
        doc = fitz.open(stream=doc_bytes, filetype="pdf")
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render at 150 DPI for good OCR quality
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat)
            images.append(pix.tobytes("png"))

        doc.close()
        return images
