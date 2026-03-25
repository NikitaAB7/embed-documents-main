"""GOT-OCR / GLM-OCR adapter for the unified parser module.

This adapter supports both GOT-OCR2.0 and GLM-OCR for document parsing.

GOT-OCR2.0 (General OCR Theory) - StepFun/THU:
- State-of-the-art OCR accuracy
- Multi-language support (excellent for Chinese, English, etc.)
- Mathematical formula recognition (LaTeX output)
- Table structure recognition
- Sheet music OCR
- Geometric shape recognition

GLM-OCR (zai-org):
- High-quality text recognition
- Supports vLLM/SGLang/Ollama/Transformers deployment
- OpenAI-compatible API for server modes
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import tempfile
from enum import Enum
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


class GLOCRMode(str, Enum):
    """Deployment mode for GLOCR."""
    API = "api"  # Generic API endpoint
    LOCAL = "local"  # Local transformers inference
    GRADIO = "gradio"  # Gradio demo
    VLLM = "vllm"  # vLLM server (OpenAI-compatible)
    SGLANG = "sglang"  # SGLang server (OpenAI-compatible)
    OLLAMA = "ollama"  # Ollama local server


class GLOCRModel(str, Enum):
    """Supported OCR models."""
    GOT_OCR2 = "stepfun-ai/GOT-OCR2_0"  # Original GOT-OCR2.0
    GLM_OCR = "zai-org/GLM-OCR"  # GLM-OCR


class GLOCRAdapter(BaseDocumentParser):
    """Adapter for GOT-OCR2.0 and GLM-OCR document parsing.

    Supports multiple deployment modes:
    1. API mode: Use hosted API endpoint
    2. Local mode: Run model locally with transformers
    3. Gradio mode: Connect to a running Gradio demo
    4. vLLM mode: Connect to vLLM server (OpenAI-compatible)
    5. SGLang mode: Connect to SGLang server (OpenAI-compatible)
    6. Ollama mode: Connect to Ollama server
    """

    def __init__(
        self,
        # Mode selection
        mode: GLOCRMode | str = GLOCRMode.API,
        model: GLOCRModel | str = GLOCRModel.GLM_OCR,
        # API/Server settings
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        server_url: str = "http://localhost:8080",
        # Ollama settings
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "glm-ocr",
        # Local settings
        use_local: bool = False,  # Deprecated, use mode=LOCAL
        model_path: Optional[str] = None,  # Override model path
        # Gradio settings
        use_gradio: bool = False,  # Deprecated, use mode=GRADIO
        gradio_url: str = "http://localhost:7860",
        # OCR settings (GOT-OCR specific)
        ocr_type: str = "ocr",  # 'ocr', 'format', 'fine-grained'
        ocr_box: Optional[str] = None,  # For fine-grained: "[x1,y1,x2,y2]"
        ocr_color: Optional[str] = None,  # For fine-grained: "red", "green", etc.
        render_latex: bool = True,
        # GLM-OCR prompt
        prompt: str = "Text Recognition:",
        max_tokens: int = 8192,
        **kwargs: Any,
    ):
        """Initialize GLOCR adapter.

        Args:
            mode: Deployment mode (api, local, gradio, vllm, sglang, ollama)
            model: Model to use (GOT_OCR2 or GLM_OCR)
            api_key: API key for hosted service
            api_url: API endpoint URL (for api mode)
            server_url: Server URL for vLLM/SGLang modes
            ollama_url: Ollama server URL
            ollama_model: Ollama model name
            use_local: [Deprecated] Use mode=LOCAL instead
            model_path: Override HuggingFace model path
            use_gradio: [Deprecated] Use mode=GRADIO instead
            gradio_url: URL of Gradio demo
            ocr_type: OCR mode for GOT-OCR ('ocr', 'format', 'fine-grained')
            ocr_box: Bounding box for fine-grained OCR "[x1,y1,x2,y2]"
            ocr_color: Color mask for fine-grained OCR
            render_latex: Render LaTeX formulas in output
            prompt: Prompt for GLM-OCR (default: "Text Recognition:")
            max_tokens: Max tokens for generation
            **kwargs: Additional BaseDocumentParser arguments
        """
        super().__init__(**kwargs)

        # Handle deprecated parameters
        if use_local:
            mode = GLOCRMode.LOCAL
        elif use_gradio:
            mode = GLOCRMode.GRADIO

        # Normalize mode
        self.mode = GLOCRMode(mode) if isinstance(mode, str) else mode

        # Normalize model
        self.model = GLOCRModel(model) if isinstance(model, str) else model

        # Model path (allow override)
        self.model_path = model_path or self.model.value

        # API/Server settings
        self.api_key = api_key or os.getenv("GOT_OCR_API_KEY")
        self.api_url = api_url or os.getenv("GOT_OCR_API_URL")
        self.server_url = server_url

        # Ollama settings
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        # Gradio settings
        self.gradio_url = gradio_url

        # OCR settings
        self.ocr_type = ocr_type
        self.ocr_box = ocr_box
        self.ocr_color = ocr_color
        self.render_latex = render_latex

        # GLM-OCR settings
        self.prompt = prompt
        self.max_tokens = max_tokens

        # Model instances (lazy loaded)
        self._model = None
        self._processor = None
        self._tokenizer = None  # For GOT-OCR compatibility

    @property
    def backend(self) -> ParserBackend:
        return ParserBackend.GLOCR

    async def _parse_impl(
        self,
        document: DocumentInput,
        source_name: str,
        **kwargs: Any,
    ) -> ParsedDocument:
        """Parse document using GLOCR (GOT-OCR or GLM-OCR)."""
        # Convert document to images
        images = await self._document_to_images(document)

        chunks = []
        page_count = len(images)

        for page_idx, image_bytes in enumerate(images):
            page_num = page_idx + 1

            # Route to appropriate parser based on mode
            if self.mode == GLOCRMode.LOCAL:
                page_result = await self._parse_page_local(image_bytes, page_num)
            elif self.mode == GLOCRMode.GRADIO:
                page_result = await self._parse_page_gradio(image_bytes, page_num)
            elif self.mode == GLOCRMode.VLLM:
                page_result = await self._parse_page_vllm(image_bytes, page_num)
            elif self.mode == GLOCRMode.SGLANG:
                page_result = await self._parse_page_sglang(image_bytes, page_num)
            elif self.mode == GLOCRMode.OLLAMA:
                page_result = await self._parse_page_ollama(image_bytes, page_num)
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
                    "model": self.model_path,
                    "mode": self.mode.value,
                    "ocr_type": self.ocr_type,
                },
            ),
        )

    # -------------------------------------------------------------------------
    # vLLM Server Mode (OpenAI-compatible API)
    # -------------------------------------------------------------------------

    async def _parse_page_vllm(
        self,
        image_bytes: bytes,
        page_num: int,
    ) -> list[ParsedChunk]:
        """Parse a single page using vLLM server (OpenAI-compatible).

        vLLM server started with:
            vllm serve zai-org/GLM-OCR --allowed-local-media-path / --port 8080
        """
        import httpx

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Build OpenAI-compatible request
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": self.prompt,
                    },
                ],
            }
        ]

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.server_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_path,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
            result = response.json()

        content = result["choices"][0]["message"]["content"]
        return self._parse_output(content, page_num)

    # -------------------------------------------------------------------------
    # SGLang Server Mode (OpenAI-compatible API)
    # -------------------------------------------------------------------------

    async def _parse_page_sglang(
        self,
        image_bytes: bytes,
        page_num: int,
    ) -> list[ParsedChunk]:
        """Parse a single page using SGLang server (OpenAI-compatible).

        SGLang server started with:
            python -m sglang.launch_server --model zai-org/GLM-OCR --port 8080
        """
        import httpx

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # SGLang uses same OpenAI-compatible API as vLLM
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": self.prompt,
                    },
                ],
            }
        ]

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.server_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_path,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
            result = response.json()

        content = result["choices"][0]["message"]["content"]
        return self._parse_output(content, page_num)

    # -------------------------------------------------------------------------
    # Ollama Mode
    # -------------------------------------------------------------------------

    async def _parse_page_ollama(
        self,
        image_bytes: bytes,
        page_num: int,
    ) -> list[ParsedChunk]:
        """Parse a single page using Ollama.

        Run Ollama with:
            ollama run glm-ocr
        """
        import httpx

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Ollama API format
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.ollama_model,
                    "prompt": self.prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "num_predict": self.max_tokens,
                        "temperature": 0.0,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()

        content = result.get("response", "")
        return self._parse_output(content, page_num)

    # -------------------------------------------------------------------------
    # Generic API Mode
    # -------------------------------------------------------------------------

    async def _parse_page_api(
        self,
        image_bytes: bytes,
        page_num: int,
    ) -> list[ParsedChunk]:
        """Parse a single page using generic API."""
        import httpx

        if not self.api_url:
            raise ValueError(
                "GOT-OCR API URL not set. "
                "Set GOT_OCR_API_URL environment variable or pass api_url parameter."
            )

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "image": image_b64,
            "ocr_type": self.ocr_type,
            "render": self.render_latex,
        }
        if self.ocr_box:
            payload["ocr_box"] = self.ocr_box
        if self.ocr_color:
            payload["ocr_color"] = self.ocr_color

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

        content = result.get("text", result.get("result", ""))
        return self._parse_output(content, page_num)

    async def _parse_page_gradio(
        self,
        image_bytes: bytes,
        page_num: int,
    ) -> list[ParsedChunk]:
        """Parse a single page using Gradio client."""
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            raise ImportError(
                "Gradio client required: pip install gradio_client"
            )

        # Save image to temp file for Gradio
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        try:
            client = Client(self.gradio_url)
            result = client.predict(
                temp_path,  # image path
                self.ocr_type,  # ocr_type
                self.ocr_box or "",  # ocr_box
                self.ocr_color or "",  # ocr_color
                api_name="/predict",
            )
            content = result if isinstance(result, str) else result[0]
        finally:
            os.unlink(temp_path)

        return self._parse_output(content, page_num)

    # -------------------------------------------------------------------------
    # Local Transformers Mode
    # -------------------------------------------------------------------------

    async def _parse_page_local(
        self,
        image_bytes: bytes,
        page_num: int,
    ) -> list[ParsedChunk]:
        """Parse a single page using local model (transformers).

        Supports both GOT-OCR2.0 and GLM-OCR models.
        """
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Local inference requires: pip install torch pillow"
            )

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Use appropriate inference method based on model
        if self.model == GLOCRModel.GLM_OCR:
            content = await self._run_glm_ocr_inference(image)
        else:
            content = await self._run_got_ocr_inference(image)

        return self._parse_output(content, page_num)

    async def _run_glm_ocr_inference(self, image) -> str:
        """Run GLM-OCR inference with transformers.

        Based on official documentation:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        """
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError:
            raise ImportError(
                "GLM-OCR requires: pip install git+https://github.com/huggingface/transformers.git"
            )

        # Lazy load model
        if self._model is None:
            logger.info(f"Loading GLM-OCR model: {self.model_path}")
            self._processor = AutoProcessor.from_pretrained(self.model_path)
            self._model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=self.model_path,
                torch_dtype="auto",
                device_map="auto",
            )
            self._model = self._model.eval()

        # Save image to temp file for the processor
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            temp_path = f.name

        try:
            # Build messages in GLM-OCR format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": temp_path,  # Can use local file path
                        },
                        {
                            "type": "text",
                            "text": self.prompt,
                        },
                    ],
                }
            ]

            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None,
                self._run_glm_ocr_generate,
                messages,
            )
        finally:
            os.unlink(temp_path)

        return content

    def _run_glm_ocr_generate(self, messages: list[dict]) -> str:
        """Run GLM-OCR generation (blocking)."""
        import torch

        # Apply chat template
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        # Remove token_type_ids if present (as per docs)
        inputs.pop("token_type_ids", None)

        # Generate
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
            )

        # Decode output (skip input tokens)
        output_text = self._processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )

        # Clean up special tokens
        output_text = output_text.replace("<|endoftext|>", "").strip()

        return output_text

    async def _run_got_ocr_inference(self, image) -> str:
        """Run GOT-OCR2.0 inference with transformers."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Local inference requires: pip install torch transformers"
            )

        # Lazy load model
        if self._model is None:
            logger.info(f"Loading GOT-OCR model: {self.model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            self._model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            self._model = self._model.eval()

        # Save to temp file (GOT-OCR expects file path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            temp_path = f.name

        try:
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None,
                self._run_got_ocr_generate,
                temp_path,
            )
        finally:
            os.unlink(temp_path)

        return content

    def _run_got_ocr_generate(self, image_path: str) -> str:
        """Run GOT-OCR generation (blocking)."""
        # GOT-OCR uses a specific chat method
        if self.ocr_type == "format":
            result = self._model.chat(
                self._tokenizer,
                image_path,
                ocr_type="format",
            )
        elif self.ocr_type == "fine-grained":
            result = self._model.chat(
                self._tokenizer,
                image_path,
                ocr_type="ocr",
                ocr_box=self.ocr_box,
                ocr_color=self.ocr_color,
            )
        else:
            result = self._model.chat(
                self._tokenizer,
                image_path,
                ocr_type="ocr",
            )

        return result

    # -------------------------------------------------------------------------
    # Output parsing
    # -------------------------------------------------------------------------

    def _parse_output(
        self,
        content: str,
        page_num: int,
    ) -> list[ParsedChunk]:
        """Parse GOT-OCR output into structured chunks."""
        chunks = []
        chunk_id = 0

        # GOT-OCR in format mode returns markdown-like output
        lines = content.split("\n")
        current_text = []
        current_type = ChunkType.TEXT

        for line in lines:
            # Detect LaTeX formulas
            if "\\begin{" in line or "$$" in line:
                if current_text:
                    chunks.append(self._create_chunk(
                        "\n".join(current_text),
                        current_type,
                        page_num,
                        chunk_id,
                    ))
                    chunk_id += 1
                    current_text = []
                current_type = ChunkType.FORMULA
                current_text.append(line)
                continue

            # Detect end of formula
            if current_type == ChunkType.FORMULA:
                current_text.append(line)
                if "\\end{" in line or (line.strip() == "$$" and len(current_text) > 1):
                    chunks.append(self._create_chunk(
                        "\n".join(current_text),
                        ChunkType.FORMULA,
                        page_num,
                        chunk_id,
                    ))
                    chunk_id += 1
                    current_text = []
                    current_type = ChunkType.TEXT
                continue

            # Detect headings (markdown style)
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
                current_type = ChunkType.TEXT
                continue

            # Detect tables (pipe characters)
            if "|" in line and line.count("|") >= 2:
                if current_type != ChunkType.TABLE and current_text:
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
                continue

            # End of table
            if current_type == ChunkType.TABLE and "|" not in line:
                chunks.append(self._create_chunk(
                    "\n".join(current_text),
                    ChunkType.TABLE,
                    page_num,
                    chunk_id,
                ))
                chunk_id += 1
                current_text = []
                current_type = ChunkType.TEXT

            # Regular text
            if line.strip():
                current_text.append(line)

        # Flush remaining
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
            chunk_id=f"glocr-p{page}-c{index}",
            metadata={
                "parser": "got-ocr",
                "ocr_type": self.ocr_type,
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

        # Check if it's already an image
        if doc_bytes[:8] == b'\x89PNG\r\n\x1a\n' or doc_bytes[:2] == b'\xff\xd8':
            return [doc_bytes]

        # Open PDF and convert to images
        doc = fitz.open(stream=doc_bytes, filetype="pdf")
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render at 200 DPI for high quality OCR
            mat = fitz.Matrix(200 / 72, 200 / 72)
            pix = page.get_pixmap(matrix=mat)
            images.append(pix.tobytes("png"))

        doc.close()
        return images


# Alias for backward compatibility
GOTOCRAdapter = GLOCRAdapter
