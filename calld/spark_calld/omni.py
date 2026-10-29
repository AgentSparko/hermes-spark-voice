"""LFM3-Omni-2B on llama-server: describe the latest video frame of the call."""
from __future__ import annotations

import base64
import io
import logging

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def jpeg_b64(rgb: np.ndarray, quality: int = 80) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OmniClient:
    def __init__(self, base_url: str, timeout_s: float = 60.0):
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s

    async def describe(self, rgb: np.ndarray, question: str) -> str:
        payload = {
            "model": "LFM3-Omni-2B", "temperature": 0.1, "max_tokens": 160,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64(rgb)}},
                {"type": "text", "text": question}]}],
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._base + "/v1/chat/completions", json=payload)
            resp.raise_for_status()
            text = ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            return text.strip() or "I could not make anything out."
        except httpx.HTTPError as e:
            logger.warning("omni unreachable: %s", e)
            return "The vision model is not available right now."
