"""Ask the Hermes agent (its OpenAI-compatible API server) to do real work for the caller."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .concierge import Concierge, plain

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    spoken: str
    full_text: str = ""
    ok: bool = True


class HermesClient:
    def __init__(self, base_url: str, api_key: str, model: str = "hermes-agent", timeout_s: float = 240.0,
                 concierge: Concierge | None = None):
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._model = model
        self._timeout = timeout_s
        self._concierge = concierge

    async def ask(self, request: str, *, session_id: str, caller: str = "", want_full: bool = False) -> Answer:
        system = ("This request came in over a live phone call and was already shaped into a brief by the "
                  "voice front desk. Do the work fully. Write the complete answer as you normally would (it is "
                  "posted to the chat); a separate model will speak a short summary of it.")
        if caller:
            system += f" The caller is {caller}."
        brief = await self._concierge.brief(request, caller) if self._concierge else request
        payload = {"model": self._model, "stream": False,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": brief}]}
        headers = {"Authorization": f"Bearer {self._key}", "X-Hermes-Session-Id": session_id}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._base + "/chat/completions", json=payload, headers=headers)
            if resp.status_code != 200:
                logger.warning("hermes api %s: %s", resp.status_code, resp.text[:200])
                return Answer(f"Hermes could not take the request right now, status {resp.status_code}.", ok=False)
            data = resp.json()
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            if not text.strip():
                return Answer("Hermes finished but had nothing to say.", ok=False)
            spoken = await self._concierge.spoken_summary(text) if self._concierge else plain(text)
            return Answer(spoken, full_text=text if want_full else "")
        except httpx.TimeoutException:
            return Answer("Hermes is still working on it; I will tell you when it is done.", ok=False)
        except httpx.HTTPError as e:
            logger.warning("hermes api unreachable: %s", e)
            return Answer("I cannot reach Hermes right now.", ok=False)

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(self._base + "/models", headers={"Authorization": f"Bearer {self._key}"})
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
