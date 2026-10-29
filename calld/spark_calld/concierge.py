"""The LFM3 "front desk": fast local text calls that shape what goes to Hermes and what gets spoken.

Two jobs, both on a small llama-server (LFM3-1.7B when installed, else LFM3-Audio in text mode):

* ``brief``: turn a rambling spoken request into a crisp task list for the big brain (Qwen via Hermes).
* ``spoken_summary``: boil a long technical answer down to a few plain sentences a person would say
  on the phone; the full text still goes to the chat.
"""
from __future__ import annotations

import logging
import re

import httpx

logger = logging.getLogger(__name__)

BRIEF_SYSTEM = (
    "You are the front desk for a powerful but slow AI agent. Rewrite the user's spoken request as a "
    "short, complete brief the agent can act on: one line of goal, then a numbered task list (1-6 items), "
    "then 'Details:' with every name, number, date, place or preference the user mentioned. Keep the "
    "user's language. Do not answer the request yourself. Do not add tasks the user did not ask for."
)
SUMMARY_SYSTEM = (
    "You summarize an AI agent's written answer for a person listening on the phone. Reply in {language} "
    "with two to four short plain sentences: what was done or found, the one or two facts that matter, "
    "and anything the person must decide. No lists, no markdown, no code, no URLs, no technical detail "
    "unless the person asked for it. End by mentioning that the full answer is in the chat only if the "
    "answer was long."
)
_FENCE = re.compile(r"```.*?```", re.S)
_MD = re.compile(r"[*_`#>|]|https?://\S+")


def plain(text: str, limit: int = 900) -> str:
    """Cheap fallback when no LFM3 text server answers: strip markdown and cut at a sentence."""
    text = _FENCE.sub(" (code omitted) ", text or "")
    text = _MD.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        cut = text[:limit]
        text = cut[: cut.rfind(". ") + 1] if ". " in cut else cut
    return text


class Concierge:
    def __init__(self, urls: list[str], *, language: str = "English", model: str = "", timeout_s: float = 60.0):
        self._urls = [u.rstrip("/") for u in urls if u]
        self._language = language
        self._model = model
        self._timeout = timeout_s

    async def _chat(self, system: str, user: str, max_tokens: int = 300) -> str | None:
        payload = {"model": self._model or "default", "temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1,
                   "max_tokens": max_tokens, "reasoning_format": "none",
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        for base in self._urls:
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(base + "/v1/chat/completions", json=payload)
                if resp.status_code != 200:
                    continue
                text = ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                if "</think>" in text:
                    text = text.split("</think>", 1)[1]
                text = text.strip()
                if text:
                    return text
            except httpx.HTTPError as e:
                logger.debug("concierge %s unavailable: %s", base, e)
        return None

    async def brief(self, transcript: str, caller: str = "") -> str:
        transcript = (transcript or "").strip()
        if len(transcript.split()) < 12:      # short requests are already a brief
            return transcript
        out = await self._chat(BRIEF_SYSTEM, (f"Caller: {caller}\n" if caller else "") + "Spoken request:\n" + transcript, 400)
        if not out:
            return transcript
        return f"{out}\n\n(Original spoken request: \"{transcript}\")"

    async def spoken_summary(self, answer: str, *, threshold_chars: int = 350) -> str:
        answer = (answer or "").strip()
        if len(answer) <= threshold_chars and "```" not in answer and "\n-" not in answer and "\n1." not in answer:
            return plain(answer)
        out = await self._chat(SUMMARY_SYSTEM.format(language=self._language), answer[:12000], 220)
        return plain(out) if out else plain(answer, 500)
