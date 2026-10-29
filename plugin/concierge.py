"""LFM3 front desk for voice notes (sync, urllib only; mirrors spark_calld/concierge.py).

* ``brief()``: a spoken, rambling request becomes a task list the slow big brain (Qwen via Hermes) acts on.
* ``spoken_summary()``: a long written answer becomes two to four plain sentences for the voice bubble;
  the full text still goes to the chat as text.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

BRIEF_SYSTEM = (
    "You are the front desk for a powerful but slow AI agent. Rewrite the user's spoken request as a "
    "short, complete brief the agent can act on: one line of goal, then a numbered task list (1-6 items), "
    "then 'Details:' with every name, number, date, place or preference the user mentioned. Keep the "
    "user's language. Do not answer the request yourself. Do not add tasks the user did not ask for."
)
SUMMARY_SYSTEM = (
    "You summarize an AI agent's written answer for a person listening to a voice message. Reply in "
    "{language} with two to four short plain sentences: what was done or found, the one or two facts that "
    "matter, and anything the person must decide. No lists, no markdown, no code, no URLs, no technical "
    "detail unless the person asked for it. If the answer was long, end with: the full answer is in the chat."
)
_FENCE = re.compile(r"```.*?```", re.S)
_MD = re.compile(r"[*_`#>|]|https?://\S+")


def plain(text: str, limit: int = 900) -> str:
    text = _FENCE.sub(" (code omitted) ", text or "")
    text = _MD.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        cut = text[:limit]
        text = cut[: cut.rfind(". ") + 1] if ". " in cut else cut
    return text


def needs_summary(answer: str, threshold_chars: int) -> bool:
    return len(answer) > threshold_chars or "```" in answer or "\n- " in answer or "\n1." in answer or "|" in answer


class Concierge:
    def __init__(self, urls: list[str], *, language: str = "English", timeout: int = 60):
        self._urls = [u.rstrip("/") for u in urls if u]
        self._language = language
        self._timeout = timeout

    def _chat(self, system: str, user: str, max_tokens: int) -> str | None:
        payload = {"model": "default", "temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1, "max_tokens": max_tokens,
                   "reasoning_format": "none",
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        data = json.dumps(payload).encode()
        for base in self._urls:
            req = urllib.request.Request(base + "/v1/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 - loopback
                    body = json.loads(resp.read().decode())
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                logger.debug("concierge %s unavailable: %s", base, e)
                continue
            text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            if text.strip():
                return text.strip()
        return None

    def brief(self, transcript: str) -> str:
        transcript = (transcript or "").strip()
        if len(transcript.split()) < 12:
            return transcript
        out = self._chat(BRIEF_SYSTEM, "Spoken request:\n" + transcript, 400)
        return transcript if not out else f"{transcript}\n\n[Front desk brief for the agent]\n{out}"

    def spoken_summary(self, answer: str, *, threshold_chars: int = 350) -> str:
        answer = (answer or "").strip()
        if not needs_summary(answer, threshold_chars):
            return plain(answer)
        out = self._chat(SUMMARY_SYSTEM.format(language=self._language), answer[:12000], 220)
        return plain(out) if out else plain(answer, 500)
