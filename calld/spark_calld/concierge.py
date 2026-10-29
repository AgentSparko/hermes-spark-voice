"""The LFM3 "front desk" for calls (async httpx; mirrors plugin/concierge.py).

* ``brief()``: a spoken, rambling request becomes a task list the slow big brain (Qwen via Hermes) acts on.
* ``spoken_summary()``: whatever the agent wrote becomes what a person would *say* about it: one to three
  short first-person sentences, no technical tokens, no reasoning preamble. The full text still goes to
  the chat as text, so the voice never has to carry the details.
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
    "You turn an AI agent's written report into what a human colleague would say out loud on the phone. "
    "Reply in {language}, first person, one to three short sentences, at most {max_words} words. Say what "
    "was done or found and what, if anything, you need from the listener. Example of the tone: "
    "\"I committed the two PRs locally, let me know if I should push and merge.\" Never read out IP "
    "addresses, ports, URLs, file paths, file names, function or variable names, commit hashes, version "
    "strings, commands or code: say 'the server', 'the file', 'the config' instead. No lists, no markdown, "
    "no headings, no 'reasoning' or 'summary' labels, no preamble. The listener can read the full text in "
    "the chat, so do not try to be complete."
)
SPOKEN_MAX_CHARS = 420
_FENCE = re.compile(r"```.*?```", re.S)
_THINK = re.compile(r"<\s*(think|thinking|reasoning|thought|analysis)\b[^>]*>.*?(</\s*\1\s*>|\Z)", re.S | re.I)
_LABEL_LINE = re.compile(r"^\s*(\*\*|#+\s*)?(reasoning|thinking|thoughts?|analysis|final answer|answer|summary|response)\s*:?\s*(\*\*)?\s*$", re.I | re.M)
_LEAD_LABEL = re.compile(r"^\s*(\*\*|#+\s*)?(reasoning|thinking|thoughts?|analysis)\s*:?\s*(\*\*)?\s*", re.I)
_MD = re.compile(r"[*_`#>|~]|https?://\S+|www\.\S+")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{2,5})?\b|\blocalhost(?::\d{2,5})?\b")
_PATH = re.compile(r"(?<![\w.])(?:~|\.{1,2})?/[\w.\-/]+|\b[\w.\-]+\.(?:py|sh|yaml|yml|json|md|toml|txt|log|service|gguf|cfg|ini|env|js|ts|html|css|go|rs|c|h|cpp)\b", re.I)
_IDENT = re.compile(r"\b[a-z_][a-z0-9_]*(?:_[a-z0-9_]+)+\b(?:\(\))?|\b[a-z][a-z0-9]*(?:[A-Z][a-z0-9]+){1,}\b(?:\(\))?|\b\w+\(\)")
_HASH = re.compile(r"\b[0-9a-f]{7,40}\b")
_VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)+[\w.-]*\b|\bb\d{4,6}\b")
_FLAG = re.compile(r"(?<!\w)--?[a-z][\w-]*")
_TECH = (_IP, _PATH, _IDENT, _HASH, _VERSION, _FLAG)


def strip_reasoning(text: str) -> str:
    """Drop <think>-style blocks, 'Reasoning:' sections and label lines the agent left in its reply."""
    text = _THINK.sub("", text or "")
    # a "Reasoning:" section before an "Answer:"/"Final answer" label: keep what follows the label
    m = re.search(r"(?im)^\s*(\*\*|#+\s*)?(final answer|answer|response|summary)\s*:?\s*(\*\*)?\s*$", text)
    if m and _LEAD_LABEL.match(text):
        text = text[m.end():]
    text = _LEAD_LABEL.sub("", text, count=1)
    text = _LABEL_LINE.sub("", text)
    return text.strip()


def humanize(text: str) -> str:
    """Replace tokens a voice should never read letter by letter with plain words."""
    text = _FENCE.sub(" the code ", text or "")
    text = _MD.sub("", text)
    text = _IP.sub("the local address", text)
    text = _PATH.sub("the file", text)
    text = _HASH.sub("the commit", text)
    text = _VERSION.sub("the new version", text)
    text = _FLAG.sub("the option", text)
    text = _IDENT.sub("that", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_technical(text: str) -> bool:
    return any(p.search(text) for p in _TECH) or "```" in text or "\n- " in text or "\n1." in text or "|" in text


def plain(text: str, limit: int = SPOKEN_MAX_CHARS) -> str:
    """Deterministic fallback when no LFM3 text server answers: strip, humanize, cut at a sentence."""
    text = humanize(strip_reasoning(text))
    if len(text) > limit:
        cut = text[:limit]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[: end + 1] if end > limit // 3 else cut.rstrip() + "."
    return text


def needs_summary(answer: str, threshold_chars: int) -> bool:
    return len(answer) > threshold_chars or looks_technical(answer)


class Concierge:
    def __init__(self, urls: list[str], *, language: str = "English", model: str = "", timeout_s: float = 60.0, max_words: int = 60):
        self._urls = [u.rstrip("/") for u in urls if u]
        self._language = language
        self._model = model
        self._timeout = timeout_s
        self._max_words = max_words

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
                text = strip_reasoning(text)
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

    async def spoken_summary(self, answer: str, *, threshold_chars: int = 200) -> str:
        answer = strip_reasoning(answer)
        if not needs_summary(answer, threshold_chars):
            return plain(answer)
        out = await self._chat(SUMMARY_SYSTEM.format(language=self._language, max_words=self._max_words), answer[:12000], 160)
        return plain(out) if out else plain(answer)
