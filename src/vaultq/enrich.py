from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import httpx

from vaultq.embedding_provider import ApiProviderConfig, resolve_llm_provider

logger = logging.getLogger("vaultq.enrich")

LLM_RETRIES = max(1, int(os.getenv("LLM_RETRIES", os.getenv("KNOWLEDGE_RETRIES", "4"))))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", os.getenv("KNOWLEDGE_TIMEOUT", "120")))
LLM_MAX_OUTPUT_TOKENS = max(
    256, int(os.getenv("LLM_MAX_OUTPUT_TOKENS", os.getenv("KNOWLEDGE_MAX_OUTPUT_TOKENS", "900")))
)
LLM_TPM_LIMIT = max(1000, int(os.getenv("LLM_TPM_LIMIT", os.getenv("KNOWLEDGE_TPM_LIMIT", "15000"))))

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
JSON_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


@dataclass
class ChunkRow:
    id: int
    chunk_index: int
    heading_path: str
    text_content: str
    start_line: int
    end_line: int


class TokenRateLimiter:
    def __init__(self, tokens_per_minute: int) -> None:
        self.tokens_per_minute = max(1, tokens_per_minute)
        self.events: Deque[Tuple[float, int]] = deque()

    def _prune(self) -> None:
        cutoff = time.time() - 60.0
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def current_usage(self) -> int:
        self._prune()
        return sum(tokens for _, tokens in self.events)

    def consume(self, tokens: int) -> None:
        tokens = max(1, tokens)
        while True:
            self._prune()
            current = self.current_usage()
            if current + tokens <= self.tokens_per_minute:
                self.events.append((time.time(), tokens))
                return
            sleep_for = max(0.25, 60.0 - (time.time() - self.events[0][0]))
            logger.info("Rate limiter waiting %.1fs", sleep_for)
            time.sleep(sleep_for)


def estimate_tokens(text: str) -> int:
    words = TOKEN_PATTERN.findall((text or "").strip())
    punctuation = re.findall(r"[^\w\s]", text or "", re.UNICODE)
    return max(1, int(math.ceil(len(words) * 1.35 + len(punctuation) * 0.35)))


def nonempty_str(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def list_of_strings(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := nonempty_str(item))]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_for_dedupe(text: str) -> str:
    return re.sub(r"[^\w\s'-]+", "", re.sub(r"\s+", " ", text.strip().lower()))


def truncate_text(text: str, max_tokens: int) -> str:
    words = text.split()
    if estimate_tokens(text) <= max_tokens:
        return text
    kept: List[str] = []
    for word in words:
        kept.append(word)
        if estimate_tokens(" ".join(kept)) >= max_tokens:
            kept = kept[:-1]
            break
    return " ".join(kept).strip()


def stable_hash(parts: Sequence[str]) -> str:
    payload = "\n".join(part.strip() for part in parts if part and part.strip())
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = JSON_FENCE_PATTERN.sub("", text or "").strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Could not parse JSON object from model response")


def build_body_text(object_type: str, payload: Dict[str, Any]) -> str:
    title = nonempty_str(payload.get("title"))
    lines: List[str] = []
    if title:
        lines.append(f"Title: {title}")
    if object_type == "segment_summary":
        summary = nonempty_str(payload.get("summary"))
        if summary:
            lines.append(f"Summary: {summary}")
    elif object_type == "document_summary":
        summary = nonempty_str(payload.get("summary"))
        key_topics = list_of_strings(payload.get("key_topics"))
        if summary:
            lines.append(f"Summary: {summary}")
        if key_topics:
            lines.append("Key topics: " + "; ".join(key_topics))
    elif object_type == "concept":
        definition = nonempty_str(payload.get("definition"))
        signals = list_of_strings(payload.get("signals"))
        applications = list_of_strings(payload.get("applications"))
        if definition:
            lines.append(f"Definition: {definition}")
        if signals:
            lines.append("Signals: " + "; ".join(signals))
        if applications:
            lines.append("Applications: " + "; ".join(applications))
    elif object_type == "principle":
        statement = nonempty_str(payload.get("statement"))
        rationale = nonempty_str(payload.get("rationale"))
        caveats = list_of_strings(payload.get("caveats"))
        if statement:
            lines.append(f"Statement: {statement}")
        if rationale:
            lines.append(f"Rationale: {rationale}")
        if caveats:
            lines.append("Caveats: " + "; ".join(caveats))
    elif object_type == "method":
        goal = nonempty_str(payload.get("goal"))
        inputs = list_of_strings(payload.get("inputs"))
        steps = list_of_strings(payload.get("steps"))
        outputs = list_of_strings(payload.get("outputs"))
        mistakes = list_of_strings(payload.get("mistakes"))
        if goal:
            lines.append(f"Goal: {goal}")
        if inputs:
            lines.append("Inputs: " + "; ".join(inputs))
        if steps:
            lines.append("Steps: " + "; ".join(steps))
        if outputs:
            lines.append("Outputs: " + "; ".join(outputs))
        if mistakes:
            lines.append("Mistakes: " + "; ".join(mistakes))
    elif object_type == "sop":
        prerequisites = list_of_strings(payload.get("prerequisites"))
        steps = list_of_strings(payload.get("steps"))
        success_criteria = list_of_strings(payload.get("success_criteria"))
        failure_modes = list_of_strings(payload.get("failure_modes"))
        if prerequisites:
            lines.append("Prerequisites: " + "; ".join(prerequisites))
        if steps:
            lines.append("Steps: " + "; ".join(steps))
        if success_criteria:
            lines.append("Success criteria: " + "; ".join(success_criteria))
        if failure_modes:
            lines.append("Failure modes: " + "; ".join(failure_modes))
    return "\n".join(lines).strip()


def dedupe_knowledge_objects(objects: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    index_by_key: Dict[Tuple[str, str], int] = {}
    for obj in objects:
        object_type = nonempty_str(obj.get("object_type"))
        title_key = normalize_for_dedupe(nonempty_str(obj.get("title")))
        if not object_type or not title_key or object_type in {"segment_summary", "document_summary"}:
            deduped.append(dict(obj))
            continue
        key = (object_type, title_key)
        existing_index = index_by_key.get(key)
        if existing_index is None:
            index_by_key[key] = len(deduped)
            deduped.append(dict(obj))
            continue
        existing = deduped[existing_index]
        merged_chunk_ids = sorted(
            {
                int(chunk_id)
                for chunk_id in (existing.get("source_chunk_ids") or []) + (obj.get("source_chunk_ids") or [])
            }
        )
        existing["source_chunk_ids"] = merged_chunk_ids
        existing["source_line_start"] = min(
            int(existing.get("source_line_start") or 10**9),
            int(obj.get("source_line_start") or 10**9),
        )
        existing["source_line_end"] = max(
            int(existing.get("source_line_end") or 0),
            int(obj.get("source_line_end") or 0),
        )
        existing["confidence"] = max(
            safe_float(existing.get("confidence"), 0.0),
            safe_float(obj.get("confidence"), 0.0),
        )
        if len(nonempty_str(obj.get("body_text"))) > len(nonempty_str(existing.get("body_text"))):
            existing.update(obj)
        deduped[existing_index] = existing
    return deduped


def build_extractor(client: Optional[httpx.Client] = None) -> Optional["KnowledgeExtractor"]:
    provider = resolve_llm_provider()
    if not provider.api_key or not provider.model:
        return None
    http_client = client or httpx.Client()
    return KnowledgeExtractor(http_client, TokenRateLimiter(LLM_TPM_LIMIT), provider)


class KnowledgeExtractor:
    def __init__(self, client: httpx.Client, rate_limiter: TokenRateLimiter, provider: ApiProviderConfig) -> None:
        self.client = client
        self.rate_limiter = rate_limiter
        self.provider = provider
        self.model = provider.model

    def _chat_completion(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        reserved_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + LLM_MAX_OUTPUT_TOKENS
        self.rate_limiter.consume(reserved_tokens)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": LLM_MAX_OUTPUT_TOKENS,
        }
        last_exc: Optional[Exception] = None
        for attempt in range(1, LLM_RETRIES + 1):
            try:
                resp = self.client.post(
                    f"{self.provider.base_url}/chat/completions",
                    headers=self.provider.headers,
                    json=payload,
                    timeout=LLM_TIMEOUT,
                )
                if resp.status_code == 200:
                    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    return extract_json_object(content)
                if resp.status_code in (408, 409, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"LLM transient error {resp.status_code}: {resp.text}")
                raise RuntimeError(f"LLM error {resp.status_code}: {resp.text}")
            except Exception as exc:
                last_exc = exc
                if attempt >= LLM_RETRIES:
                    break
                time.sleep(min(2**attempt, 20))
        raise RuntimeError("LLM request failed") from last_exc

    def extract_chunk_objects(self, document: Dict[str, Any], chunk: ChunkRow) -> List[Dict[str, Any]]:
        system_prompt = (
            "You extract reusable knowledge from markdown note sections. "
            "Stay strictly grounded in the provided source text. "
            "Do not invent frameworks or missing steps. Return only valid JSON."
        )
        user_prompt = f"""
Extract grounded reusable knowledge from this markdown section.

Return a JSON object with this exact structure:
{{
  "segment_summary": {{
    "title": "short section title",
    "summary": "2-4 sentence summary"
  }},
  "concepts": [
    {{
      "title": "concept name",
      "definition": "short definition",
      "signals": ["observable cue"],
      "applications": ["where to use it"],
      "confidence": 0.0
    }}
  ],
  "principles": [
    {{
      "title": "principle name",
      "statement": "clear rule",
      "rationale": "why it matters",
      "caveats": ["limitation"],
      "confidence": 0.0
    }}
  ],
  "methods": [
    {{
      "title": "method name",
      "goal": "desired outcome",
      "inputs": ["input"],
      "steps": ["step 1", "step 2"],
      "outputs": ["output"],
      "mistakes": ["common mistake"],
      "confidence": 0.0
    }}
  ],
  "sops": [
    {{
      "title": "SOP name",
      "prerequisites": ["needed before start"],
      "steps": ["step 1", "step 2"],
      "success_criteria": ["success signal"],
      "failure_modes": ["failure mode"],
      "confidence": 0.0
    }}
  ]
}}

Rules:
- If a type is not clearly present, return an empty array for that type.
- Prefer precision over volume.
- Max 2 concepts, 2 principles, 1 method, 1 sop.
- Confidence must be between 0 and 1.

Document title: {document.get("title") or "Untitled"}
Relative path: {document.get("rel_path") or ""}
Heading path: {chunk.heading_path}
Chunk index: {chunk.chunk_index}
Line span: {chunk.start_line} to {chunk.end_line}

Source text:
{truncate_text(html.unescape(chunk.text_content), max_tokens=650)}
""".strip()
        parsed = self._chat_completion(system_prompt, user_prompt)
        objects: List[Dict[str, Any]] = []
        segment_summary = parsed.get("segment_summary") if isinstance(parsed.get("segment_summary"), dict) else {}
        summary_title = nonempty_str(segment_summary.get("title")) or f"Segment {chunk.chunk_index}"
        summary_text = nonempty_str(segment_summary.get("summary"))
        if summary_text:
            payload = {"title": summary_title, "summary": summary_text}
            objects.append(
                {
                    "object_type": "segment_summary",
                    "title": summary_title,
                    "body_text": build_body_text("segment_summary", payload),
                    "json_payload": payload,
                    "confidence": 1.0,
                    "source_chunk_ids": [chunk.id],
                    "source_line_start": chunk.start_line,
                    "source_line_end": chunk.end_line,
                    "content_hash": stable_hash(("segment_summary", summary_title, summary_text)),
                }
            )

        def add_objects(source_key: str, object_type: str, allowed_fields: Sequence[str]) -> None:
            values = parsed.get(source_key)
            if not isinstance(values, list):
                return
            for obj in values:
                if not isinstance(obj, dict):
                    continue
                title = nonempty_str(obj.get("title"))
                if not title:
                    continue
                normalized_payload = {field: obj.get(field) for field in allowed_fields}
                normalized_payload["title"] = title
                body_text = build_body_text(object_type, normalized_payload)
                if not body_text:
                    continue
                objects.append(
                    {
                        "object_type": object_type,
                        "title": title,
                        "body_text": body_text,
                        "json_payload": normalized_payload,
                        "confidence": max(0.0, min(1.0, safe_float(obj.get("confidence"), 0.5))),
                        "source_chunk_ids": [chunk.id],
                        "source_line_start": chunk.start_line,
                        "source_line_end": chunk.end_line,
                        "content_hash": stable_hash((object_type, title, body_text)),
                    }
                )

        add_objects("concepts", "concept", ("definition", "signals", "applications", "confidence"))
        add_objects("principles", "principle", ("statement", "rationale", "caveats", "confidence"))
        add_objects("methods", "method", ("goal", "inputs", "steps", "outputs", "mistakes", "confidence"))
        add_objects("sops", "sop", ("prerequisites", "steps", "success_criteria", "failure_modes", "confidence"))
        return objects

    def build_document_summary(self, document: Dict[str, Any], segment_summaries: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not segment_summaries:
            return None
        summary_corpus = "\n\n".join(
            f"- {obj['title']}: {obj['json_payload'].get('summary', '')}" for obj in segment_summaries
        )
        system_prompt = "You synthesize grounded document-level knowledge from section summaries. Return only valid JSON."
        user_prompt = f"""
Create a grounded document-level summary from these section summaries.

Return a JSON object with this structure:
{{
  "title": "short overall title",
  "summary": "120-260 word summary",
  "key_topics": ["topic 1", "topic 2", "topic 3"]
}}

Rules:
- Focus on practical reusable knowledge.
- Keep key_topics to 5-10 concise entries.
- Do not add claims not supported by the section summaries.

Document title: {document.get("title") or "Untitled"}
Relative path: {document.get("rel_path") or ""}

Section summaries:
{truncate_text(summary_corpus, max_tokens=1200)}
""".strip()
        parsed = self._chat_completion(system_prompt, user_prompt)
        title = nonempty_str(parsed.get("title")) or (document.get("title") or f"Document {document.get('id')}")
        summary = nonempty_str(parsed.get("summary"))
        key_topics = list_of_strings(parsed.get("key_topics"))
        if not summary:
            return None
        payload = {"title": title, "summary": summary, "key_topics": key_topics}
        return {
            "object_type": "document_summary",
            "title": title,
            "body_text": build_body_text("document_summary", payload),
            "json_payload": payload,
            "confidence": 1.0,
            "source_chunk_ids": [obj["source_chunk_ids"][0] for obj in segment_summaries if obj.get("source_chunk_ids")],
            "source_line_start": min(int(obj.get("source_line_start") or 1) for obj in segment_summaries),
            "source_line_end": max(int(obj.get("source_line_end") or 1) for obj in segment_summaries),
            "content_hash": stable_hash(("document_summary", title, summary, ";".join(key_topics))),
        }
