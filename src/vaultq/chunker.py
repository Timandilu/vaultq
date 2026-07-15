from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "120"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "320"))
CHUNK_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "220"))
CHUNK_OVERLAP_FRAC = float(os.getenv("CHUNK_OVERLAP_FRAC", "0.12"))

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_PARAGRAPH_SEP = re.compile(r"\n{2,}")
_SENTENCE_SEP = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class Chunk:
    text: str
    index: int
    heading_path: str
    section_title: str
    token_count: int
    start_line: int
    end_line: int


@dataclass
class Section:
    text: str
    heading_path: str
    section_title: str
    start_line: int
    end_line: int


def token_len(text: str) -> int:
    return len(ENC.encode(text))


def _build_heading_path(headings: list[tuple[int, str]]) -> str:
    if not headings:
        return ""
    return " > ".join(title for _, title in headings)


def _line_number_at(text: str, offset: int) -> int:
    return text[: max(0, offset)].count("\n") + 1


def _extract_sections(text: str) -> List[Section]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        lines = text.count("\n") + 1
        return [Section(text=text.strip(), heading_path="", section_title="", start_line=1, end_line=lines)]

    sections: List[Section] = []
    heading_stack: list[tuple[int, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        level = len(match.group(1))
        title = match.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        raw = text[start:end].strip()
        if not raw:
            continue
        sections.append(
            Section(
                text=raw,
                heading_path=_build_heading_path(heading_stack),
                section_title=title,
                start_line=_line_number_at(text, start),
                end_line=_line_number_at(text, end),
            )
        )
    return sections


def _split_by_sentences(text: str, max_tokens: int) -> List[str]:
    sentences = _SENTENCE_SEP.split(text)
    if len(sentences) <= 1:
        tokens = ENC.encode(text)
        return [ENC.decode(tokens[i : i + max_tokens]).strip() for i in range(0, len(tokens), max_tokens) if ENC.decode(tokens[i : i + max_tokens]).strip()]
    pieces: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence.strip()
        if token_len(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                pieces.append(current)
            if token_len(sentence.strip()) > max_tokens:
                tokens = ENC.encode(sentence.strip())
                pieces.extend(
                    ENC.decode(tokens[i : i + max_tokens]).strip()
                    for i in range(0, len(tokens), max_tokens)
                    if ENC.decode(tokens[i : i + max_tokens]).strip()
                )
                current = ""
            else:
                current = sentence.strip()
    if current:
        pieces.append(current)
    return pieces


def _split_section(section: Section, max_tokens: int) -> List[Tuple[str, int, int]]:
    if token_len(section.text) <= max_tokens:
        return [(section.text.strip(), section.start_line, section.end_line)]

    lines = section.text.splitlines()
    paragraphs: List[Tuple[str, int, int]] = []
    start_idx = 0
    buffer: List[str] = []
    para_start = section.start_line
    for idx, line in enumerate(lines):
        if not buffer:
            para_start = section.start_line + idx
        if line.strip():
            buffer.append(line)
        else:
            if buffer:
                paragraph = "\n".join(buffer).strip()
                paragraphs.append((paragraph, para_start, section.start_line + idx))
                buffer = []
            start_idx = idx + 1
    if buffer:
        paragraphs.append(( "\n".join(buffer).strip(), para_start, section.start_line + len(lines) - 1))

    pieces: List[Tuple[str, int, int]] = []
    current_parts: List[str] = []
    current_start = section.start_line
    current_end = section.start_line
    for paragraph, start_line, end_line in paragraphs:
        candidate = "\n\n".join(current_parts + [paragraph]).strip()
        if current_parts and token_len(candidate) > max_tokens:
            pieces.append(("\n\n".join(current_parts).strip(), current_start, current_end))
            current_parts = []
        if token_len(paragraph) > max_tokens:
            if current_parts:
                pieces.append(("\n\n".join(current_parts).strip(), current_start, current_end))
                current_parts = []
            for sentence_piece in _split_by_sentences(paragraph, max_tokens):
                pieces.append((sentence_piece, start_line, end_line))
            current_start = end_line + 1
            current_end = end_line
            continue
        if not current_parts:
            current_start = start_line
        current_parts.append(paragraph)
        current_end = end_line
    if current_parts:
        pieces.append(("\n\n".join(current_parts).strip(), current_start, current_end))
    return pieces


def _merge_small_chunks(chunks: List[Tuple[str, str, str, int, int]]) -> List[Tuple[str, str, str, int, int]]:
    if not chunks:
        return []
    merged = [chunks[0]]
    for text, heading_path, section_title, start_line, end_line in chunks[1:]:
        prev_text, prev_path, prev_title, prev_start, prev_end = merged[-1]
        if token_len(prev_text) < CHUNK_MIN_TOKENS and token_len(f"{prev_text}\n\n{text}") <= CHUNK_MAX_TOKENS:
            merged[-1] = (f"{prev_text}\n\n{text}".strip(), prev_path or heading_path, prev_title or section_title, prev_start, end_line)
        else:
            merged.append((text, heading_path, section_title, start_line, end_line))
    return merged


def _add_overlap(chunks: List[Tuple[str, str, str, int, int]]) -> List[Tuple[str, str, str, int, int]]:
    if len(chunks) <= 1 or CHUNK_OVERLAP_FRAC <= 0:
        return chunks
    overlap_tokens = max(0, int(CHUNK_TARGET_TOKENS * CHUNK_OVERLAP_FRAC))
    result = [chunks[0]]
    for idx in range(1, len(chunks)):
        prev_text = chunks[idx - 1][0]
        prev_tokens = ENC.encode(prev_text)
        overlap_text = ENC.decode(prev_tokens[-overlap_tokens:]).strip() if overlap_tokens and prev_tokens else ""
        text, heading_path, section_title, start_line, end_line = chunks[idx]
        combined = f"{overlap_text}\n\n{text}".strip() if overlap_text else text
        if token_len(combined) > CHUNK_MAX_TOKENS:
            combined = text
        result.append((combined, heading_path, section_title, start_line, end_line))
    return result


def chunk_markdown(text: str) -> List[Chunk]:
    if not text or not text.strip():
        return []
    raw_chunks: List[Tuple[str, str, str, int, int]] = []
    for section in _extract_sections(text):
        for piece_text, start_line, end_line in _split_section(section, CHUNK_MAX_TOKENS):
            raw_chunks.append((piece_text, section.heading_path, section.section_title, start_line, end_line))
    raw_chunks = _merge_small_chunks(raw_chunks)
    raw_chunks = _add_overlap(raw_chunks)
    return [
        Chunk(
            text=piece_text.strip(),
            index=idx,
            heading_path=heading_path,
            section_title=section_title,
            token_count=token_len(piece_text),
            start_line=start_line,
            end_line=end_line,
        )
        for idx, (piece_text, heading_path, section_title, start_line, end_line) in enumerate(raw_chunks)
        if piece_text.strip()
    ]
