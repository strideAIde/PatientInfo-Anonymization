from __future__ import annotations

import re
from dataclasses import dataclass

from config import LINE_Y_TOLERANCE_PX
from anonymizer.pii.patterns import (
    contains_kier_id,
    is_kier_id,
    is_name_label,
    is_name_stop_word,
    is_strong_id_label,
    is_weak_id_label,
    kier_id_start,
    looks_like_kier_id,
    looks_like_patient_name,
)

_COMBINED_NAME = re.compile(r"(?i)^(subject\s+name|name)\s*[;:]?\s+(.+)$")
_COMBINED_STRONG_ID = re.compile(r"(?i)^(patient\s+id)\s*[;:]?\s+(.+)$")
_COMBINED_WEAK_ID = re.compile(r"(?i)^(id)\s*[;:]?\s+(kier\s*\d{4,8}.*)$")

_HONORIFICS: frozenset[str] = frozenset(["DR", "MR", "MRS", "MS", "PROF", "REV", "ST"])


@dataclass(frozen=True)
class OcrToken:
    text: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionResult:
    redact_boxes: list[tuple[int, int, int, int]]


def detect(tokens: list[OcrToken]) -> DetectionResult:
    expanded = []
    for tok in tokens:
        expanded.extend(_split_combined_token(tok))
    lines = _group_into_lines(expanded)
    boxes: list[tuple[int, int, int, int]] = []
    for line in lines:
        boxes.extend(_scan_line(line))
    return DetectionResult(redact_boxes=boxes)


def _group_into_lines(tokens: list[OcrToken]) -> list[list[OcrToken]]:
    if not tokens:
        return []

    def _y_mid(tok: OcrToken) -> float:
        return (tok.bbox[1] + tok.bbox[3]) / 2.0

    ordered = sorted(tokens, key=_y_mid)
    lines: list[list[OcrToken]] = []
    current: list[OcrToken] = [ordered[0]]
    ref_y = _y_mid(ordered[0])

    for tok in ordered[1:]:
        y = _y_mid(tok)
        if abs(y - ref_y) <= LINE_Y_TOLERANCE_PX:
            current.append(tok)
            ref_y = y
        else:
            lines.append(sorted(current, key=lambda t: t.bbox[0]))
            current = [tok]
            ref_y = y

    lines.append(sorted(current, key=lambda t: t.bbox[0]))
    return lines


def _split_combined_token(tok: OcrToken) -> list[OcrToken]:
    x1, y1, x2, y2 = tok.bbox
    for pattern in (_COMBINED_NAME, _COMBINED_STRONG_ID, _COMBINED_WEAK_ID):
        m = pattern.match(tok.text)
        if m:
            label_text = m.group(1)
            value_text = m.group(2).strip()
            if not value_text:
                break
            total = len(tok.text)
            split_x = x1 + int((len(tok.text) - len(value_text)) / total * (x2 - x1))
            split_x = max(x1 + 1, min(split_x, x2 - 1))
            return [
                OcrToken(text=label_text, bbox=(x1, y1, split_x, y2)),
                OcrToken(text=value_text, bbox=(split_x, y1, x2, y2)),
            ]

    sep_pos = next((i for i, c in enumerate(tok.text) if c in ":;"), -1)
    if sep_pos > 0:
        label = tok.text[:sep_pos].strip()
        value = tok.text[sep_pos + 1:].strip()
        if value and 1 <= len(label) <= 10 and " " not in label and label.upper() not in _HONORIFICS:
            if looks_like_patient_name(value) or looks_like_kier_id(value):
                canonical = "Name" if looks_like_patient_name(value) else "ID"
                total = len(tok.text)
                split_x = x1 + int((len(tok.text) - len(value)) / total * (x2 - x1))
                split_x = max(x1 + 1, min(split_x, x2 - 1))
                return [
                    OcrToken(text=canonical, bbox=(x1, y1, split_x, y2)),
                    OcrToken(text=value, bbox=(split_x, y1, x2, y2)),
                ]

    return [tok]


def _scan_line(line: list[OcrToken]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    n = len(line)

    for i, tok in enumerate(line):
        if is_name_label(tok.text):
            for j in range(i + 1, n):
                val = line[j]
                if (
                    is_name_stop_word(val.text)
                    or is_name_label(val.text)
                    or is_strong_id_label(val.text)
                    or is_weak_id_label(val.text)
                    or any(c in val.text for c in ":;")
                ):
                    break
                boxes.append(val.bbox)

        elif is_strong_id_label(tok.text) and i + 1 < n:
            boxes.append(line[i + 1].bbox)

        elif is_weak_id_label(tok.text) and i + 1 < n:
            val = line[i + 1]
            if is_kier_id(val.text) or contains_kier_id(val.text) or looks_like_kier_id(val.text):
                boxes.append(val.bbox)

        elif contains_kier_id(tok.text) and not is_kier_id(tok.text) and not is_name_label(tok.text):
            x1, y1, x2, y2 = tok.bbox
            start = kier_id_start(tok.text)
            total = len(tok.text)
            val_x1 = x1 + int(start / total * (x2 - x1)) if total > 0 else x1
            boxes.append((val_x1, y1, x2, y2))

    return boxes
