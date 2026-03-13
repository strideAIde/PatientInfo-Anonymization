from __future__ import annotations

from dataclasses import dataclass

from config import LINE_Y_TOLERANCE_PX
from anonymizer.pii.patterns import (
    is_kier_id,
    is_name_label,
    is_name_stop_word,
    is_strong_id_label,
    is_weak_id_label,
)


@dataclass(frozen=True)
class OcrToken:
    text: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionResult:
    redact_boxes: list[tuple[int, int, int, int]]


def detect(tokens: list[OcrToken]) -> DetectionResult:
    lines = _group_into_lines(tokens)
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
        else:
            lines.append(sorted(current, key=lambda t: t.bbox[0]))
            current = [tok]
            ref_y = y

    lines.append(sorted(current, key=lambda t: t.bbox[0]))
    return lines


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
                ):
                    break
                boxes.append(val.bbox)

        elif is_strong_id_label(tok.text) and i + 1 < n:
            boxes.append(line[i + 1].bbox)

        elif is_weak_id_label(tok.text) and i + 1 < n and is_kier_id(line[i + 1].text):
            boxes.append(line[i + 1].bbox)

    return boxes
