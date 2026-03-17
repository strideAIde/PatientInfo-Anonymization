from __future__ import annotations

import re

from config import NAME_STOP_WORDS

_NAME_LABEL = re.compile(r"(?i)^(subject\s+name|name\s*:?)$")
_STRONG_ID_LABEL = re.compile(r"(?i)^patient\s+id\s*:?$")
_WEAK_ID_LABEL = re.compile(r"(?i)^id\s*:?$")
_KIER_ID_VALUE = re.compile(r"(?i)^kier\s*\d{4,8}$")
_KIER_ANYWHERE = re.compile(r"(?i)kier[i1l]?\s*\d{3,8}")

_STOP_WORDS: frozenset[str] = frozenset(w.upper() for w in NAME_STOP_WORDS)

_NON_NAME_WORDS: frozenset[str] = frozenset(["OF", "AND", "THE", "FOR", "WITH", "IN", "AT", "TO"])
_KIER_LIKE = re.compile(r"(?i)^K[A-Z0-9]{2,4}\s?[A-Za-z0-9]{0,2}\d{4,8}$")


def looks_like_patient_name(value: str) -> bool:
    words = value.strip().split()
    if not 2 <= len(words) <= 5:
        return False
    return all(
        w.isupper() and (1 <= len(w) <= 12) and w not in _NON_NAME_WORDS
        for w in words
    )


def looks_like_kier_id(value: str) -> bool:
    return bool(_KIER_LIKE.fullmatch(value.strip()))


def is_name_label(token: str) -> bool:
    return bool(_NAME_LABEL.fullmatch(token.strip()))


def is_strong_id_label(token: str) -> bool:
    return bool(_STRONG_ID_LABEL.fullmatch(token.strip()))


def is_weak_id_label(token: str) -> bool:
    return bool(_WEAK_ID_LABEL.fullmatch(token.strip()))


def is_kier_id(token: str) -> bool:
    return bool(_KIER_ID_VALUE.fullmatch(token.strip()))


def contains_kier_id(token: str) -> bool:
    return bool(_KIER_ANYWHERE.search(token.strip()))


def kier_id_start(token: str) -> int:
    m = _KIER_ANYWHERE.search(token.strip())
    return m.start() if m else 0


def is_name_stop_word(token: str) -> bool:
    return token.strip().upper() in _STOP_WORDS
