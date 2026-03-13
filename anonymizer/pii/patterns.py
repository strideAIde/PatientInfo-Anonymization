from __future__ import annotations

import re

from config import NAME_STOP_WORDS

_NAME_LABEL = re.compile(r"(?i)^(subject\s+name|name\s*:?)$")
_STRONG_ID_LABEL = re.compile(r"(?i)^patient\s+id\s*:?$")
_WEAK_ID_LABEL = re.compile(r"(?i)^id\s*:?$")
_KIER_ID_VALUE = re.compile(r"(?i)^kier\s*\d{4,8}$")

_STOP_WORDS: frozenset[str] = frozenset(w.upper() for w in NAME_STOP_WORDS)


def is_name_label(token: str) -> bool:
    return bool(_NAME_LABEL.fullmatch(token.strip()))


def is_strong_id_label(token: str) -> bool:
    return bool(_STRONG_ID_LABEL.fullmatch(token.strip()))


def is_weak_id_label(token: str) -> bool:
    return bool(_WEAK_ID_LABEL.fullmatch(token.strip()))


def is_kier_id(token: str) -> bool:
    return bool(_KIER_ID_VALUE.fullmatch(token.strip()))


def is_name_stop_word(token: str) -> bool:
    return token.strip().upper() in _STOP_WORDS
