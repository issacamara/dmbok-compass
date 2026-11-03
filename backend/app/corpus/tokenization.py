"""Pinned, provider-independent token counting for corpus boundaries."""

from __future__ import annotations

import tiktoken


TOKENIZER_ENCODING = "cl100k_base"
TOKENIZER_PACKAGE_VERSION = "0.12.0"
_ENCODING = tiktoken.get_encoding(TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    """Return the stable cl100k_base count used by corpus segmentation."""

    return len(_ENCODING.encode(text, disallowed_special=()))
