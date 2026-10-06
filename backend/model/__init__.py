"""Provider-neutral model generation contracts and local test adapters."""

from .protocol import (
    GenerationCandidate,
    GenerationError,
    GenerationErrorCode,
    GenerationRequest,
    GenerationResult,
    ModelAdapter,
    Passage,
    TimeoutBudget,
    UsageMetadata,
    parse_candidate,
)
from .gemini import FALLBACK_MODEL, PRIMARY_MODEL, GeminiAdapter, create_gemini_adapter

__all__ = [
    "GenerationCandidate",
    "GenerationError",
    "GenerationErrorCode",
    "GenerationRequest",
    "GenerationResult",
    "ModelAdapter",
    "Passage",
    "TimeoutBudget",
    "UsageMetadata",
    "parse_candidate",
    "FALLBACK_MODEL",
    "PRIMARY_MODEL",
    "GeminiAdapter",
    "create_gemini_adapter",
]
