"""Provider-neutral generation contracts and model adapters."""

from .openrouter import (
    OPENROUTER_FALLBACK_MODEL,
    OPENROUTER_PRIMARY_MODEL,
    OpenRouterAdapter,
)
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
    "OpenRouterAdapter",
    "OPENROUTER_FALLBACK_MODEL",
    "OPENROUTER_PRIMARY_MODEL",
    "Passage",
    "TimeoutBudget",
    "UsageMetadata",
    "parse_candidate",
    "FALLBACK_MODEL",
    "PRIMARY_MODEL",
    "GeminiAdapter",
    "create_gemini_adapter",
]
