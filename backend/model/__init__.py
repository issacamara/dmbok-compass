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
]
