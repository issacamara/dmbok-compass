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
]
