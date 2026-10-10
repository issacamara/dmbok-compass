"""Provider-neutral structured generation protocol.

The request contains only the prompt and retrieved passages needed for one
call.  Implementations must not persist either value.  Responses deliberately
contain only validated candidate data and content-free usage metadata.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.contracts.api import Citation, ContractModel, ModelAttempt, Outcome


class Passage(BaseModel):
    """Ephemeral evidence supplied to one generation request."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    passage_id: str = Field(min_length=1, max_length=256)
    page: int = Field(ge=1)
    section: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=20_000)


class TimeoutBudget(BaseModel):
    """A finite per-call budget, expressed in milliseconds."""

    model_config = ConfigDict(extra="forbid")

    total_ms: int = Field(default=8_000, ge=1, le=60_000)


class GenerationRequest(BaseModel):
    """The ephemeral input boundary for an adapter call."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=20_000)
    passages: list[Passage] = Field(default_factory=list, max_length=20)
    timeout: TimeoutBudget = Field(default_factory=TimeoutBudget)


class GenerationCandidate(ContractModel):
    """The only structured content an adapter may return."""

    outcome: Outcome
    answer_text: str | None = Field(default=None, max_length=20_000)
    synthesis: bool = False
    citations: list[Citation] = Field(default_factory=list, max_length=20)


class UsageMetadata(BaseModel):
    """Safe, content-free accounting data returned with a candidate."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=128)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    attempts: list[ModelAttempt] = Field(default_factory=list)


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: GenerationCandidate
    usage: UsageMetadata
    raw_output: str | None = Field(default=None, max_length=20_000)


class GenerationErrorCode(StrEnum):
    INVALID_OUTPUT = "invalid_output"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    AUTHENTICATION_FAILED = "authentication_failed"
    REQUEST_REJECTED = "request_rejected"
    UNKNOWN = "unknown"


class GenerationError(Exception):
    """Normalized, safe operational failure from any adapter."""

    def __init__(
        self,
        code: GenerationErrorCode,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        attempts: tuple[ModelAttempt, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code
        self.attempts = attempts

    def as_dict(self) -> dict[str, str | bool]:
        return {"code": self.code.value, "message": self.message, "retryable": self.retryable}


class ModelAdapter(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one candidate from ephemeral request data."""


def parse_candidate(payload: object) -> GenerationCandidate:
    """Validate provider output and normalize schema failures."""

    try:
        candidate = GenerationCandidate.model_validate(payload)
    except ValidationError as exc:
        raise GenerationError(
            GenerationErrorCode.INVALID_OUTPUT,
            "The model returned an invalid structured candidate.",
        ) from exc

    if candidate.outcome in {"answer", "qualified"} and not candidate.answer_text:
        raise GenerationError(
            GenerationErrorCode.INVALID_OUTPUT,
            "The model returned an incomplete structured candidate.",
        )
    if candidate.outcome == "refusal" and candidate.citations:
        raise GenerationError(
            GenerationErrorCode.INVALID_OUTPUT,
            "A refusal candidate cannot contain citations.",
        )
    return candidate


def normalize_error(error: BaseException) -> GenerationError:
    """Map adapter/provider failures to the stable operational error set."""

    if isinstance(error, GenerationError):
        return error
    return GenerationError(GenerationErrorCode.UNKNOWN, "Model generation failed.")
