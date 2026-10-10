"""Check both configured OpenRouter routes with synthetic evidence before release."""

from __future__ import annotations

import asyncio
import os

from .openrouter import OPENROUTER_FALLBACK_MODEL, OPENROUTER_PRIMARY_MODEL, OpenRouterAdapter
from .protocol import GenerationError, GenerationRequest, Passage


async def main() -> None:
    key = os.environ["OPENROUTER_API_KEY"]
    models = (
        os.environ.get("OPENROUTER_PRIMARY_MODEL", OPENROUTER_PRIMARY_MODEL),
        os.environ.get("OPENROUTER_FALLBACK_MODEL", OPENROUTER_FALLBACK_MODEL),
    )
    if len(set(models)) != 2:
        raise SystemExit("Primary and fallback must be distinct models.")
    adapter = OpenRouterAdapter(key)
    request = GenerationRequest(
        prompt="Using only the supplied evidence, name the synthetic quality dimension and cite it.",
        passages=[Passage(
            passage_id="smoke-1", page=1, section="Synthetic test",
            text="The synthetic quality dimension is sample completeness.",
        )],
    )
    for model in models:
        try:
            result = await adapter._generate_model(request, model)
            if not result.candidate.answer_text or not any(
                citation.citation_id == "smoke-1" for citation in result.candidate.citations
            ):
                raise ValueError("missing synthetic answer or citation")
        except GenerationError as exc:
            raise SystemExit(
                f"OpenRouter route failed: {model} ({exc.code.value}, HTTP {exc.status_code or 'n/a'})"
            ) from None
        except ValueError as exc:
            raise SystemExit(f"OpenRouter route failed: {model} ({exc})") from None
        print(f"OpenRouter route passed: {model}")


if __name__ == "__main__":
    asyncio.run(main())
