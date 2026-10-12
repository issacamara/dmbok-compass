"""Verify both configured OpenRouter routes with native LangChain chat models.

Use only authored, non-corpus text. Set OPENROUTER_API_KEY,
OPENROUTER_PRIMARY_MODEL, and OPENROUTER_FALLBACK_MODEL explicitly.
"""

from __future__ import annotations

import os

import openrouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel


class ProbeAnswer(BaseModel):
    answer: str


def main() -> None:
    key = os.environ["OPENROUTER_API_KEY"].strip()
    models = [
        os.environ["OPENROUTER_PRIMARY_MODEL"],
        os.environ["OPENROUTER_FALLBACK_MODEL"],
    ]
    if not key.strip() or any(not model.strip() for model in models):
        raise ValueError("Explicit API key and both route IDs are required")
    schema_supported = os.environ.get("OPENROUTER_JSON_SCHEMA_SUPPORTED", "true").lower()
    if schema_supported not in {"true", "false"}:
        raise ValueError("OPENROUTER_JSON_SCHEMA_SUPPORTED must be true or false")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Return a JSON object with one string field named answer."),
            ("human", "Set answer to the single word ok."),
        ]
    )
    chains = []
    for model_id in models:
        # The integration's max_retries=0 omits retry_config, which otherwise
        # leaves the OpenRouter SDK's one-hour default backoff in effect.
        sdk = openrouter.OpenRouter(api_key=key, retry_config=None, timeout_ms=8000)
        model = ChatOpenRouter(
            model=model_id,
            api_key=key,
            client=sdk,
            max_retries=0,
            timeout=8000,
            temperature=0,
            max_tokens=64,
        )
        if schema_supported == "true":
            chain = prompt | model.with_structured_output(ProbeAnswer, method="json_schema")
        else:
            chain = (
                prompt
                | model.bind(response_format={"type": "json_object"})
                | PydanticOutputParser(pydantic_object=ProbeAnswer)
            )
        chains.append(chain)
        result = chain.invoke({})
        assert isinstance(result, ProbeAnswer), type(result)
        assert result.answer.strip().lower() == "ok", result.answer
        print(f"LangChain OpenRouter structured output passed: {model_id}")

    def eligible_primary_failure(_):
        raise TimeoutError("simulated eligible primary timeout")

    fallback = RunnableLambda(eligible_primary_failure).with_fallbacks(
        [chains[1]], exceptions_to_handle=(TimeoutError,)
    )
    result = fallback.invoke({})
    assert isinstance(result, ProbeAnswer) and result.answer.strip().lower() == "ok"
    print("LangChain eligible fallback through configured live route: passed")


if __name__ == "__main__":
    main()
