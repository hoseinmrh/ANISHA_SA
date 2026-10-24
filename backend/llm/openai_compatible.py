"""Optional OpenAI-compatible adapter for changing providers later."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Optional, Sequence

from .base import ChatMessage, LLMResponse, ToolCall


class OpenAICompatibleLLMClient:
    """Chat-completions adapter for OpenAI and compatible providers."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: Optional[float] = 0.2,
        max_output_tokens: int = 512,
        max_tokens_parameter: str = "max_tokens",
        client: Optional[Any] = None,
    ) -> None:
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if max_tokens_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("unsupported max token parameter")
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The OpenAI Python package is required for this provider."
                ) from exc
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_tokens_parameter = max_tokens_parameter
        self._client = client

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        response_schema: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_message(message) for message in messages],
            self.max_tokens_parameter: self.max_output_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = list(tools)
            # A phase exposes only tools that are currently safe. Keeping the
            # choice serial prevents a model from selecting a later-phase tool
            # before the runtime has evaluated the first tool result.
            payload["parallel_tool_calls"] = False
            payload["tool_choice"] = "required"
        if response_schema and not tools:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "anisha_response",
                    "strict": True,
                    "schema": _strict_json_schema(response_schema),
                },
            }

        response = await self._client.chat.completions.create(**payload)
        message = response.choices[0].message
        tool_calls = tuple(
            _normalize_tool_call(call, index)
            for index, call in enumerate(message.tool_calls or ())
        )
        return LLMResponse(content=message.content or "", tool_calls=tool_calls)


def _to_openai_message(message: ChatMessage) -> dict[str, Any]:
    result: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        result["tool_call_id"] = message.tool_call_id
    return result


def _normalize_tool_call(value: Any, index: int) -> ToolCall:
    arguments = value.function.arguments or "{}"
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = {"_invalid_json": arguments}
    return ToolCall(
        id=value.id or f"openai-tool-{index}",
        name=value.function.name,
        arguments=parsed,
    )


def _strict_json_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Convert Pydantic-style schemas to OpenAI strict structured output.

    Pydantic omits fields with defaults from ``required`` and permits arbitrary
    mapping keys. Strict structured output requires every property to be present
    and every object to reject additional properties. ANISHA's terminal schemas
    use known fields, so requiring their defaulted values is safe and removes a
    class of partially populated model responses.
    """

    normalized = copy.deepcopy(dict(schema))

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            properties = node.get("properties")
            if node.get("type") == "object" or isinstance(properties, dict):
                object_properties = properties if isinstance(properties, dict) else {}
                node["additionalProperties"] = False
                node["required"] = list(object_properties)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(normalized)
    return normalized
