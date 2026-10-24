"""Ollama implementation of the ANISHA LLM contract."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional, Sequence

from .base import ChatMessage, LLMResponse, ToolCall


class OllamaLLMClient:
    """Async Ollama client with normalized tool calls and structured output."""

    def __init__(
        self,
        *,
        model: str,
        host: str = "http://127.0.0.1:11434",
        context_length: int = 8_192,
        max_output_tokens: int = 512,
        temperature: float = 0.2,
        keep_alive: str = "10m",
        think: bool = False,
        client: Optional[Any] = None,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        if client is None:
            try:
                from ollama import AsyncClient
            except ImportError as exc:
                raise RuntimeError(
                    "The Ollama Python package is required. Install backend requirements."
                ) from exc
            client = AsyncClient(host=host)

        self.model = model
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.think = think
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
            "messages": [_to_ollama_message(message) for message in messages],
            "stream": False,
            "think": self.think,
            "options": {
                "num_ctx": self.context_length,
                "num_predict": self.max_output_tokens,
                "temperature": self.temperature,
            },
            "keep_alive": self.keep_alive,
        }
        if tools:
            payload["tools"] = list(tools)
        if response_schema:
            payload["format"] = dict(response_schema)

        response = await self._client.chat(**payload)
        message = _value(response, "message", {})
        content = str(_value(message, "content", "") or "")
        raw_tool_calls = _value(message, "tool_calls", ()) or ()
        tool_calls = tuple(
            _normalize_tool_call(call, index) for index, call in enumerate(raw_tool_calls)
        )
        if not tool_calls:
            tool_calls = _tool_calls_from_content(content)
        if tool_calls:
            # Ollama's Qwen templates choose either assistant content or tool
            # calls when serializing history. Preserve the canonical tool-call
            # structure so the next request can consume the tool responses.
            content = ""
        return LLMResponse(
            content=content,
            thinking=_optional_string(_value(message, "thinking", None)),
            tool_calls=tool_calls,
        )


def _to_ollama_message(message: ChatMessage) -> dict[str, Any]:
    result: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "type": "function",
                "function": {
                    "index": index,
                    "name": call.name,
                    "arguments": call.arguments,
                },
            }
            for index, call in enumerate(message.tool_calls)
        ]
    if message.tool_name:
        result["tool_name"] = message.tool_name
    return result


def _normalize_tool_call(value: Any, index: int) -> ToolCall:
    function = _value(value, "function", {})
    arguments = _value(function, "arguments", {}) or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"_invalid_json": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return ToolCall(
        id=str(_value(value, "id", "") or f"ollama-tool-{index}"),
        name=str(_value(function, "name", "")),
        arguments=arguments,
    )


def _value(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_string(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _tool_calls_from_content(content: str) -> tuple[ToolCall, ...]:
    """Parse Qwen's documented <tool_call> JSON fallback format.

    Native Ollama tool calls remain the primary path. This only handles tagged
    JSON blocks, rather than attempting to execute arbitrary model prose.
    """

    calls: list[ToolCall] = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL):
        decoder = json.JSONDecoder()
        position = 0
        while position < len(block):
            while position < len(block) and block[position].isspace():
                position += 1
            if position >= len(block):
                break
            try:
                value, position = decoder.raw_decode(block, position)
            except json.JSONDecodeError:
                break
            if not isinstance(value, Mapping) or not isinstance(value.get("name"), str):
                continue
            arguments = value.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if not isinstance(arguments, Mapping):
                continue
            calls.append(
                ToolCall(
                    id=f"ollama-content-tool-{len(calls)}",
                    name=value["name"],
                    arguments=dict(arguments),
                )
            )
    return tuple(calls)
