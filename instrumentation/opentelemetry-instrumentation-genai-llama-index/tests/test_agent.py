# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.llms import ChatMessage, MockFunctionCallingLLM
from llama_index.core.tools import FunctionTool

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import SpanKind, StatusCode


def _spans_named(span_exporter, name: str):
    return [s for s in span_exporter.get_finished_spans() if s.name == name]


@pytest.mark.asyncio
async def test_agent_run_span(
    span_exporter, instrument_llama_index_with_content
) -> None:
    agent = FunctionAgent(
        name="math-agent",
        description="Answers math questions.",
        system_prompt="Be concise.",
        llm=MockFunctionCallingLLM(is_chat_model=True),
        streaming=False,
    )

    result = await agent.run(user_msg="What is two plus two?")
    assert result.response.content

    spans = _spans_named(span_exporter, "invoke_agent math-agent")
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == SpanKind.INTERNAL
    assert span.status.status_code != StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert attrs[GenAIAttributes.GEN_AI_AGENT_NAME] == "math-agent"
    assert attrs[GenAIAttributes.GEN_AI_AGENT_DESCRIPTION] == (
        "Answers math questions."
    )
    assert isinstance(attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL], str)
    input_messages = json.loads(attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages == [
        {
            "parts": [{"content": "What is two plus two?", "type": "text"}],
            "role": "user",
        }
    ]
    output_messages = json.loads(attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["role"] == "assistant"
    assert json.loads(attrs[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS]) == [
        {"content": "Be concise.", "type": "text"}
    ]
    assert GenAIAttributes.GEN_AI_PROVIDER_NAME not in attrs


@pytest.mark.asyncio
async def test_delegated_inference_is_not_reemitted(
    span_exporter, instrument_llama_index, openai_llm, vcr
) -> None:
    agent = FunctionAgent(
        name="provider-agent",
        llm=openai_llm,
        streaming=False,
    )

    with vcr.use_cassette("inference.yaml"):
        result = await agent.run(user_msg="Hello!")
    assert result.response.content == "Hello! How can I assist you today?"

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["invoke_agent provider-agent"]
    assert all(
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] != "chat"
        for span in spans
    )


@pytest.mark.asyncio
async def test_agent_error_re_raises(
    span_exporter, instrument_llama_index
) -> None:
    agent = FunctionAgent(
        name="broken-agent",
        llm=MockFunctionCallingLLM(is_chat_model=True),
        streaming=False,
    )

    async def fail(*args, **kwargs):
        raise ValueError("agent failed")

    with patch.object(FunctionAgent, "take_step", side_effect=fail):
        with pytest.raises(ValueError, match="agent failed"):
            await agent.run(user_msg="fail")

    spans = _spans_named(span_exporter, "invoke_agent broken-agent")
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ValueError"


@pytest.mark.asyncio
async def test_react_agent_run_span(
    span_exporter, instrument_llama_index_with_content
) -> None:
    def response_generator(messages, **kwargs):
        return ChatMessage(
            role="assistant",
            content="Thought: I know the answer.\nAnswer: Four.",
        )

    agent = ReActAgent(
        name="react-math-agent",
        description="Answers math questions with ReAct reasoning.",
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=response_generator,
        ),
        streaming=False,
    )

    result = await agent.run(user_msg="What is two plus two?")
    assert result.response.content == "Four."

    spans = _spans_named(span_exporter, "invoke_agent react-math-agent")
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == SpanKind.INTERNAL
    assert span.status.status_code != StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert attrs[GenAIAttributes.GEN_AI_AGENT_NAME] == "react-math-agent"
    assert attrs[GenAIAttributes.GEN_AI_AGENT_DESCRIPTION] == (
        "Answers math questions with ReAct reasoning."
    )
    assert isinstance(attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL], str)
    assert json.loads(attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES]) == [
        {
            "parts": [{"content": "What is two plus two?", "type": "text"}],
            "role": "user",
        }
    ]
    output_messages = json.loads(attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"] == [
        {"content": "Four.", "type": "text"}
    ]


@pytest.mark.asyncio
async def test_react_agent_error_re_raises(
    span_exporter, instrument_llama_index
) -> None:
    agent = ReActAgent(
        name="broken-react-agent",
        llm=MockFunctionCallingLLM(is_chat_model=True),
        streaming=False,
    )

    with patch.object(
        ReActAgent,
        "take_step",
        side_effect=ValueError("react agent failed"),
    ):
        with pytest.raises(ValueError, match="react agent failed"):
            await agent.run(user_msg="fail")

    spans = _spans_named(span_exporter, "invoke_agent broken-react-agent")
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ValueError"


def test_sync_tool_span(
    span_exporter, instrument_llama_index_with_content
) -> None:
    def multiply(value: int) -> int:
        """Multiply a number by two."""
        return value * 2

    tool = FunctionTool.from_defaults(multiply)
    result = tool(value=4)
    assert result.raw_output == 8

    spans = _spans_named(span_exporter, "execute_tool multiply")
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert attrs[GenAIAttributes.GEN_AI_TOOL_NAME] == "multiply"
    assert attrs[GenAIAttributes.GEN_AI_TOOL_TYPE] == "function"
    assert isinstance(attrs[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION], str)
    assert json.loads(attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]) == {
        "value": 4
    }
    assert attrs[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT] == 8


@pytest.mark.asyncio
async def test_async_tool_span_nested_under_agent(
    span_exporter, instrument_llama_index_with_content
) -> None:
    async def weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    tool = FunctionTool.from_defaults(async_fn=weather)

    def response_generator(messages, **kwargs):
        if any(message.role.value == "tool" for message in messages):
            return ChatMessage(role="assistant", content="It is sunny.")
        return ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(
                    tool_call_id="weather-call",
                    tool_name="weather",
                    tool_kwargs={"city": "Paris"},
                )
            ],
        )

    agent = FunctionAgent(
        name="weather-agent",
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=response_generator,
        ),
        tools=[tool],
        streaming=False,
    )
    await agent.run(user_msg="What is the weather?")

    agent_span = _spans_named(span_exporter, "invoke_agent weather-agent")[0]
    tool_spans = _spans_named(span_exporter, "execute_tool weather")
    assert len(tool_spans) == 1
    tool_span = tool_spans[0]
    assert tool_span.context.trace_id == agent_span.context.trace_id
    assert tool_span.parent is not None
    tool_attrs = dict(tool_span.attributes or {})
    assert tool_attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ID] == "weather-call"
    assert isinstance(tool_attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ID], str)


def test_tool_content_is_opt_in(span_exporter, instrument_llama_index) -> None:
    tool = FunctionTool.from_defaults(lambda value: value * 2, name="double")
    tool(value=3)

    span = _spans_named(span_exporter, "execute_tool double")[0]
    attrs = dict(span.attributes or {})
    assert GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS not in attrs
    assert GenAIAttributes.GEN_AI_TOOL_CALL_RESULT not in attrs


def test_tool_error_re_raises(span_exporter, instrument_llama_index) -> None:
    def broken() -> None:
        raise RuntimeError("tool failed")

    tool = FunctionTool.from_defaults(broken)
    with pytest.raises(RuntimeError, match="tool failed"):
        tool()

    spans = _spans_named(span_exporter, "execute_tool broken")
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"
