# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, cast

from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    ToolCall,
    ToolCallResult,
)
from llama_index.core.base.llms.types import ChatMessage, TextBlock
from llama_index.core.instrumentation.span import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.tools import FunctionTool, ToolOutput
from pydantic import PrivateAttr

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    GenAIInvocation,
    ToolInvocation,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolDefinition,
)


def _method_name(span_id: str) -> str:
    return span_id.partition("-")[0].rsplit(".", 1)[-1]


def _chat_message_parts(message: ChatMessage) -> list[MessagePart]:
    return [
        Text(content=block.text)
        for block in message.blocks
        if isinstance(block, TextBlock) and block.text
    ]


def _input_message(message: ChatMessage) -> InputMessage:
    return InputMessage(
        role=message.role.value,
        parts=_chat_message_parts(message),
    )


def _output_message(message: ChatMessage) -> OutputMessage:
    return OutputMessage(
        role=message.role.value,
        parts=_chat_message_parts(message),
        finish_reason="stop",
    )


def _agent_input(bound_args: inspect.BoundArguments) -> list[InputMessage]:
    start_event = bound_args.arguments.get("start_event")
    if start_event is None:
        return []

    history_value: object = start_event.get("chat_history", default=None)
    history: Sequence[object] = (
        cast(Sequence[object], history_value)
        if isinstance(history_value, Sequence)
        else cast(Sequence[object], ())
    )
    messages = [
        _input_message(message)
        for message in history or []
        if isinstance(message, ChatMessage)
    ]
    user_message = start_event.get("user_msg", default=None)
    if isinstance(user_message, str) and user_message:
        messages.append(
            InputMessage(role="user", parts=[Text(content=user_message)])
        )
    return messages


def _request_model(agent: BaseWorkflowAgent) -> str | None:
    try:
        model_name = agent.llm.metadata.model_name
    except Exception:  # LLM integrations can compute metadata dynamically.
        model_name = getattr(agent.llm, "model", None)
    return model_name if isinstance(model_name, str) and model_name else None


def _tool_definition(candidate: object) -> ToolDefinition | None:
    if not isinstance(candidate, FunctionTool):
        return None
    metadata = candidate.metadata
    parameters = cast(
        dict[str, Any], cast(Any, metadata).get_parameters_dict()
    )
    return FunctionToolDefinition(
        name=metadata.get_name(),
        description=metadata.description or "",
        type="function",
        parameters=parameters,
    )


def _tool_definitions(agent: BaseWorkflowAgent) -> list[ToolDefinition] | None:
    definitions = [
        definition
        for candidate in cast(Sequence[object], cast(Any, agent).tools or ())
        if (definition := _tool_definition(candidate)) is not None
    ]
    return definitions or None


def _set_agent_output(invocation: AgentInvocation, result: Any) -> None:
    output = getattr(result, "result", None)
    response = getattr(output, "response", None)
    if isinstance(response, ChatMessage):
        invocation.output_messages = [_output_message(response)]


def _tool_arguments(bound_args: inspect.BoundArguments) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    positional = bound_args.arguments.get("args")
    if isinstance(positional, Sequence):
        arguments["args"] = list(cast(Sequence[Any], positional))
    keyword = bound_args.arguments.get("kwargs")
    if isinstance(keyword, Mapping):
        arguments.update(cast(Mapping[str, Any], keyword))
    return arguments


class _LlamaIndexSpan(BaseSpan):
    _invocation: GenAIInvocation = PrivateAttr()

    def __init__(
        self,
        *,
        id_: str,
        parent_id: str | None,
        invocation: GenAIInvocation,
    ) -> None:
        super().__init__(id_=id_, parent_id=parent_id)
        self._invocation = invocation


class LlamaIndexSpanHandler(BaseSpanHandler[_LlamaIndexSpan]):
    """Map LlamaIndex-owned agent and tool operations to GenAI spans."""

    _handler: TelemetryHandler = PrivateAttr()

    def __init__(self, handler: TelemetryHandler) -> None:
        super().__init__()
        self._handler = handler

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _LlamaIndexSpan | None:
        method_name = _method_name(id_)
        invocation: GenAIInvocation

        if isinstance(instance, BaseWorkflowAgent) and method_name == "run":
            agent_name = instance.name or type(instance).__name__
            agent_invocation = self._handler.invoke_local_agent(
                request_model=_request_model(instance),
                agent_name=agent_name,
            )
            agent_invocation.agent_description = instance.description
            agent_invocation.input_messages = _agent_input(bound_args)
            agent_invocation.tool_definitions = _tool_definitions(instance)
            if instance.system_prompt:
                agent_invocation.system_instruction = [
                    Text(content=instance.system_prompt)
                ]
            invocation = agent_invocation
        elif method_name == "call_tool" and isinstance(
            (tool_call := bound_args.arguments.get("ev")), ToolCall
        ):
            tool_invocation = self._handler.tool(
                tool_call.tool_name,
                tool_call_id=tool_call.tool_id,
            )
            if tool_invocation.should_capture_content_on_span:
                tool_invocation.arguments = cast(
                    dict[str, Any], cast(Any, tool_call).tool_kwargs
                )
            invocation = tool_invocation
        elif isinstance(instance, FunctionTool) and method_name in {
            "call",
            "acall",
        }:
            parent = self.open_spans.get(parent_span_id or "")
            if parent is not None and isinstance(
                parent._invocation, ToolInvocation
            ):
                return None
            metadata = instance.metadata
            tool_invocation = self._handler.tool(
                metadata.get_name(),
                tool_type="function",
                tool_description=metadata.description or None,
            )
            if tool_invocation.should_capture_content_on_span:
                tool_invocation.arguments = _tool_arguments(bound_args)
            invocation = tool_invocation
        else:
            return None

        return _LlamaIndexSpan(
            id_=id_, parent_id=parent_span_id, invocation=invocation
        )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        result: Any | None = None,
        **kwargs: Any,
    ) -> _LlamaIndexSpan | None:
        span = self.open_spans.get(id_)
        if span is None:
            return None
        if isinstance(span._invocation, AgentInvocation):
            _set_agent_output(span._invocation, result)
        elif (
            isinstance(span._invocation, ToolInvocation)
            and span._invocation.should_capture_content_on_span
        ):
            if isinstance(result, ToolCallResult):
                span._invocation.tool_result = result.tool_output.raw_output
            elif isinstance(result, ToolOutput):
                span._invocation.tool_result = result.raw_output
        span._invocation.stop()
        return span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        err: BaseException | None = None,
        **kwargs: Any,
    ) -> _LlamaIndexSpan | None:
        span = self.open_spans.get(id_)
        if span is None:
            return None
        if err is None:
            span._invocation.stop()
        else:
            span._invocation.fail(err)
        return span
