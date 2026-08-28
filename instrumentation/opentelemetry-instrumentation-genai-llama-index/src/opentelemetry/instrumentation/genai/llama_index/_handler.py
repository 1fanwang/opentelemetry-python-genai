# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from base64 import b64decode
from collections.abc import Mapping, Sequence
from mimetypes import guess_type
from typing import Any, cast

from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    ToolCall,
    ToolCallResult,
)
from llama_index.core.base.llms.types import (
    AudioBlock,
    ChatMessage,
    DocumentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.core.instrumentation.span import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.tools import BaseTool, FunctionTool, ToolOutput
from pydantic import PrivateAttr

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    GenAIInvocation,
    ToolInvocation,
)
from opentelemetry.util.genai.types import (
    Blob,
    FunctionToolDefinition,
    GenericToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    Reasoning,
    Text,
    ToolCallRequest,
    ToolDefinition,
    Uri,
)


def _method_name(span_id: str) -> str:
    return span_id.partition("-")[0].rsplit(".", 1)[-1]


def _chat_message_parts(message: ChatMessage) -> list[MessagePart]:
    parts: list[MessagePart] = []
    for block in message.blocks:
        if isinstance(block, TextBlock) and block.text:
            parts.append(Text(content=block.text))
        elif isinstance(block, ToolCallBlock):
            parts.append(
                ToolCallRequest(
                    arguments=block.tool_kwargs,
                    name=block.tool_name,
                    id=block.tool_call_id,
                )
            )
        elif isinstance(block, ThinkingBlock) and block.content:
            parts.append(Reasoning(content=block.content))
        elif isinstance(block, ImageBlock):
            part = _media_part(
                data=block.image,
                path=block.path,
                url=block.url,
                mime_type=block.image_mimetype,
                modality="image",
            )
            if part is not None:
                parts.append(part)
        elif isinstance(block, AudioBlock):
            part = _media_part(
                data=block.audio,
                path=block.path,
                url=block.url,
                mime_type=_audio_mime_type(block.format),
                modality="audio",
            )
            if part is not None:
                parts.append(part)
        elif isinstance(block, DocumentBlock):
            part = _media_part(
                data=block.data,
                path=block.path,
                url=block.url,
                mime_type=block.document_mimetype,
                modality="document",
            )
            if part is not None:
                parts.append(part)
    return parts


def _media_part(
    *,
    data: object,
    path: object,
    url: object,
    mime_type: str | None,
    modality: str,
) -> MessagePart | None:
    if isinstance(data, bytes):
        # LlamaIndex normalizes inline media to base64 bytes during validation.
        try:
            return Blob(
                content=b64decode(data, validate=True),
                mime_type=mime_type,
                modality=modality,
            )
        except ValueError:
            pass
    reference = url or path
    if reference is not None:
        return Uri(
            uri=str(reference),
            mime_type=mime_type,
            modality=modality,
        )
    return None


def _audio_mime_type(format_: str | None) -> str | None:
    if not format_ or "/" in format_:
        return format_
    return guess_type(f"file.{format_}")[0] or f"audio/{format_}"


def _input_message(message: ChatMessage) -> InputMessage:
    return InputMessage(
        role=message.role.value,
        parts=_chat_message_parts(message),
    )


def _output_message(message: ChatMessage) -> OutputMessage:
    return OutputMessage(
        role=message.role.value,
        parts=_chat_message_parts(message),
        finish_reason=(
            "tool_calls"
            if any(
                isinstance(block, ToolCallBlock) for block in message.blocks
            )
            else "stop"
        ),
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
    if isinstance(user_message, ChatMessage):
        messages.append(_input_message(user_message))
    elif isinstance(user_message, str) and user_message:
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
    if not isinstance(candidate, BaseTool):
        return None
    # Tool metadata may be incomplete even when the agent can otherwise run.
    try:
        metadata = candidate.metadata
        name = metadata.name
        if not name:
            return None
        description = metadata.description or None
    except Exception:
        return None
    if not isinstance(candidate, FunctionTool):
        return GenericToolDefinition(name=name, type=type(candidate).__name__)
    try:
        parameters = cast(
            dict[str, Any], cast(Any, metadata).get_parameters_dict()
        )
    except Exception:
        return None
    return FunctionToolDefinition(
        name=name,
        description=description,
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


def _tool_arguments(
    tool: FunctionTool, bound_args: inspect.BoundArguments
) -> dict[str, Any]:
    positional = bound_args.arguments.get("args")
    args = (
        tuple(cast(Sequence[Any], positional))
        if isinstance(positional, Sequence)
        else ()
    )
    keyword = bound_args.arguments.get("kwargs")
    kwargs: dict[str, Any] = {}
    if isinstance(keyword, Mapping):
        kwargs.update(cast(Mapping[str, Any], keyword))
    try:
        arguments = dict(
            inspect.signature(tool.real_fn)
            .bind_partial(*args, **kwargs)
            .arguments
        )
    except (TypeError, ValueError):
        arguments = {"args": list(args), **kwargs}
    if tool.ctx_param_name:
        arguments.pop(tool.ctx_param_name, None)
    return arguments


class _LlamaIndexInvocation(BaseSpan):
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


class LlamaIndexSpanHandler(BaseSpanHandler[_LlamaIndexInvocation]):
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
    ) -> _LlamaIndexInvocation | None:
        method_name = _method_name(id_)
        invocation: GenAIInvocation

        if isinstance(instance, BaseWorkflowAgent) and method_name == "run":
            capture_content = self._handler.should_capture_content()
            agent_name = instance.name or type(instance).__name__
            request_model = _request_model(instance)
            agent_description = instance.description
            input_messages = (
                _agent_input(bound_args) if capture_content else []
            )
            tool_definitions = _tool_definitions(instance)
            system_prompt = instance.system_prompt
            system_instruction: list[MessagePart] = (
                [Text(content=system_prompt)]
                if capture_content and system_prompt
                else []
            )
            agent_invocation = self._handler.invoke_local_agent(
                request_model=request_model,
                agent_name=agent_name,
            )
            agent_invocation.agent_description = agent_description
            agent_invocation.input_messages = input_messages
            agent_invocation.tool_definitions = tool_definitions
            agent_invocation.system_instruction = system_instruction
            invocation = agent_invocation
        elif method_name == "call_tool" and isinstance(
            (tool_call := bound_args.arguments.get("ev")), ToolCall
        ):
            tool_invocation = self._handler.tool(
                tool_call.tool_name,
                tool_call_id=tool_call.tool_id,
                tool_type="function",
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
                tool_invocation.arguments = _tool_arguments(
                    instance, bound_args
                )
            invocation = tool_invocation
        else:
            return None

        return _LlamaIndexInvocation(
            id_=id_,
            parent_id=parent_span_id,
            invocation=invocation,
        )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        result: Any | None = None,
        **kwargs: Any,
    ) -> _LlamaIndexInvocation | None:
        span = self.open_spans.get(id_)
        if span is None:
            return None
        if isinstance(span._invocation, AgentInvocation):
            if self._handler.should_capture_content():
                _set_agent_output(span._invocation, result)
        elif isinstance(span._invocation, ToolInvocation):
            tool_output: ToolOutput | None = None
            if isinstance(result, ToolCallResult):
                tool_output = result.tool_output
            elif isinstance(result, ToolOutput):
                tool_output = result
            if tool_output is not None:
                if span._invocation.should_capture_content_on_span:
                    span._invocation.tool_result = tool_output.raw_output
                if tool_output.is_error:
                    # LlamaIndex reports failures such as unknown tools without an
                    # exception, so provide one to record error telemetry:
                    # https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/agent/workflow/base_agent.py
                    error = (
                        tool_output.exception
                        if isinstance(tool_output.exception, BaseException)
                        else RuntimeError(tool_output.content)
                    )
                    span._invocation.fail(error)
                    return span
        span._invocation.stop()
        return span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        err: BaseException | None = None,
        **kwargs: Any,
    ) -> _LlamaIndexInvocation | None:
        span = self.open_spans.get(id_)
        if span is None:
            return None
        if err is None:
            span._invocation.stop()
        else:
            span._invocation.fail(err)
        return span
