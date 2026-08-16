# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable corpus for classifying real LangChain agent callbacks.

The expectations in ``CASES`` describe API boundaries, not the current
classifier's output.  Set ``LANGCHAIN_CORPUS_DUMP`` to print the complete
recorded callback stream for every case.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self
from unittest import mock
from uuid import UUID

import langchain.agents
import pytest

create_agent = getattr(langchain.agents, "create_agent", None)
if create_agent is None:
    pytest.skip(
        "create_agent requires a newer langchain version",
        allow_module_level=True,
    )

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    WorkflowInvocation,
)


class FakeModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Self:
        return self


@tool
def noop() -> str:
    """Do nothing."""
    return "ok"


class CallbackRecorder(BaseCallbackHandler):
    """Record callback ordering and the complete metadata emitted at starts."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): CallbackRecorder._jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [CallbackRecorder._jsonable(item) for item in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return repr(value)

    def _record(self, event: str, **values: Any) -> None:
        self.events.append(
            {
                "event": event,
                **{
                    key: self._jsonable(value) for key, value in values.items()
                },
            }
        )

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record(
            "chain_start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
            kwargs=kwargs,
        )

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record(
            "chain_end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            kwargs=kwargs,
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record(
            "chain_error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error=error,
            kwargs=kwargs,
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record(
            "chat_model_start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
            kwargs=kwargs,
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record(
            "llm_end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            kwargs=kwargs,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record(
            "llm_error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error=error,
            kwargs=kwargs,
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record(
            "tool_start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            tags=tags,
            metadata=metadata,
            kwargs=kwargs,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record(
            "tool_end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            kwargs=kwargs,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record(
            "tool_error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error=error,
            kwargs=kwargs,
        )


def _handler() -> tuple[OpenTelemetryLangChainCallbackHandler, mock.MagicMock]:
    telemetry = mock.MagicMock()

    workflow = mock.MagicMock(spec=WorkflowInvocation)
    workflow.span = mock.MagicMock()
    workflow.span.is_recording.return_value = False
    telemetry.workflow.return_value = workflow

    def make_agent(*args: Any, **kwargs: Any) -> mock.MagicMock:
        invocation = mock.MagicMock(spec=AgentInvocation)
        invocation.agent_name = kwargs.get("agent_name")
        invocation.span = mock.MagicMock()
        invocation.span.is_recording.return_value = False
        return invocation

    telemetry.invoke_local_agent.side_effect = make_agent
    return OpenTelemetryLangChainCallbackHandler(telemetry), telemetry


def _callbacks(
    recorder: CallbackRecorder,
    handler: OpenTelemetryLangChainCallbackHandler,
) -> RunnableConfig:
    return {"callbacks": [recorder, handler]}


def _top_level_unnamed(config: RunnableConfig) -> None:
    create_agent(
        FakeModel(responses=[AIMessage(content="done")]), [noop]
    ).invoke({"messages": [("user", "hi")]}, config)


def _top_level_named(config: RunnableConfig) -> None:
    create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        name="named_agent",
    ).invoke({"messages": [("user", "hi")]}, config)


def _nested_run_name(config: RunnableConfig) -> None:
    agent = create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        name="planner_agent",
    )
    RunnableLambda(
        lambda _: agent.invoke(
            {"messages": [("user", "hi")]}, {"run_name": "step1"}
        )
    ).with_config(run_name="planner").invoke({}, config)


def _nested_different_name_through_tool(config: RunnableConfig) -> None:
    inner = create_agent(
        FakeModel(responses=[AIMessage(content="inner done")]),
        [noop],
        name="researcher",
    )

    @tool
    def delegate(config: RunnableConfig) -> str:
        """Delegate to the researcher."""
        result = inner.invoke({"messages": [("user", "research")]}, config)
        return str(result["messages"][-1].content)

    outer = create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "delegate", "args": {}, "id": "call-1"}
                    ],
                ),
                AIMessage(content="outer done"),
            ]
        ),
        [delegate],
        name="manager",
    )
    outer.invoke({"messages": [("user", "start")]}, config)


def _nested_without_config_forwarding(config: RunnableConfig) -> None:
    inner = create_agent(
        FakeModel(responses=[AIMessage(content="inner done")]),
        [noop],
        name="researcher",
    )

    @tool
    def delegate() -> str:
        """Delegate to the researcher without forwarding configuration."""
        result = inner.invoke({"messages": [("user", "research")]})
        return str(result["messages"][-1].content)

    outer = create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "delegate", "args": {}, "id": "call-1"}
                    ],
                ),
                AIMessage(content="outer done"),
            ]
        ),
        [delegate],
        name="manager",
    )
    outer.invoke({"messages": [("user", "start")]}, config)


def _nested_unnamed_in_langgraph_node(config: RunnableConfig) -> None:
    inner = create_agent(
        FakeModel(responses=[AIMessage(content="done")]), [noop]
    )

    def invoke_agent(state: MessagesState) -> dict[str, Any]:
        return inner.invoke(state)

    builder = StateGraph(MessagesState)
    builder.add_node("LangGraph", invoke_agent)
    builder.add_edge(START, "LangGraph")
    builder.add_edge("LangGraph", END)
    builder.compile().invoke({"messages": [("user", "hi")]}, config)


def _same_display_name(config: RunnableConfig) -> None:
    inner = create_agent(
        FakeModel(responses=[AIMessage(content="inner done")]),
        [noop],
        name="assistant",
    )

    @tool
    def delegate(config: RunnableConfig) -> str:
        """Delegate to another assistant."""
        result = inner.invoke({"messages": [("user", "finish")]}, config)
        return str(result["messages"][-1].content)

    outer = create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "delegate", "args": {}, "id": "call-1"}
                    ],
                ),
                AIMessage(content="outer done"),
            ]
        ),
        [delegate],
        name="assistant",
    )
    outer.invoke({"messages": [("user", "start")]}, config)


def _user_langgraph_node_metadata(config: RunnableConfig) -> None:
    configured = dict(config)
    configured["metadata"] = {"langgraph_node": "user_supplied_node"}
    create_agent(
        FakeModel(responses=[AIMessage(content="done")]), [noop]
    ).invoke({"messages": [("user", "hi")]}, configured)


def _user_langgraph_node_with_config(config: RunnableConfig) -> None:
    create_agent(
        FakeModel(responses=[AIMessage(content="done")]), [noop]
    ).with_config(metadata={"langgraph_node": "user_supplied_node"}).invoke(
        {"messages": [("user", "hi")]}, config
    )


def _ordinary_sequence_in_tool(config: RunnableConfig) -> None:
    sequence = PromptTemplate.from_template("Value: {value}") | RunnableLambda(
        lambda prompt: prompt.to_string()
    )

    @tool
    def format_value(value: str) -> str:
        """Format a value."""
        return sequence.invoke({"value": value})

    agent = create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "format_value",
                            "args": {"value": "x"},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        [format_value],
        name="measured_agent",
    )
    agent.invoke({"messages": [("user", "format x")]}, config)


def _create_agent_internal_nodes(config: RunnableConfig) -> None:
    agent = create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "noop", "args": {}, "id": "call-1"}],
                ),
                AIMessage(content="done"),
            ]
        ),
        [noop],
        name="internal_node_control",
    )
    agent.invoke({"messages": [("user", "run noop")]}, config)


def _plain_langgraph_in_tool(config: RunnableConfig) -> None:
    builder = StateGraph(dict[str, Any])
    # Deliberately use the same default graph name and first-node name as an
    # unnamed create_agent.  This is a negative control for any discriminator
    # based on checkpoint namespace nesting or graph-entry relationships.
    builder.add_node("model", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    graph = builder.compile()

    @tool
    def run_graph(value: int) -> str:
        """Run a non-agent graph."""
        return str(graph.invoke({"value": value})["value"])

    agent = create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "run_graph",
                            "args": {"value": 1},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        [run_graph],
        name="outer_agent",
    )
    agent.invoke({"messages": [("user", "run graph")]}, config)


def _plain_langgraph(config: RunnableConfig) -> None:
    builder = StateGraph(dict[str, Any])
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    builder.compile().invoke({"value": 1}, config)


@dataclass(frozen=True)
class Case:
    run: Callable[[RunnableConfig], None]
    expected_agents: tuple[str, ...]
    expected_workflows: int


# Expected span sets, deliberately declared before inspecting a discriminator.
CASES = {
    "top_level_unnamed": Case(_top_level_unnamed, ("LangGraph",), 0),
    "top_level_named": Case(_top_level_named, ("named_agent",), 0),
    "nested_run_name_override": Case(_nested_run_name, ("planner_agent",), 1),
    # Known limitation: callbacks cannot distinguish either inner create_agent
    # root from metadata inherited from the enclosing create_agent.
    "known_limitation_nested_agent_with_config_forwarding": Case(
        _nested_different_name_through_tool, ("manager",), 0
    ),
    "known_limitation_nested_agent_without_config_forwarding": Case(
        _nested_without_config_forwarding, ("manager",), 0
    ),
    "nested_unnamed_in_outer_langgraph_node": Case(
        _nested_unnamed_in_langgraph_node, ("LangGraph",), 1
    ),
    "known_limitation_nested_agents_same_display_name": Case(
        _same_display_name, ("assistant",), 0
    ),
    "user_metadata_langgraph_node": Case(
        _user_langgraph_node_metadata, ("LangGraph",), 0
    ),
    "user_metadata_langgraph_node_with_config": Case(
        _user_langgraph_node_with_config, ("LangGraph",), 0
    ),
    "ordinary_sequence_in_tool": Case(
        _ordinary_sequence_in_tool, ("measured_agent",), 0
    ),
    "create_agent_internal_model_and_tools_nodes": Case(
        _create_agent_internal_nodes, ("internal_node_control",), 0
    ),
    "plain_langgraph": Case(_plain_langgraph, (), 1),
    "plain_langgraph_nested_in_agent_tool": Case(
        _plain_langgraph_in_tool, ("outer_agent",), 0
    ),
}


@pytest.mark.parametrize("case_name", CASES)
def test_agent_classification_corpus(case_name: str) -> None:
    case = CASES[case_name]
    recorder = CallbackRecorder()
    handler, telemetry = _handler()

    case.run(_callbacks(recorder, handler))

    if os.environ.get("LANGCHAIN_CORPUS_DUMP"):
        print(
            json.dumps(
                {"case": case_name, "events": recorder.events},
                indent=2,
                sort_keys=True,
            )
        )

    actual_agents = tuple(
        call.kwargs["agent_name"]
        for call in telemetry.invoke_local_agent.call_args_list
    )
    assert actual_agents == case.expected_agents
    assert telemetry.workflow.call_count == case.expected_workflows
