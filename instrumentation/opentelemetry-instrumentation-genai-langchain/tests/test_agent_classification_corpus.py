# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Focused regression corpus for LangChain create_agent classification."""

from __future__ import annotations

from typing import Any, Self
from unittest import mock
from uuid import uuid4

import langchain.agents
import pytest

create_agent = getattr(langchain.agents, "create_agent", None)
if create_agent is None:
    pytest.skip(
        "create_agent requires a newer langchain version",
        allow_module_level=True,
    )

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

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


def _agent_names(telemetry: mock.MagicMock) -> list[str]:
    return [
        call.kwargs["agent_name"]
        for call in telemetry.invoke_local_agent.call_args_list
    ]


@pytest.mark.parametrize(
    ("name", "expected_name"),
    [(None, "LangGraph"), ("named_agent", "named_agent")],
)
def test_create_agent_root(name: str | None, expected_name: str) -> None:
    handler, telemetry = _handler()
    agent_kwargs = {"name": name} if name is not None else {}
    create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        **agent_kwargs,
    ).invoke({"messages": [("user", "hi")]}, {"callbacks": [handler]})

    assert _agent_names(telemetry) == [expected_name]


def test_create_agent_name_wins_over_run_name_override() -> None:
    handler, telemetry = _handler()
    agent = create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        name="planner_agent",
    )
    RunnableLambda(
        lambda _: agent.invoke(
            {"messages": [("user", "hi")]}, {"run_name": "step1"}
        )
    ).with_config(run_name="planner").invoke({}, {"callbacks": [handler]})

    assert _agent_names(telemetry) == ["planner_agent"]


def test_create_agent_internal_nodes_are_not_agents() -> None:
    handler, telemetry = _handler()
    create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "noop", "args": {}, "id": "1"}],
                ),
                AIMessage(content="done"),
            ]
        ),
        [noop],
        name="agent",
    ).invoke({"messages": [("user", "hi")]}, {"callbacks": [handler]})

    assert _agent_names(telemetry) == ["agent"]


def test_ordinary_runnable_is_not_an_agent() -> None:
    handler, telemetry = _handler()
    RunnableLambda(lambda value: value).with_config(
        run_name="ordinary"
    ).invoke("value", {"callbacks": [handler]})

    telemetry.invoke_local_agent.assert_not_called()


def test_plain_state_graph_is_not_an_agent() -> None:
    handler, telemetry = _handler()
    builder = StateGraph(dict[str, int])
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    builder.compile().invoke({"value": 1}, {"callbacks": [handler]})

    telemetry.invoke_local_agent.assert_not_called()


def test_incomplete_ancestry_does_not_claim_create_agent_root() -> None:
    handler, telemetry = _handler()
    outer_run_id = uuid4()
    missing_middle_run_id = uuid4()
    inner_run_id = uuid4()
    marker = {
        "ls_integration": "langchain_create_agent",
        "lc_agent_name": "agent",
    }

    handler.on_chain_start(
        {}, {}, run_id=outer_run_id, metadata=marker, name="outer"
    )
    handler.on_chain_start(
        {},
        {},
        run_id=inner_run_id,
        parent_run_id=missing_middle_run_id,
        metadata=marker,
        name="inner",
    )

    assert _agent_names(telemetry) == ["agent"]
    assert (
        handler._invocation_manager.get_parent_run_id(inner_run_id)
        == missing_middle_run_id
    )


def test_user_supplied_create_agent_marker_declares_agent() -> None:
    """The callback API has no provenance to distinguish this from create_agent."""
    handler, telemetry = _handler()
    RunnableLambda(lambda value: value).with_config(
        run_name="ordinary",
        metadata={"ls_integration": "langchain_create_agent"},
    ).invoke("value", {"callbacks": [handler]})

    assert _agent_names(telemetry) == ["ordinary"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LangChain callbacks do not identify a nested create_agent root "
        "separately from metadata inherited from its enclosing agent"
    ),
)
def test_nested_create_agent_is_known_limitation() -> None:
    handler, telemetry = _handler()
    inner = create_agent(
        FakeModel(responses=[AIMessage(content="inner done")]),
        [noop],
        name="inner",
    )

    @tool
    def delegate(config: RunnableConfig) -> str:
        """Delegate to the inner agent."""
        result = inner.invoke({"messages": [("user", "work")]}, config)
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
        name="outer",
    )
    outer.invoke({"messages": [("user", "start")]}, {"callbacks": [handler]})

    assert _agent_names(telemetry) == ["outer", "inner"]
