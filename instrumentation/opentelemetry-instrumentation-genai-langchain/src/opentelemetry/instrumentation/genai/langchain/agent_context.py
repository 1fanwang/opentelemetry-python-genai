# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Identify ``create_agent`` graph invocations from outside the callback API.

LangChain callbacks cannot tell a nested ``create_agent`` root apart from any
other named runnable invoked inside a tool: the enclosing agent's ``config`` is
merged over the inner agent's own, so ``lc_agent_name`` and ``ls_integration``
in the *callback* metadata describe the outer agent. The compiled graph's own
bound config is never shadowed, so this module reads the marker there and has
each graph entry point announce itself on a context stack that the callback
handler consults when the root run starts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry.instrumentation.genai.langchain.operation_mapping import (
    create_agent_graph_name,
)


_AGENT_NAME_ATTR = "_otel_genai_agent_name"


@dataclass
class _PendingAgent:
    """An agent graph that has started running but whose root run is not seen yet."""

    name: str
    claimed: bool = False


_pending: ContextVar[tuple[_PendingAgent, ...]] = ContextVar(
    "otel_genai_pending_agents", default=()
)


def claim_agent() -> str | None:
    """Return the announced agent name if this run is a create_agent graph root.

    The announcement is made as the graph starts, so the first chain run to see
    it is the graph's root. Only the innermost announcement is claimable, and
    only once, so internal nodes fall through to metadata-based classification.
    The root run's own name is not checked - ``with_config(run_name=...)``
    renames it without making it any less of an agent.
    """
    pending = _pending.get()
    if not pending:
        return None
    entry = pending[-1]
    if entry.claimed:
        return None
    entry.claimed = True
    return entry.name


def _agent_name(graph: Any) -> str | None:
    """Return the agent name a compiled graph identifies itself with, if any."""
    marked = getattr(graph, _AGENT_NAME_ATTR, None)
    if marked:
        return str(marked)
    return create_agent_graph_name(getattr(graph, "config", None))


def wrap_create_react_agent(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Tag a ``create_react_agent`` graph so it announces itself like create_agent.

    The deprecated LangGraph builder compiles its own graph and leaves ``config``
    unset, so it carries none of the markers ``create_agent`` bakes in. Marking
    the object keeps it out of the graph's runtime config, which would otherwise
    show up in the user's own run metadata.
    """
    graph = wrapped(*args, **kwargs)
    name = kwargs.get("name") or getattr(graph, "name", None)
    if name:
        try:
            object.__setattr__(graph, _AGENT_NAME_ATTR, str(name))
        except (AttributeError, TypeError):
            pass
    return graph


def _push(name: str) -> _PendingAgent:
    """Announce ``name`` as the innermost running agent."""
    entry = _PendingAgent(name)
    _pending.set(_pending.get() + (entry,))
    return entry


def _pop(entry: _PendingAgent) -> None:
    """Withdraw ``entry``, tolerating a stack the caller's context no longer owns."""
    pending = _pending.get()
    if pending and pending[-1] is entry:
        _pending.set(pending[:-1])


def wrap_stream(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Announce an agent graph for the duration of ``Pregel.stream``.

    ``Pregel.invoke`` runs through ``stream``, so this covers both entry points.
    """
    name = _agent_name(instance)
    if name is None:
        return wrapped(*args, **kwargs)
    return _announce_at_stream_start(wrapped(*args, **kwargs), name)


def wrap_astream(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Announce an agent graph for the duration of ``Pregel.astream``.

    ``Pregel.ainvoke`` runs through ``astream``, so this covers both entry points.
    """
    name = _agent_name(instance)
    if name is None:
        return wrapped(*args, **kwargs)
    return _announce_at_astream_start(wrapped(*args, **kwargs), name)


def _announce_at_stream_start(
    stream: Iterator[Any], name: str
) -> Iterator[Any]:
    """Announce ``name`` for the first step of ``stream`` only.

    The graph opens its root run before it can produce anything, so the
    announcement has served its purpose once the first item is in hand.
    Withdrawing it there keeps an abandoned stream from leaving the
    announcement standing until the generator is garbage collected - by which
    point ``_pop`` would run in an unrelated context and quietly do nothing.
    """
    # A generator body runs in the consumer's context, so the announcement lands
    # where the callbacks fire - on the first ``next()``, not here.
    iterator = iter(stream)
    entry = _push(name)
    try:
        first = next(iterator)
    except StopIteration:
        return
    finally:
        _pop(entry)
    yield first
    yield from iterator


async def _announce_at_astream_start(
    stream: AsyncIterator[Any], name: str
) -> AsyncIterator[Any]:
    """Announce ``name`` for the first step of ``stream`` only."""
    iterator = stream.__aiter__()
    entry = _push(name)
    try:
        first = await iterator.__anext__()
    except StopAsyncIteration:
        return
    finally:
        _pop(entry)
    yield first
    async for chunk in iterator:
        yield chunk
