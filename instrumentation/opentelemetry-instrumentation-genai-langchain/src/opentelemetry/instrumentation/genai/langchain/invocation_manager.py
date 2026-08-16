# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from uuid import UUID

from opentelemetry.util.genai.types import GenAIInvocation

__all__ = ["_InvocationManager"]


@dataclass
class _InvocationState:
    invocation: GenAIInvocation | None
    has_create_agent_marker: bool = False
    children: list[UUID] = field(default_factory=lambda: list())
    parent_run_id: UUID | None = None
    ended: bool = False


class _InvocationManager:
    def __init__(
        self,
    ) -> None:
        # Map from run_id -> _InvocationState, to keep track of invocations and parent/child relationships
        # TODO: TTL cache to avoid memory leaks in long-running processes.
        self._invocations: dict[UUID, _InvocationState] = {}

    def add_invocation_state(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        invocation: GenAIInvocation | None,
        has_create_agent_marker: bool = False,
    ) -> None:
        invocation_state = _InvocationState(
            invocation=invocation,
            has_create_agent_marker=has_create_agent_marker,
        )

        if parent_run_id is not None and parent_run_id in self._invocations:
            invocation_state.parent_run_id = parent_run_id

            parent_invocation_state = self._invocations[parent_run_id]
            parent_invocation_state.children.append(run_id)

        self._invocations[run_id] = invocation_state

    def get_invocation(self, run_id: UUID) -> GenAIInvocation | None:
        invocation_state = self._invocations.get(run_id)
        return invocation_state.invocation if invocation_state else None

    def get_parent_run_id(self, run_id: UUID) -> UUID | None:
        invocation_state = self._invocations.get(run_id)
        return invocation_state.parent_run_id if invocation_state else None

    def has_create_agent_ancestor(self, run_id: UUID | None) -> bool:
        current: UUID | None = run_id
        visited: set[UUID] = set()
        while current is not None:
            if current in visited:
                return False
            visited.add(current)
            invocation_state = self._invocations.get(current)
            if invocation_state is None:
                return False
            if invocation_state.has_create_agent_marker:
                return True
            current = invocation_state.parent_run_id
        return False

    def delete_invocation_state(self, run_id: UUID) -> None:
        invocation_state = self._invocations.get(run_id)
        if not invocation_state:
            return

        invocation_state.ended = True

        # Defer removal if any children are still live, so upward traversal
        # (e.g. _find_nearest_agent) can still walk through this node.
        if any(c in self._invocations for c in invocation_state.children):
            return

        self._invocations.pop(run_id, None)

        # Propagate cleanup upward: if the parent has already ended and has no
        # more live children, it can now be removed too.
        if invocation_state.parent_run_id:
            parent_state = self._invocations.get(
                invocation_state.parent_run_id
            )
            if parent_state is not None and parent_state.ended:
                self.delete_invocation_state(invocation_state.parent_run_id)
