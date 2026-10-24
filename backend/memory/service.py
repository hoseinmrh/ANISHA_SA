"""Deterministic contextual-memory loading and retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import (
    ContextBundle,
    MemoryConflict,
    MemoryConflictState,
    MemoryQuery,
    MemoryRecord,
    MemoryStatus,
    ProjectBrief,
    Sensitivity,
    SourceAuthority,
)
from .store import InMemoryContextStore


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
AUTHORITY_WEIGHT = {
    SourceAuthority.USER_CONFIRMED: 1.0,
    SourceAuthority.WORKSPACE_RECORD: 0.9,
    SourceAuthority.AGENT_REPORT: 0.75,
    SourceAuthority.EXTERNAL_SOURCE: 0.5,
    SourceAuthority.GENERATED: 0.25,
}


class ContextMemoryService:
    """Return bounded context without exposing the backing store to agents."""

    def __init__(self, store: InMemoryContextStore) -> None:
        self.store = store

    def load_seed_file(self, path: Path) -> ProjectBrief:
        payload = json.loads(path.read_text(encoding="utf-8"))
        brief = ProjectBrief.model_validate(payload["brief"])
        memories = [
            MemoryRecord.model_validate(value)
            for value in payload.get("memories", [])
        ]
        conflicts = [
            MemoryConflict.model_validate(value)
            for value in payload.get("conflicts", [])
        ]
        self.store.seed_project(brief, memories, conflicts)
        return brief.model_copy(deep=True)

    def retrieve(self, query: MemoryQuery) -> ContextBundle:
        brief = self.store.get_brief(query.project_id)
        candidates = [
            memory
            for memory in self.store.list_memories(query.project_id)
            if self._is_visible(memory, query)
        ]
        ranked = sorted(
            candidates,
            key=lambda memory: self._ranking_key(memory, brief, query),
        )

        selected: list[MemoryRecord] = []
        used_characters = 0
        omitted_for_budget = 0
        for memory in ranked:
            if len(selected) >= query.limit:
                break
            memory_size = len(memory.model_dump_json())
            if used_characters + memory_size > query.max_characters:
                omitted_for_budget += 1
                continue
            selected.append(memory)
            used_characters += memory_size

        selected_ids = {memory.memory_id for memory in selected}
        conflicts = [
            conflict
            for conflict in self.store.list_conflicts(query.project_id)
            if conflict.state == MemoryConflictState.OPEN
            and selected_ids.intersection(conflict.memory_ids)
        ]
        warnings: list[str] = []
        if not selected:
            warnings.append("No reliable project memory matched this request.")
        omitted_for_limit = max(
            0,
            len(ranked) - len(selected) - omitted_for_budget,
        )
        if omitted_for_limit:
            warnings.append(
                f"Returned the top {len(selected)} memory record(s); "
                f"{omitted_for_limit} lower-ranked record(s) were omitted."
            )
        if omitted_for_budget:
            warnings.append(
                f"Omitted {omitted_for_budget} memory record(s) to respect the "
                "context character budget."
            )
        if conflicts:
            warnings.append(
                f"The selected context contains {len(conflicts)} unresolved "
                "memory conflict(s)."
            )

        return ContextBundle(
            query=query,
            brief=brief,
            memories=selected,
            conflicts=conflicts,
            warnings=warnings,
            truncated=(omitted_for_budget > 0 or omitted_for_limit > 0),
        )

    @staticmethod
    def _is_visible(memory: MemoryRecord, query: MemoryQuery) -> bool:
        if memory.status != MemoryStatus.ACTIVE:
            return False
        if query.kinds and memory.kind not in query.kinds:
            return False
        if query.agent_id is not None and memory.allowed_agents:
            if query.agent_id not in memory.allowed_agents:
                return False
        if query.agent_id is not None and memory.sensitivity != Sensitivity.PROJECT:
            return query.agent_id in memory.allowed_agents
        return True

    @classmethod
    def _ranking_key(
        cls,
        memory: MemoryRecord,
        brief: ProjectBrief,
        query: MemoryQuery,
    ) -> tuple[float, float, str]:
        score = cls._score(memory, brief, query)
        return (-score, -memory.captured_at.timestamp(), memory.memory_id)

    @classmethod
    def _score(
        cls,
        memory: MemoryRecord,
        brief: ProjectBrief,
        query: MemoryQuery,
    ) -> float:
        query_terms = cls._tokens(query.query)
        memory_terms = cls._tokens(
            " ".join(
                [memory.content, memory.kind.value, *memory.tags]
            )
        )
        if query_terms:
            relevance = len(query_terms.intersection(memory_terms)) / len(query_terms)
        else:
            relevance = 1.0 if memory.memory_id in brief.memory_refs else 0.0

        authority = max(
            AUTHORITY_WEIGHT[source.authority] for source in memory.source_refs
        )
        role_match = 1.0
        if query.agent_id is not None and not memory.allowed_agents:
            role_match = 0.5
        current_validity = 1.0
        return (
            0.40 * relevance
            + 0.25 * authority
            + 0.15 * current_validity
            + 0.10 * memory.importance
            + 0.10 * role_match
        )

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return set(TOKEN_PATTERN.findall(value.casefold()))
