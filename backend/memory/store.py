"""In-memory persistence boundary for contextual-memory records."""

from __future__ import annotations

from .models import MemoryConflict, MemoryRecord, ProjectBrief


class DuplicateMemoryRecordError(ValueError):
    pass


class UnknownMemoryRecordError(KeyError):
    pass


class UnknownProjectContextError(KeyError):
    pass


class InMemoryContextStore:
    """Store defensive copies of project briefs, memories, and conflicts."""

    def __init__(self) -> None:
        self._briefs: dict[str, ProjectBrief] = {}
        self._memories: dict[tuple[str, str], MemoryRecord] = {}
        self._conflicts: dict[tuple[str, str], MemoryConflict] = {}

    def seed_project(
        self,
        brief: ProjectBrief,
        memories: list[MemoryRecord],
        conflicts: list[MemoryConflict] | None = None,
    ) -> None:
        """Validate and add one project atomically."""

        project_id = brief.project_id
        if project_id in self._briefs:
            raise DuplicateMemoryRecordError(project_id)

        memory_ids = [memory.memory_id for memory in memories]
        if len(set(memory_ids)) != len(memory_ids):
            raise DuplicateMemoryRecordError("duplicate memory ID in project seed")
        if any(memory.project_id != project_id for memory in memories):
            raise ValueError("all seeded memories must match the brief project_id")
        missing_refs = set(brief.memory_refs) - set(memory_ids)
        if missing_refs:
            raise ValueError(
                "project brief references missing seeded memories: "
                + ", ".join(sorted(missing_refs))
            )

        seeded_conflicts = list(conflicts or [])
        conflict_ids = [conflict.conflict_id for conflict in seeded_conflicts]
        if len(set(conflict_ids)) != len(conflict_ids):
            raise DuplicateMemoryRecordError("duplicate conflict ID in project seed")
        if any(conflict.project_id != project_id for conflict in seeded_conflicts):
            raise ValueError("all seeded conflicts must match the brief project_id")
        unknown_conflict_memories = {
            memory_id
            for conflict in seeded_conflicts
            for memory_id in conflict.memory_ids
            if memory_id not in memory_ids
        }
        if unknown_conflict_memories:
            raise ValueError(
                "memory conflict references missing seeded memories: "
                + ", ".join(sorted(unknown_conflict_memories))
            )

        self._briefs[project_id] = brief.model_copy(deep=True)
        for memory in memories:
            self._memories[(project_id, memory.memory_id)] = memory.model_copy(
                deep=True
            )
        for conflict in seeded_conflicts:
            self._conflicts[(project_id, conflict.conflict_id)] = (
                conflict.model_copy(deep=True)
            )

    def get_brief(self, project_id: str) -> ProjectBrief:
        brief = self._briefs.get(project_id)
        if brief is None:
            raise UnknownProjectContextError(project_id)
        return brief.model_copy(deep=True)

    def get_memory(self, project_id: str, memory_id: str) -> MemoryRecord:
        memory = self._memories.get((project_id, memory_id))
        if memory is None:
            raise UnknownMemoryRecordError(memory_id)
        return memory.model_copy(deep=True)

    def list_memories(self, project_id: str) -> list[MemoryRecord]:
        self.get_brief(project_id)
        return [
            memory.model_copy(deep=True)
            for (stored_project_id, _), memory in self._memories.items()
            if stored_project_id == project_id
        ]

    def list_conflicts(self, project_id: str) -> list[MemoryConflict]:
        self.get_brief(project_id)
        return [
            conflict.model_copy(deep=True)
            for (stored_project_id, _), conflict in self._conflicts.items()
            if stored_project_id == project_id
        ]

    def add_memory_and_update_brief(
        self, memory: MemoryRecord, brief: ProjectBrief
    ) -> None:
        """Atomically add one derived record and replace its project brief."""

        project_id = brief.project_id
        self.get_brief(project_id)
        if memory.project_id != project_id:
            raise ValueError("memory and brief must belong to the same project")
        key = (project_id, memory.memory_id)
        if key in self._memories:
            raise DuplicateMemoryRecordError(memory.memory_id)
        known_ids = {
            stored_memory_id
            for (stored_project_id, stored_memory_id) in self._memories
            if stored_project_id == project_id
        }
        known_ids.add(memory.memory_id)
        missing_refs = set(brief.memory_refs) - known_ids
        if missing_refs:
            raise ValueError(
                "project brief references missing memories: "
                + ", ".join(sorted(missing_refs))
            )

        self._memories[key] = memory.model_copy(deep=True)
        self._briefs[project_id] = brief.model_copy(deep=True)
