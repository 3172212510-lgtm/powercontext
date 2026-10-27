# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Propagate Source withdrawal and restoration through an evidence DAG.

Callers register the evidence graph they already know: Sources, the Memory
entries and Topic Memory artifacts derived from them, and the ``all``/``any``
policy that makes each derived body usable. Availability is then a pure
function of the Source states, so withdrawal and restoration both propagate
without rewriting any body. Deactivation only changes usage eligibility; the
original references stay readable for audit.

Native application is a separate, retryable step: the propagated state is
committed first and each Memory entry carries an ``applied`` flag until its
own artifact revision reflects that state. A failed write therefore leaves a
resumable obligation instead of a silently inconsistent Memory, and re-sending
the same event completes it without adding revisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from powercontext.artifacts import ArtifactRef
from powercontext.artifacts.models import MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.source_lifecycle import (
    LifecycleNodeKind,
    LifecyclePolicy,
    SourceLifecycleRepository,
    StoredLifecycleNode,
)
from powercontext.limits import MAX_LIFECYCLE_NODE_ID_LENGTH
from powercontext.sources import SourceRef

TOPIC_MEMORY_FAMILY = "topic-memory"
GENERATION_PURPOSE = "generation"
AUDIT_PURPOSE = "audit"
_PURPOSES = frozenset({GENERATION_PURPOSE, AUDIT_PURPOSE})


class SourceLifecycleError(ValueError):
    """Report an invalid lifecycle registration, event, or resolution."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


class SourceLifecycleEvent(BaseModel):
    """One caller-supplied Source availability change."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: str
    revision: int
    sequence: int
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_visible_text(cls, value: str, info: Any) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")  # noqa: TRY003
        if info.field_name == "node_id" and len(value) > MAX_LIFECYCLE_NODE_ID_LENGTH:
            raise ValueError("node_id exceeds the identity length limit")  # noqa: TRY003
        return value

    @field_validator("revision", "sequence")
    @classmethod
    def require_positive_integer(cls, value: int, info: Any) -> int:
        if value < 1:
            raise ValueError(f"{info.field_name} must be a positive integer")  # noqa: TRY003
        return value


def _validate_node_id(node_id: object) -> str:
    if not isinstance(node_id, str):
        raise SourceLifecycleError("node-id", "must be a string")
    if not node_id.strip() or node_id != node_id.strip():
        raise SourceLifecycleError("node-id", "must be non-empty and trimmed")
    if len(node_id) > MAX_LIFECYCLE_NODE_ID_LENGTH:
        raise SourceLifecycleError("node-id", "exceeds the identity length limit")
    return node_id


def _validate_policy(policy: object) -> LifecyclePolicy:
    if policy not in {"all", "any"}:
        raise SourceLifecycleError("policy", "must be 'all' or 'any'")
    return policy  # type: ignore[return-value]


def _validate_parents(node_id: str, parents: object, registered: Mapping[str, StoredLifecycleNode]) -> tuple[str, ...]:
    if isinstance(parents, str) or not isinstance(parents, Sequence):
        raise SourceLifecycleError("parents", "must be a sequence of node identifiers")
    ordered: list[str] = []
    for parent in parents:
        identifier = _validate_node_id(parent)
        if identifier == node_id:
            raise SourceLifecycleError("cycle", "a node cannot depend on itself")
        if identifier in ordered:
            raise SourceLifecycleError("duplicate-parent", identifier)
        if identifier not in registered:
            raise SourceLifecycleError("missing-parent", identifier)
        ordered.append(identifier)
    if not ordered:
        raise SourceLifecycleError("parents", "a derived node requires at least one parent")
    return tuple(ordered)


def _encode_reference(node_id: str, kind: LifecycleNodeKind, reference: object) -> dict[str, Any]:
    if kind == "source":
        if not isinstance(reference, SourceRef):
            raise SourceLifecycleError("reference", f"{node_id} requires a SourceRef")
        return {"source_type": reference.source_type, "source_id": reference.source_id}
    if kind == "memory":
        if not isinstance(reference, MemoryCitation):
            raise SourceLifecycleError("reference", f"{node_id} requires a MemoryCitation")
        return {
            "memory_ref": reference.memory_ref.model_dump(mode="json"),
            "entry_id": reference.entry_id,
            "entry_version_id": reference.entry_version_id,
        }
    if not isinstance(reference, ArtifactRef):
        raise SourceLifecycleError("reference", f"{node_id} requires an ArtifactRef")
    return reference.model_dump(mode="json")


def _decode_reference(node: StoredLifecycleNode) -> SourceRef | MemoryCitation | ArtifactRef:
    value = node.reference
    if node.kind == "source":
        return SourceRef(source_type=value["source_type"], source_id=value["source_id"])
    if node.kind == "memory":
        return MemoryCitation(
            memory_ref=ArtifactRef.model_validate(value["memory_ref"]),
            entry_id=value["entry_id"],
            entry_version_id=value["entry_version_id"],
        )
    return ArtifactRef.model_validate(value)


def _propagate(
    nodes: Sequence[StoredLifecycleNode],
    availability: Mapping[str, bool],
) -> dict[str, bool]:
    """Recompute every derived node in registration order (a topological order)."""

    resolved = dict(availability)
    for node in nodes:
        if node.kind == "source":
            continue
        support = [resolved[parent] for parent in node.parents]
        resolved[node.node_id] = all(support) if node.policy == "all" else any(support)
    return resolved


class ScopedSourceLifecycle:
    """Register one Scope's evidence graph and drive its availability."""

    def __init__(self, runtime: Any, database: AsyncDatabase, scope_id: str) -> None:
        self._runtime = runtime
        self._database = database
        self.scope_id = scope_id
        self._repository = SourceLifecycleRepository()

    # -- registration -----------------------------------------------------

    async def register_source(self, node_id: str, reference: SourceRef, /, *, revision: int) -> None:
        identifier = _validate_node_id(node_id)
        if not isinstance(reference, SourceRef):
            raise SourceLifecycleError("reference", f"{identifier} requires a SourceRef")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise SourceLifecycleError("revision", "must be a positive integer")
        await self._require_source(reference)
        await self._register(
            identifier,
            kind="source",
            policy="all",
            parents=(),
            reference=reference,
            revision=revision,
            available=True,
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: Sequence[str],
        policy: LifecyclePolicy = "all",
    ) -> None:
        identifier = _validate_node_id(node_id)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("reference", f"{identifier} requires a MemoryCitation")
        await self._require_citation(citation)
        await self._register_derived(
            identifier,
            kind="memory",
            policy=policy,
            parents=parents,
            reference=citation,
            revision=1,
        )

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        /,
        *,
        parents: Sequence[str],
        policy: LifecyclePolicy = "all",
    ) -> None:
        identifier = _validate_node_id(node_id)
        if not isinstance(reference, ArtifactRef) or reference.family != TOPIC_MEMORY_FAMILY:
            raise SourceLifecycleError("reference", f"{identifier} requires a Topic Memory ArtifactRef")
        await self._require_topic(reference)
        await self._register_derived(
            identifier,
            kind="topic",
            policy=policy,
            parents=parents,
            reference=reference,
            revision=1,
        )

    async def _register_derived(
        self,
        node_id: str,
        *,
        kind: LifecycleNodeKind,
        policy: LifecyclePolicy,
        parents: Sequence[str],
        reference: object,
        revision: int,
    ) -> None:
        validated_policy = _validate_policy(policy)
        async with self._database.transaction() as connection:
            registered = {node.node_id: node for node in await self._repository.load_scope(connection, self.scope_id)}
        ordered = _validate_parents(node_id, parents, registered)
        available = self._derive(registered, ordered, validated_policy)
        await self._register(
            node_id,
            kind=kind,
            policy=validated_policy,
            parents=ordered,
            reference=reference,
            revision=revision,
            available=available,
        )

    @staticmethod
    def _derive(
        registered: Mapping[str, StoredLifecycleNode],
        parents: Sequence[str],
        policy: LifecyclePolicy,
    ) -> bool:
        support = [registered[parent].available for parent in parents]
        return all(support) if policy == "all" else any(support)

    async def _register(
        self,
        node_id: str,
        *,
        kind: LifecycleNodeKind,
        policy: LifecyclePolicy,
        parents: Sequence[str],
        reference: object,
        revision: int,
        available: bool,
    ) -> None:
        encoded = _encode_reference(node_id, kind, reference)
        async with self._runtime._context(self.scope_id), self._runtime._locked(self.scope_id):
            async with self._database.transaction() as connection:
                current = next(
                    (node for node in await self._repository.load_scope(connection, self.scope_id) if node.node_id == node_id),
                    None,
                )
                if current is not None:
                    self._require_same_binding(current, kind, policy, parents, encoded, revision)
                    pending = None if current.applied or current.kind != "memory" else current
                else:
                    node = StoredLifecycleNode(
                        node_id=node_id,
                        kind=kind,
                        policy=policy,
                        parents=tuple(parents),
                        reference=encoded,
                        revision=revision,
                        sequence=0,
                        available=available,
                        applied=kind != "memory" or available,
                    )
                    await self._repository.insert(connection, self.scope_id, node)
                    pending = None if node.applied else node
            if pending is not None:
                await self._apply_native(pending)

    @staticmethod
    def _require_same_binding(
        current: StoredLifecycleNode,
        kind: LifecycleNodeKind,
        policy: LifecyclePolicy,
        parents: Sequence[str],
        reference: Mapping[str, Any],
        revision: int,
    ) -> None:
        if (
            current.kind != kind
            or current.policy != policy
            or current.parents != tuple(parents)
            or current.reference != reference
            or current.revision != revision
        ):
            raise SourceLifecycleError("conflict", f"{current.node_id} is already registered differently")

    # -- events -----------------------------------------------------------

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event", "must be a SourceLifecycleEvent")
        async with self._runtime._context(self.scope_id), self._runtime._locked(self.scope_id):
            async with self._database.transaction() as connection:
                nodes = await self._repository.load_scope(connection, self.scope_id)
                indexed = {node.node_id: node for node in nodes}
                source = indexed.get(event.node_id)
                if source is None or source.kind != "source":
                    raise SourceLifecycleError("unknown-source", event.node_id)
                if event.revision != source.revision:
                    raise SourceLifecycleError("revision", f"{event.node_id} expects revision {source.revision}")
                replay = self._event_position(source, event)
                availability = {node.node_id: node.available for node in nodes}
                if replay is None:
                    availability[source.node_id] = event.available
                propagated = _propagate(nodes, availability)
                updated: dict[str, StoredLifecycleNode] = {}
                for node in nodes:
                    available = propagated[node.node_id]
                    if node.kind == "source":
                        if node.node_id == source.node_id and replay is None:
                            updated[node.node_id] = node.model_copy(
                                update={"sequence": event.sequence, "available": available}
                            )
                        continue
                    # A Topic needs no native write, so only its recorded state
                    # matters. A Memory entry is applied once its own revision
                    # reflects this availability.
                    applied = True if node.kind == "topic" else node.applied and available == node.available
                    updated[node.node_id] = node.model_copy(update={"available": available, "applied": applied})
                await self._repository.save_states(connection, self.scope_id, updated)
                pending = tuple(
                    updated[node.node_id] for node in nodes if node.kind == "memory" and not updated[node.node_id].applied
                )
            for node in pending:
                await self._apply_native(node)
        return sorted(name for name, available in propagated.items() if available)

    @staticmethod
    def _event_position(source: StoredLifecycleNode, event: SourceLifecycleEvent) -> int | None:
        """Return None when the event advances, or the replayed position when it repeats."""

        if event.sequence == source.sequence:
            if event.available != source.available:
                raise SourceLifecycleError("sequence", f"{event.node_id} already recorded a different state")
            return source.sequence
        if event.sequence < source.sequence:
            raise SourceLifecycleError("sequence", f"{event.node_id} requires a sequence above {source.sequence}")
        return None

    # -- resolution -------------------------------------------------------

    async def resolve(self, node_id: str, /, *, purpose: str) -> SourceRef | MemoryCitation | ArtifactRef:
        identifier = _validate_node_id(node_id)
        if purpose not in _PURPOSES:
            raise SourceLifecycleError("purpose", "must be 'generation' or 'audit'")
        async with self._database.transaction() as connection:
            nodes = await self._repository.load_scope(connection, self.scope_id)
        node = next((item for item in nodes if item.node_id == identifier), None)
        if node is None:
            raise SourceLifecycleError("unknown-node", identifier)
        if purpose == GENERATION_PURPOSE and not node.available:
            raise SourceLifecycleError("unavailable", identifier)
        return _decode_reference(node)

    async def available_nodes(self) -> tuple[str, ...]:
        async with self._database.transaction() as connection:
            nodes = await self._repository.load_scope(connection, self.scope_id)
        return tuple(sorted(node.node_id for node in nodes if node.available))

    async def inactive_topic_refs(self) -> frozenset[tuple[str, int]]:
        async with self._database.transaction() as connection:
            nodes = await self._repository.load_scope(connection, self.scope_id)
        return frozenset(
            (str(node.reference["artifact_id"]), int(node.reference["revision"]))
            for node in nodes
            if node.kind == "topic" and not node.available
        )

    # -- native application ----------------------------------------------

    async def _apply_native(self, node: StoredLifecycleNode) -> None:
        """Make one Memory entry's native state match its propagated availability."""

        if node.kind != "memory":
            return
        citation = _decode_reference(node)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("reference", f"{node.node_id} is not a Memory citation")
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            current = await service.head(context.artifacts.memory_artifact_id)
            manifest = {item.entry_id: item for item in current.content.manifest.entries}
            item = manifest.get(citation.entry_id)
            if item is None:
                raise SourceLifecycleError("unknown-entry", citation.entry_id)
            target = "active" if node.available else "inactive"
            if item.state != target:
                versions = {version.entry_id: version for version in await service.entries(current)}
                version = versions.get(citation.entry_id)
                if version is None:
                    raise SourceLifecycleError("unknown-entry", citation.entry_id)
                reason = f"source lifecycle {node.node_id}"
                if node.available:
                    await service.reactivate(current, entries=(version,), reason=reason)
                else:
                    await service.forget(current, entries=(version,), reason=reason)
        await self._mark_applied(node.node_id)

    async def _mark_applied(self, node_id: str) -> None:
        async with self._database.transaction() as connection:
            nodes = await self._repository.load_scope(connection, self.scope_id)
            current = next((node for node in nodes if node.node_id == node_id), None)
            if current is None or current.applied:
                return
            await self._repository.save_states(
                connection, self.scope_id, {node_id: current.model_copy(update={"applied": True})}
            )

    # -- reference validation --------------------------------------------

    async def _require_source(self, reference: SourceRef) -> None:
        await self._runtime.records.for_scope(self.scope_id).get_source(
            reference.source_type, reference.source_id
        )

    async def _require_citation(self, citation: MemoryCitation) -> None:
        from powercontext.builtin.runtime.models import GetMemoryEntryRequest

        await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=citation))

    async def _require_topic(self, reference: ArtifactRef) -> None:
        await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
            TOPIC_MEMORY_FAMILY, reference.artifact_id, reference.revision
        )


class SourceLifecycleApplication:
    """Select the scoped lifecycle service."""

    def __init__(self, runtime: Any, database: AsyncDatabase) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, self._database, scope_id)


__all__ = [
    "AUDIT_PURPOSE",
    "GENERATION_PURPOSE",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "ScopedSourceLifecycle",
]
