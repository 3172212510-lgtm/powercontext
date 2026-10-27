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

"""Durable evidence-dependency nodes for one Scope.

The graph is registered once and never rewritten, so a node record is a single
canonical payload: the immutable binding (kind, policy, parents, original
reference, lifecycle revision) plus the mutable lifecycle position (sequence,
propagated availability, native-application state).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, JsonValue
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.codec import dump_model, load_model
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.limits import MAX_SCOPE_ID_LENGTH

LifecycleNodeKind = Literal["source", "memory", "topic"]
LifecyclePolicy = Literal["all", "any"]


class StoredLifecycleNode(BaseModel):
    """One registered node and its current lifecycle position."""

    node_id: str
    kind: LifecycleNodeKind
    policy: LifecyclePolicy
    parents: tuple[str, ...] = ()
    reference: dict[str, JsonValue]
    revision: int
    sequence: int = 0
    available: bool = True
    applied: bool = True


class SourceLifecycleRepository:
    """Read and write the immutable graph plus its mutable state."""

    async def load_scope(self, connection: AsyncConnection, scope_id: str, /) -> tuple[StoredLifecycleNode, ...]:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        rows = (
            await connection.execute(
                select(table.c.node_id, table.c.payload)
                .where(table.c.scope_id == scope_id)
                .order_by(table.c.ordinal)
            )
        ).all()
        return tuple(
            load_model(StoredLifecycleNode, payload, kind="source-lifecycle", name=str(node_id))
            for node_id, payload in rows
        )

    async def next_ordinal(self, connection: AsyncConnection, scope_id: str, /) -> int:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        current = (
            await connection.execute(
                select(table.c.ordinal).where(table.c.scope_id == scope_id).order_by(table.c.ordinal.desc()).limit(1)
            )
        ).scalar()
        return 1 if current is None else int(current) + 1

    async def insert(self, connection: AsyncConnection, scope_id: str, node: StoredLifecycleNode, /) -> None:
        ordinal = await self.next_ordinal(connection, scope_id)
        await connection.execute(
            SOURCE_LIFECYCLE_NODES_TABLE.insert().values(
                scope_id=scope_id,
                node_id=node.node_id,
                payload=dump_model(node, kind="source-lifecycle", name=node.node_id),
                ordinal=ordinal,
            )
        )

    async def save_states(
        self,
        connection: AsyncConnection,
        scope_id: str,
        nodes: Mapping[str, StoredLifecycleNode],
        /,
    ) -> None:
        """Persist the mutable position of the given nodes in one transaction."""

        table = SOURCE_LIFECYCLE_NODES_TABLE
        for node_id, node in nodes.items():
            await connection.execute(
                update(table)
                .where(table.c.scope_id == scope_id, table.c.node_id == node_id)
                .values(payload=dump_model(node, kind="source-lifecycle", name=node_id))
            )


def require_scope_id(scope_id: object) -> str:
    if not isinstance(scope_id, str) or not scope_id.strip() or scope_id != scope_id.strip():
        raise ValueError("scope_id must be a non-empty trimmed identifier")  # noqa: TRY003
    if len(scope_id) > MAX_SCOPE_ID_LENGTH:
        raise ValueError("scope_id exceeds the identity length limit")  # noqa: TRY003
    return scope_id


__all__ = [
    "LifecycleNodeKind",
    "LifecyclePolicy",
    "SourceLifecycleRepository",
    "StoredLifecycleNode",
    "require_scope_id",
]
