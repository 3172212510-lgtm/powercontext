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

"""OceanBase Memory search indexes using FULLTEXT and vector indexes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pyobvector import VECTOR, VectorIndex
from sqlalchemy import (
    BigInteger,
    Column,
    MetaData,
    Table,
    TextClause,
    UniqueConstraint,
    bindparam,
    delete,
    insert,
    select,
    text,
)
from sqlalchemy.dialects.mysql import match
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import (
    CapabilityNotSupportedError,
    EmbeddingProfile,
    MemoryCapabilities,
    MemoryProjection,
    MemorySearchChannels,
    MemorySearchRequest,
)
from powercontext.builtin.artifacts.memory.canonical import embedding_content_hash, validate_embedding
from powercontext.builtin.persistence.memory_index import memory_channel_hits
from powercontext.builtin.persistence.tables import (
    ARTIFACT_HEADS_TABLE,
    MAX_MEMORY_ENTRY_ID_LENGTH,
    MAX_MEMORY_HASH_LENGTH,
    MEMORY_ENTRY_HEADS_TABLE,
    MEMORY_ENTRY_VERSIONS_TABLE,
    identity_string,
)
from powercontext.builtin.persistence.tags import memory_tag_parameters, memory_tag_sql, tag_predicate
from powercontext.limits import MAX_ARTIFACT_ID_LENGTH, MAX_SCOPE_ID_LENGTH

_OCEANBASE_FTS_INDEX_NAME = "ix_pc_memory_entry_heads_fts"
_OCEANBASE_CREATE_FTS_SQL = f"""
CREATE FULLTEXT INDEX {_OCEANBASE_FTS_INDEX_NAME}
ON pc_memory_entry_heads (searchable_text) WITH PARSER SPACE
"""
_OCEANBASE_FTS_INDEX_EXISTS_SQL = text(
    """
    SELECT COUNT(*)
    FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'pc_memory_entry_heads'
      AND index_name = :index_name
    """
)
_OCEANBASE_VECTOR_TABLE_NAME = "pc_memory_vector_entries"
_OCEANBASE_VECTOR_INDEX_NAME = "ix_pc_memory_vector_entries_embedding"
_OCEANBASE_VECTOR_TYPE_SQL = text(
    """
    SELECT data_type
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
      AND table_name = :table_name
      AND column_name = 'embedding'
    """
)
_OCEANBASE_VECTOR_SEARCH_SQL = """
SELECT
    m.memory_artifact_id,
    m.head_revision,
    m.entry_id,
    m.entry_version_id,
    v.text,
    l2_distance(m.embedding, :query_vector) AS distance
FROM pc_memory_vector_entries AS m
JOIN pc_memory_entry_versions AS v
  ON v.scope_id = m.scope_id
 AND v.memory_artifact_id = m.memory_artifact_id
 AND v.entry_version_id = m.entry_version_id
WHERE m.scope_id = :scope_id
  AND m.memory_artifact_id IN :memory_artifact_ids
ORDER BY l2_distance(m.embedding, :query_vector) APPROXIMATE
LIMIT :candidate_limit
"""
# One statement, two ranked candidate sets. Each branch keeps the exact shape of
# its single-channel query — same predicate, same ordering key, same LIMIT — so
# the FULLTEXT and vector index plans survive inside the union and the candidate
# sets stay identical to the two-query path. The branch columns are aligned by
# position: ``relevance`` is NULL for the semantic branch and ``distance`` is
# NULL for the lexical one, and the shared fusion ranks each channel on its own
# ordering column after this statement returns.
_OCEANBASE_HYBRID_SEARCH_SQL = """
SELECT
    channel,
    memory_artifact_id,
    head_revision,
    entry_id,
    entry_version_id,
    text,
    relevance,
    distance
FROM (
    (SELECT
        'fts' AS channel,
        h.memory_artifact_id AS memory_artifact_id,
        h.head_revision AS head_revision,
        h.entry_id AS entry_id,
        h.entry_version_id AS entry_version_id,
        v.text AS text,
        MATCH (h.searchable_text) AGAINST (:analyzed_query) AS relevance,
        NULL AS distance
    FROM pc_memory_entry_heads AS h
    JOIN pc_memory_entry_versions AS v
      ON v.scope_id = h.scope_id
     AND v.memory_artifact_id = h.memory_artifact_id
     AND v.entry_version_id = h.entry_version_id
    WHERE h.scope_id = :scope_id
      AND h.memory_artifact_id IN :memory_artifact_ids
      AND MATCH (h.searchable_text) AGAINST (:analyzed_query)
      %FTS_TAG%
    ORDER BY MATCH (h.searchable_text) AGAINST (:analyzed_query) DESC,
             h.memory_artifact_id,
             h.entry_id,
             h.entry_version_id
    LIMIT :candidate_limit)
    UNION ALL
    (SELECT
        'vector' AS channel,
        m.memory_artifact_id AS memory_artifact_id,
        m.head_revision AS head_revision,
        m.entry_id AS entry_id,
        m.entry_version_id AS entry_version_id,
        v.text AS text,
        NULL AS relevance,
        l2_distance(m.embedding, :query_vector) AS distance
    FROM pc_memory_vector_entries AS m
    JOIN pc_memory_entry_versions AS v
      ON v.scope_id = m.scope_id
     AND v.memory_artifact_id = m.memory_artifact_id
     AND v.entry_version_id = m.entry_version_id
    WHERE m.scope_id = :scope_id
      AND m.memory_artifact_id IN :memory_artifact_ids
      %VECTOR_TAG%
    ORDER BY l2_distance(m.embedding, :query_vector) APPROXIMATE
    LIMIT :candidate_limit)
) AS candidates
"""


class OceanBaseMemoryFTSIndex:
    """OceanBase FULLTEXT strategy over the relational active-head projection."""

    capabilities = MemoryCapabilities(fts=True, tag_filter=True)
    tables: tuple[Table, ...] = ()

    async def initialize(self, connection: AsyncConnection, /) -> None:
        if connection.dialect.name != "mysql":
            raise CapabilityNotSupportedError("oceanbase-fts")
        count = await connection.scalar(
            _OCEANBASE_FTS_INDEX_EXISTS_SQL,
            {"index_name": _OCEANBASE_FTS_INDEX_NAME},
        )
        if count == 0:
            await connection.exec_driver_sql(_OCEANBASE_CREATE_FTS_SQL)
        probe = match(MEMORY_ENTRY_HEADS_TABLE.c.searchable_text, against="powercontext")
        await connection.execute(select(MEMORY_ENTRY_HEADS_TABLE.c.entry_version_id).where(probe).limit(1))

    async def replace(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        del connection, scope_id, memory_ref, projections

    async def delete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        entry_ids: tuple[str, ...],
        /,
    ) -> None:
        del connection, scope_id, memory_ref, entry_ids

    async def upsert(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        del connection, scope_id, memory_ref, projections

    async def search(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        /,
    ) -> MemorySearchChannels:
        if request.mode not in {"fts", "hybrid"}:
            return MemorySearchChannels()
        score = match(MEMORY_ENTRY_HEADS_TABLE.c.searchable_text, against=request.analyzed_query)
        rows = (
            await connection.execute(
                select(
                    MEMORY_ENTRY_HEADS_TABLE.c.memory_artifact_id,
                    MEMORY_ENTRY_HEADS_TABLE.c.head_revision,
                    MEMORY_ENTRY_HEADS_TABLE.c.entry_id,
                    MEMORY_ENTRY_HEADS_TABLE.c.entry_version_id,
                    MEMORY_ENTRY_VERSIONS_TABLE.c.text,
                )
                .join(
                    MEMORY_ENTRY_VERSIONS_TABLE,
                    (MEMORY_ENTRY_VERSIONS_TABLE.c.scope_id == MEMORY_ENTRY_HEADS_TABLE.c.scope_id)
                    & (
                        MEMORY_ENTRY_VERSIONS_TABLE.c.memory_artifact_id
                        == MEMORY_ENTRY_HEADS_TABLE.c.memory_artifact_id
                    )
                    & (MEMORY_ENTRY_VERSIONS_TABLE.c.entry_version_id == MEMORY_ENTRY_HEADS_TABLE.c.entry_version_id),
                )
                .where(
                    MEMORY_ENTRY_HEADS_TABLE.c.scope_id == scope_id,
                    MEMORY_ENTRY_HEADS_TABLE.c.memory_artifact_id.in_(
                        tuple(memory.artifact_id for memory in request.memories)
                    ),
                    score,
                    *(
                        ()
                        if request.tag_filter is None
                        else (
                            tag_predicate(
                                scope_id,
                                "memory",
                                MEMORY_ENTRY_HEADS_TABLE.c.memory_artifact_id,
                                "memory_entry",
                                MEMORY_ENTRY_HEADS_TABLE.c.entry_id,
                                request.tag_filter,
                            ),
                        )
                    ),
                )
                .order_by(
                    score.desc(),
                    MEMORY_ENTRY_HEADS_TABLE.c.memory_artifact_id,
                    MEMORY_ENTRY_HEADS_TABLE.c.entry_id,
                    MEMORY_ENTRY_HEADS_TABLE.c.entry_version_id,
                )
                .limit(request.candidate_limit)
            )
        ).mappings()
        return MemorySearchChannels(fts=memory_channel_hits(rows, request.memories))

    async def vector_complete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memories: tuple[ArtifactRef, ...],
        profile: EmbeddingProfile,
        /,
    ) -> bool:
        del connection, scope_id, memories, profile
        return False

    async def hydrate(
        self,
        connection: AsyncConnection,
        scope_id: str,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> tuple[MemoryProjection, ...]:
        del connection, scope_id
        return projections


class OceanBaseMemoryVectorIndex:
    """OceanBase HNSW strategy over rebuildable active-head embeddings."""

    def __init__(self, profile: EmbeddingProfile) -> None:
        if profile.dimension < 1 or profile.distance != "l2" or profile.normalization != "unit":
            raise CapabilityNotSupportedError(
                "vector",
                "OceanBase requires a positive unit-normalized L2 embedding profile",
            )
        self.profile = profile
        self.capabilities = MemoryCapabilities(
            tag_filter=True,
            fts=False,
            vector=True,
            embedding_profile=profile,
        )
        metadata = MetaData()
        self.table = Table(
            _OCEANBASE_VECTOR_TABLE_NAME,
            metadata,
            Column("vector_id", BigInteger, primary_key=True, autoincrement=True),
            Column("scope_id", identity_string(MAX_SCOPE_ID_LENGTH), nullable=False),
            Column("memory_artifact_id", identity_string(MAX_ARTIFACT_ID_LENGTH), nullable=False),
            Column("head_revision", BigInteger, nullable=False),
            Column("entry_id", identity_string(MAX_MEMORY_ENTRY_ID_LENGTH), nullable=False),
            Column("entry_version_id", identity_string(MAX_MEMORY_ENTRY_ID_LENGTH), nullable=False),
            Column("entry_content_hash", identity_string(MAX_MEMORY_HASH_LENGTH), nullable=False),
            Column("embedding_content_hash", identity_string(MAX_MEMORY_HASH_LENGTH), nullable=False),
            Column("embedding", VECTOR(profile.dimension), nullable=False),
            UniqueConstraint(
                "scope_id",
                "memory_artifact_id",
                "entry_id",
                name="uq_pc_memory_vector_entries_head",
            ),
        )
        self.tables: tuple[Table, ...] = (self.table,)
        self._vector_index = VectorIndex(
            _OCEANBASE_VECTOR_INDEX_NAME,
            self.table.c.embedding,
            params="distance=l2,type=hnsw",
        )
        self._search_sql = text(_OCEANBASE_VECTOR_SEARCH_SQL).bindparams(
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("query_vector", type_=VECTOR(profile.dimension)),
        )
        self._tagged_search_sql = text(
            _OCEANBASE_VECTOR_SEARCH_SQL.replace("ORDER BY", memory_tag_sql("m") + "ORDER BY").replace(
                " APPROXIMATE", ""
            )
        ).bindparams(
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("tag_keys", expanding=True),
            bindparam("tag_hashes", expanding=True),
            bindparam("query_vector", type_=VECTOR(profile.dimension)),
        )

    async def initialize(self, connection: AsyncConnection, /) -> None:
        if connection.dialect.name != "mysql":
            raise CapabilityNotSupportedError("oceanbase-vector")
        column_type = await connection.scalar(
            _OCEANBASE_VECTOR_TYPE_SQL,
            {"table_name": _OCEANBASE_VECTOR_TABLE_NAME},
        )
        expected = f"VECTOR({self.profile.dimension})"
        if str(column_type).upper() != expected:
            raise CapabilityNotSupportedError(
                "vector",
                f"OceanBase projection uses {column_type!r}; expected {expected}",
            )
        await connection.run_sync(lambda sync_connection: self._vector_index.create(sync_connection, checkfirst=True))

    async def replace(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        await connection.execute(
            delete(self.table).where(
                self.table.c.scope_id == scope_id,
                self.table.c.memory_artifact_id == memory_ref.artifact_id,
            )
        )
        await self.upsert(connection, scope_id, memory_ref, projections)

    async def delete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        entry_ids: tuple[str, ...],
        /,
    ) -> None:
        if not entry_ids:
            return
        await connection.execute(
            delete(self.table).where(
                self.table.c.scope_id == scope_id,
                self.table.c.memory_artifact_id == memory_ref.artifact_id,
                self.table.c.entry_id.in_(entry_ids),
            )
        )

    async def upsert(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        values = []
        for projection in projections:
            if projection.embedding is None or projection.embedding_content_hash is None:
                continue
            entry = projection.entry_version
            values.append({
                "scope_id": scope_id,
                "memory_artifact_id": memory_ref.artifact_id,
                "head_revision": memory_ref.revision,
                "entry_id": entry.entry_id,
                "entry_version_id": entry.entry_version_id,
                "entry_content_hash": entry.entry_content_hash,
                "embedding_content_hash": projection.embedding_content_hash,
                "embedding": validate_embedding(
                    projection.embedding,
                    dimension=self.profile.dimension,
                ),
            })
        if values:
            await connection.execute(insert(self.table), values)

    async def search(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        /,
    ) -> MemorySearchChannels:
        if request.mode not in {"vector", "hybrid"}:
            return MemorySearchChannels()
        if request.embedding_profile != self.profile or request.query_vector is None:
            raise CapabilityNotSupportedError("embedding-profile")
        if not await self.vector_complete(connection, scope_id, request.memories, self.profile):
            raise CapabilityNotSupportedError("vector")
        rows = (
            await connection.execute(
                self._search_sql if request.tag_filter is None else self._tagged_search_sql,
                {
                    "scope_id": scope_id,
                    "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
                    "query_vector": validate_embedding(
                        request.query_vector,
                        dimension=self.profile.dimension,
                    ),
                    "candidate_limit": request.candidate_limit,
                    **memory_tag_parameters(request.tag_filter),
                },
            )
        ).mappings()
        return MemorySearchChannels(vector=memory_channel_hits(rows, request.memories))

    async def vector_complete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memories: tuple[ArtifactRef, ...],
        profile: EmbeddingProfile,
        /,
    ) -> bool:
        if profile != self.profile:
            return False
        for memory in memories:
            current = await connection.scalar(
                select(ARTIFACT_HEADS_TABLE.c.revision).where(
                    ARTIFACT_HEADS_TABLE.c.scope_id == scope_id,
                    ARTIFACT_HEADS_TABLE.c.family == "memory",
                    ARTIFACT_HEADS_TABLE.c.artifact_id == memory.artifact_id,
                )
            )
            if current is None or int(current) != memory.revision:
                # The caller validated this head before the search. If it moved while
                # this query ran, report a stale head so the runtime can retry with it.
                raise CapabilityNotSupportedError("head")
            heads = (
                await connection.execute(
                    select(
                        MEMORY_ENTRY_HEADS_TABLE.c.entry_id,
                        MEMORY_ENTRY_HEADS_TABLE.c.entry_version_id,
                        MEMORY_ENTRY_HEADS_TABLE.c.entry_content_hash,
                    ).where(
                        MEMORY_ENTRY_HEADS_TABLE.c.scope_id == scope_id,
                        MEMORY_ENTRY_HEADS_TABLE.c.memory_artifact_id == memory.artifact_id,
                    )
                )
            ).all()
            metadata = (
                await connection.execute(
                    select(
                        self.table.c.entry_id,
                        self.table.c.entry_version_id,
                        self.table.c.entry_content_hash,
                        self.table.c.embedding_content_hash,
                    ).where(
                        self.table.c.scope_id == scope_id,
                        self.table.c.memory_artifact_id == memory.artifact_id,
                    )
                )
            ).all()
            expected = {(str(row[0]), str(row[1]), str(row[2])) for row in heads}
            actual = {(str(row[0]), str(row[1]), str(row[2])) for row in metadata}
            if actual != expected:
                return False
            if any(str(row[3]) != _embedding_hash(self.profile, str(row[2])) for row in metadata):
                return False
        return True

    async def hydrate(
        self,
        connection: AsyncConnection,
        scope_id: str,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> tuple[MemoryProjection, ...]:
        hydrated: list[MemoryProjection] = []
        for projection in projections:
            entry = projection.entry_version
            row = (
                await connection.execute(
                    select(
                        self.table.c.entry_content_hash,
                        self.table.c.embedding_content_hash,
                        self.table.c.embedding,
                    ).where(
                        self.table.c.scope_id == scope_id,
                        self.table.c.memory_artifact_id == entry.memory_artifact_id,
                        self.table.c.entry_id == entry.entry_id,
                        self.table.c.entry_version_id == entry.entry_version_id,
                    )
                )
            ).one_or_none()
            if (
                row is None
                or str(row[0]) != entry.entry_content_hash
                or str(row[1]) != _embedding_hash(self.profile, entry.entry_content_hash)
            ):
                hydrated.append(projection)
                continue
            embedding = validate_embedding(
                tuple(float(value) for value in row[2]),
                dimension=self.profile.dimension,
            )
            hydrated.append(
                MemoryProjection(
                    entry_version=entry,
                    searchable_text=projection.searchable_text,
                    embedding=embedding,
                    embedding_content_hash=str(row[1]),
                )
            )
        return tuple(hydrated)


class OceanBaseHybridMemoryIndex:
    """One native statement returning both FULLTEXT and vector candidates.

    The two single-channel indexes stay the owners of their own projections,
    schema and integrity checks; this index only replaces the *candidate query*
    of a ``hybrid`` search. It is a drop-in :class:`MemoryIndex`, so a deployment
    without an embedding profile keeps using the plain FULLTEXT index and a
    deployment with one keeps the same capabilities as the former composite.
    """

    def __init__(self, fts: OceanBaseMemoryFTSIndex, vector: OceanBaseMemoryVectorIndex) -> None:
        self._fts = fts
        self._vector = vector
        self.profile = vector.profile
        self.capabilities = MemoryCapabilities(
            fts=True,
            vector=True,
            hybrid=True,
            tag_filter=True,
            embedding_profile=vector.profile,
        )
        self.tables = fts.tables + vector.tables
        self._statement = _hybrid_statement(tagged=False, dimension=vector.profile.dimension)
        self._tagged_statement = _hybrid_statement(tagged=True, dimension=vector.profile.dimension)

    async def initialize(self, connection: AsyncConnection, /) -> None:
        await self._fts.initialize(connection)
        await self._vector.initialize(connection)

    async def replace(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        await self._fts.replace(connection, scope_id, memory_ref, projections)
        await self._vector.replace(connection, scope_id, memory_ref, projections)

    async def delete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        entry_ids: tuple[str, ...],
        /,
    ) -> None:
        await self._fts.delete(connection, scope_id, memory_ref, entry_ids)
        await self._vector.delete(connection, scope_id, memory_ref, entry_ids)

    async def upsert(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        await self._fts.upsert(connection, scope_id, memory_ref, projections)
        await self._vector.upsert(connection, scope_id, memory_ref, projections)

    async def search(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        /,
    ) -> MemorySearchChannels:
        if request.mode == "fts":
            return await self._fts.search(connection, scope_id, request)
        if request.mode == "vector":
            return await self._vector.search(connection, scope_id, request)
        if request.embedding_profile != self.profile or request.query_vector is None:
            raise CapabilityNotSupportedError("embedding-profile")
        rows = list(
            (
                await connection.execute(
                    self._tagged_statement if request.tag_filter is not None else self._statement,
                    {
                        "scope_id": scope_id,
                        "analyzed_query": request.analyzed_query,
                        "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
                        "query_vector": validate_embedding(
                            request.query_vector,
                            dimension=self.profile.dimension,
                        ),
                        "candidate_limit": request.candidate_limit,
                        **memory_tag_parameters(request.tag_filter),
                    },
                )
            ).mappings()
        )
        return MemorySearchChannels(
            fts=memory_channel_hits(_lexical_ranking(rows), request.memories),
            vector=memory_channel_hits(_semantic_ranking(rows), request.memories),
        )

    async def vector_complete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memories: tuple[ArtifactRef, ...],
        profile: EmbeddingProfile,
        /,
    ) -> bool:
        return await self._vector.vector_complete(connection, scope_id, memories, profile)

    async def hydrate(
        self,
        connection: AsyncConnection,
        scope_id: str,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> tuple[MemoryProjection, ...]:
        hydrated = await self._fts.hydrate(connection, scope_id, projections)
        return await self._vector.hydrate(connection, scope_id, hydrated)


def _hybrid_statement(*, tagged: bool, dimension: int) -> TextClause:
    """Bind the shared hybrid template, optionally with both tag predicates."""

    statement = _OCEANBASE_HYBRID_SEARCH_SQL
    if tagged:
        statement = statement.replace("%FTS_TAG%", memory_tag_sql("h")).replace("%VECTOR_TAG%", memory_tag_sql("m"))
    else:
        statement = statement.replace("%FTS_TAG%", "").replace("%VECTOR_TAG%", "")
    compiled = text(statement).bindparams(
        bindparam("memory_artifact_ids", expanding=True),
        bindparam("query_vector", type_=VECTOR(dimension)),
    )
    if tagged:
        compiled = compiled.bindparams(
            bindparam("tag_keys", expanding=True),
            bindparam("tag_hashes", expanding=True),
        )
    return compiled


def _lexical_ranking(rows: Iterable[Mapping[Any, Any]]) -> list[Mapping[Any, Any]]:
    """FULLTEXT candidates in descending relevance with an identity tie-break."""

    return _distinct(
        sorted(
            (row for row in rows if str(row["channel"]) == "fts"),
            key=lambda row: (-float(row["relevance"]), *_identity(row)),
        )
    )


def _semantic_ranking(rows: Iterable[Mapping[Any, Any]]) -> list[Mapping[Any, Any]]:
    """Vector candidates in ascending L2 distance with an identity tie-break."""

    return _distinct(
        sorted(
            (row for row in rows if str(row["channel"]) == "vector"),
            key=lambda row: (float(row["distance"]), *_identity(row)),
        )
    )


def _identity(row: Mapping[Any, Any]) -> tuple[str, str, str]:
    return (str(row["memory_artifact_id"]), str(row["entry_id"]), str(row["entry_version_id"]))


def _distinct(rows: list[Mapping[Any, Any]]) -> list[Mapping[Any, Any]]:
    """Keep the first occurrence of each identity so a channel never repeats one."""

    seen: set[tuple[str, str, str]] = set()
    unique: list[Mapping[Any, Any]] = []
    for row in rows:
        identity = _identity(row)
        if identity not in seen:
            seen.add(identity)
            unique.append(row)
    return unique


def _embedding_hash(profile: EmbeddingProfile, entry_hash: str) -> str:
    return embedding_content_hash(
        profile_id=profile.profile_id,
        model=profile.model,
        dimension=profile.dimension,
        distance=profile.distance,
        normalization=profile.normalization,
        entry_content_hash=entry_hash,
    )
