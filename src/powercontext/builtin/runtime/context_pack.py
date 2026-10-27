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

"""Chunked Context Packs and the prefix-reuse delta protocol.

A full pack carries the prepared context as a deterministic sequence of chunks
plus a digest and its own wire size. A delta keeps a valid common prefix of the
base pack and appends the target suffix, so a receiver that already holds the
base pack can rebuild the target from serialized bytes alone.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

CONTEXT_PACK_SCHEMA: Final = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA: Final = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES: Final = 512

_PACK_FIELDS: Final = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS: Final = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """Raised when a Context Pack or delta breaks the wire contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_for(items: tuple[bytes, ...]) -> str:
    unsigned = {
        "schema": CONTEXT_PACK_SCHEMA,
        "items": [base64.b64encode(item).decode("ascii") for item in items],
    }
    return "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _decode_chunks(value: object, field: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{field} must be a list")
    chunks: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError(f"{field} entries must be base64 strings")
        try:
            chunks.append(base64.b64decode(entry.encode("ascii"), validate=True))
        except (binascii.Error, UnicodeEncodeError, ValueError) as error:
            raise ContextPackError(f"{field} entries must be standard base64") from error
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete, self-verifying sequence of context chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...], /) -> ContextPack:
        """Build a full pack from an ordered tuple of chunk bytes."""

        chunks = tuple(bytes(item) for item in items)
        digest = _digest_for(chunks)
        return cls(chunks, digest, _wire_size(chunks, digest))

    @classmethod
    def from_content(cls, content: str | bytes, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split text into deterministic, UTF-8 safe chunks and pack them."""

        return cls.build(_chunk_content(content, chunk_bytes))

    def to_bytes(self) -> bytes:
        return _canonical(
            {
                "schema": CONTEXT_PACK_SCHEMA,
                "items": [base64.b64encode(item).decode("ascii") for item in self.items],
                "digest": self.digest,
                "byte_count": self.byte_count,
            }
        )

    @classmethod
    def from_bytes(cls, raw: bytes, /) -> ContextPack:
        payload = _load_object(raw, _PACK_FIELDS, "context pack")
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("unexpected context pack schema")
        items = _decode_chunks(payload["items"], "items")
        digest = payload["digest"]
        if not isinstance(digest, str):
            raise ContextPackError("digest must be a string")
        if digest != _digest_for(items):
            raise ContextPackError("context pack digest does not match its items")
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count != len(raw):
            raise ContextPackError("byte_count must equal the serialized length")
        return cls(items, digest, byte_count)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A base-relative update carrying only the target suffix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Choose the longest common prefix of two packs and carry the rest."""

        prefix = 0
        for left, right in zip(base.items, target.items):
            if left != right:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _canonical(
            {
                "schema": CONTEXT_PACK_DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": [base64.b64encode(item).decode("ascii") for item in self.appended],
            }
        )

    @classmethod
    def from_bytes(cls, raw: bytes, /) -> ContextPackDelta:
        payload = _load_object(raw, _DELTA_FIELDS, "context pack delta")
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("unexpected context pack delta schema")
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if not isinstance(base_digest, str) or not isinstance(target_digest, str):
            raise ContextPackError("delta digests must be strings")
        prefix_items = payload["prefix_items"]
        if type(prefix_items) is not int or prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        appended = _decode_chunks(payload["appended"], "appended")
        return cls(base_digest, target_digest, prefix_items, appended)


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Rebuild the target pack from a base pack and a delta."""

    if delta.base_digest != base.digest:
        raise ContextPackError("delta does not apply to this base pack")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("prefix_items exceeds the base pack chunk count")
    rebuilt = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("delta target digest does not match the rebuilt pack")
    return rebuilt


def _wire_size(items: tuple[bytes, ...], digest: str) -> int:
    """Resolve the self-referential wire size to its fixed point."""

    byte_count = 0
    while True:
        payload = {
            "schema": CONTEXT_PACK_SCHEMA,
            "items": [base64.b64encode(item).decode("ascii") for item in items],
            "digest": digest,
            "byte_count": byte_count,
        }
        size = len(_canonical(payload))
        if size == byte_count:
            return byte_count
        byte_count = size


def _load_object(raw: bytes, fields: frozenset[str], label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as error:
        raise ContextPackError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError(f"{label} must be a JSON object")
    keys = set(payload)
    if keys != fields:
        missing = fields - keys
        extra = keys - fields
        raise ContextPackError(f"{label} fields mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    return payload


def _chunk_content(content: str | bytes, chunk_bytes: int) -> tuple[bytes, ...]:
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ContextPackError("chunk_bytes must be a positive integer")
    data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    if not data:
        return ()
    chunks: list[bytes] = []
    start = 0
    while start < len(data):
        limit = min(start + chunk_bytes, len(data))
        end = limit
        if limit < len(data):
            newline = data.rfind(b"\n", start, limit)
            if newline >= start:
                end = newline + 1
        while end > start:
            try:
                data[start:end].decode("utf-8")
            except UnicodeDecodeError:
                end -= 1
                continue
            break
        if end == start:
            raise ContextPackError("chunk_bytes cannot hold a single UTF-8 character")
        chunks.append(data[start:end])
        start = end
    return tuple(chunks)


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
