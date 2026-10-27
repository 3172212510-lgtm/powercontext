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

"""Capture the visible events of one selected Pi session branch.

A Pi archive is a tree: every record names its parent, so a leaf plus its
ancestors is one history path. Control records keep the graph connected but
produce no text evidence. Each visible event becomes one immutable Source, and
the Connector only submits events its durable checkpoint has not seen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    TextEvidence,
)
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME: Final = "pi-event"
PI_SESSION_CONNECTOR_NAME: Final = "pi-session-history"
PI_SESSION_VERSION: Final = "1"
CHECKPOINT_SCHEMA: Final = "powercontext.pi-session-checkpoint.v1"

_CONTROL_EVENT_TYPES: Final = frozenset(
    {
        "session",
        "model_change",
        "thinking_level_change",
        "service_tier_change",
        "label",
        "permission_change",
        "system",
    }
)


class PiSessionInput(BaseModel):
    """One complete Pi JSONL archive and the branch leaf to capture."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event with its original record and derived text body."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


class PiEventSource(Source):
    """Captured evidence for exactly one visible session event."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


class _PiEventTextEvidence:
    """Project one event into the standard textual evidence envelope."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class _PiEventAdapter:
    """Resolve and read event Sources without touching a catalog."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    adapter=_PiEventAdapter(),
    version=PI_SESSION_VERSION,
    projections=(_PiEventTextEvidence(),),
)


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Return the stable identity of one event inside one session."""

    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"pi event identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return identity


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Return the visible events of the selected branch, ordered root to leaf."""

    session_id, nodes, order = _load_archive(value.content)
    if not order:
        return ()
    leaf = value.leaf_id
    if leaf is None:
        leaf = order[-1]
    elif leaf not in nodes:
        raise ValueError(f"leaf {leaf!r} is not present in the session archive")

    chain: list[str] = []
    cursor: str | None = leaf
    while cursor is not None:
        if cursor in chain:
            raise ValueError("session archive contains a parent cycle")
        record = nodes.get(cursor)
        if record is None:
            raise ValueError(f"parent {cursor!r} is not present in the session archive")
        chain.append(cursor)
        cursor = record.get("parentId")
    chain.reverse()

    events: list[PiEventInput] = []
    for event_id in chain:
        event = _event_from_record(session_id, event_id, nodes[event_id])
        if event is not None:
            events.append(event)
    return tuple(events)


def pi_record_digest(record: dict[str, Any]) -> str:
    """Return a stable digest of one raw record for change detection."""

    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class PiSessionConnector:
    """Submit only the visible events of one branch that are not yet imported."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        checkpoint = session.checkpoint
        imported, session_id = _read_checkpoint(checkpoint)
        events = parse_pi_session(PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id))

        digests: dict[str, str] = {}
        for event in events:
            digest = pi_record_digest(event.record)
            digests[event.event_id] = digest
            previous = imported.get(event.event_id)
            if previous is not None:
                if previous != digest:
                    raise ValueError(f"already imported event {event.event_id!r} was rewritten")
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            imported[event.event_id] = digest

        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={
                "schema": CHECKPOINT_SCHEMA,
                "session_id": session_id,
                "events": imported,
            },
        )


def _read_checkpoint(checkpoint: Any) -> tuple[dict[str, str], str | None]:
    if checkpoint is None:
        return {}, None
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        return {}, None
    session_id = checkpoint.get("session_id")
    events = checkpoint.get("events")
    if not isinstance(events, dict):
        return {}, session_id if isinstance(session_id, str) else None
    imported = {key: value for key, value in events.items() if isinstance(key, str) and isinstance(value, str)}
    return imported, session_id if isinstance(session_id, str) else None


def _load_archive(content: bytes) -> tuple[str, dict[str, dict[str, Any]], list[str]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session archive must be strict UTF-8") from error

    records: list[Any] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError("Pi session archive must contain one JSON object per line") from error
    if not records:
        raise ValueError("Pi session archive must start with a session header")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Pi session archive records must be JSON objects")

    header = records[0]
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("Pi session archive must declare type=session and version=3")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header must carry a non-empty id")

    nodes: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records[1:]:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("every Pi session record must carry a non-empty id")
        if event_id in nodes:
            raise ValueError(f"duplicate Pi session record id {event_id!r}")
        parent = record.get("parentId")
        if parent is not None and (not isinstance(parent, str) or parent not in nodes):
            raise ValueError(f"record {event_id!r} names an unknown parent")
        nodes[event_id] = record
        order.append(event_id)
    return session_id, nodes, order


def _event_from_record(session_id: str, event_id: str, record: dict[str, Any]) -> PiEventInput | None:
    event_type = record.get("type")
    if not isinstance(event_type, str) or not event_type:
        return None

    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        role = role if isinstance(role, str) and role else "message"
        body = _message_body(role, message)
    elif event_type in {"compaction", "branch_summary"}:
        role = event_type
        summary = record.get("summary")
        body = summary if isinstance(summary, str) else ""
    elif event_type == "custom_message":
        role = "custom"
        body = _text_blocks(record.get("content"))
    elif event_type in _CONTROL_EVENT_TYPES:
        return None
    else:
        return None

    if body is None or not body.strip():
        return None
    parent_id = record.get("parentId")
    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=parent_id if isinstance(parent_id, str) else None,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_body(role: str, message: dict[str, Any]) -> str | None:
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        body = f"$ {command if isinstance(command, str) else ''}"
        if isinstance(output, str) and output:
            body = f"{body}\n{output}"
        return body
    if role == "toolResult":
        tool_name = message.get("toolName")
        call_id = message.get("toolCallId")
        text = _text_blocks(message.get("content"))
        prefix = f"[tool_result {tool_name if isinstance(tool_name, str) else ''} {call_id if isinstance(call_id, str) else ''}]"
        return f"{prefix} {text}" if text else prefix
    return _text_blocks(message.get("content"), tool_calls=True)


def _text_blocks(content: Any, *, tool_calls: bool = False) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "toolCall" and tool_calls:
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            parts.append(
                f"[tool_call {name if isinstance(name, str) else ''} {call_id if isinstance(call_id, str) else ''}] "
                f"{json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(',', ':'))}"
            )
    return "\n".join(parts)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
    "pi_record_digest",
]
