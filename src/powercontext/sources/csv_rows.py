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

"""Row-level CSV Sources that keep the original record recoverable.

One row is one ``CAPTURED`` Source, identified by the document and the primary
key value. Header order and field text are preserved verbatim so a search hit can
be traced back to the exact record that produced it.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Final
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

CSV_ROW_SOURCE_NAME: Final = "csv-row"
CSV_ROW_SOURCE_VERSION: Final = "1"
_CSV_DIALECT: Final = {
    "delimiter": ",",
    "quotechar": '"',
    "doublequote": True,
    "strict": True,
}


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document addressed by its primary key column."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One data row in header order, with the primary key value that identifies it."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured evidence for exactly one CSV data row."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class _CSVRowTextEvidence:
    """Project a row into the standard textual evidence envelope."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.name,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


class _CSVRowAdapter:
    """Resolve and read row Sources without touching a catalog."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    adapter=_CSVRowAdapter(),
    version=CSV_ROW_SOURCE_VERSION,
    projections=(_CSVRowTextEvidence(),),
)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the stable identity of one row Source.

    Both components are percent-encoded with an empty safe set, so a ``:`` inside
    a document id or a primary key value can never be mistaken for the separator.
    """

    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without surrounding whitespace")
    if not isinstance(primary_key_value, str):
        raise ValueError("primary_key_value must be a string")
    identity = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"csv row identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return identity


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse and validate one whole CSV document into row inputs.

    The document is fully parsed and validated before anything is returned; any
    encoding, CSV format or data error fails the whole call with ``ValueError``.
    """

    document_id = value.document_id
    primary_key = value.primary_key
    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without surrounding whitespace")
    if not isinstance(primary_key, str) or not primary_key:
        raise ValueError("primary_key must be a non-empty string")

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV document must be strict UTF-8") from error

    reader = csv.reader(io.StringIO(text, newline=""), **_CSV_DIALECT)
    try:
        records = [record for record in reader]
    except csv.Error as error:
        raise ValueError("CSV document is malformed") from error

    if not records:
        raise ValueError("CSV document must contain a header record")

    header = records[0]
    _validate_header(header)
    if primary_key not in header:
        raise ValueError(f"primary key {primary_key!r} is not a header column")
    key_index = header.index(primary_key)
    columns = tuple(header)

    rows: list[CSVRowInput] = []
    seen: set[str] = set()
    for record in records[1:]:
        if not record:
            continue
        if len(record) != len(header):
            raise ValueError("CSV record width must match the header width")
        primary_key_value = record[key_index]
        if not primary_key_value.strip():
            raise ValueError("primary key value must not be blank")
        if primary_key_value in seen:
            raise ValueError(f"duplicate primary key value {primary_key_value!r}")
        seen.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


def _validate_header(header: list[str]) -> None:
    if not header:
        raise ValueError("CSV header must not be empty")
    seen: set[str] = set()
    for column in header:
        if not column.strip():
            raise ValueError("CSV header columns must not be blank")
        if column in seen:
            raise ValueError(f"duplicate CSV header column {column!r}")
        seen.add(column)


__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
