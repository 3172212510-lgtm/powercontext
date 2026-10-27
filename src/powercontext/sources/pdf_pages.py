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

"""Page-level Sources for a complete PDF and question-to-page retrieval.

Every physical page of the original file becomes one ``CAPTURED`` Source, so a
search hit can be traced back to the exact page of the exact document revision.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Final
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.artifacts.models import MemoryCitation
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

PDF_PAGE_SOURCE_NAME: Final = "pdf-page"
PDF_PAGE_SOURCE_VERSION: Final = "1"
_SHA_PREFIX_LENGTH: Final = 32
_MAX_TERMS: Final = 24
_RRF_OFFSET: Final = 60


class PDFDocumentInput(BaseModel):
    """One complete original PDF addressed by a caller-stable document id."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """The extractable text of one physical page."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured evidence for exactly one physical PDF page."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


class _PDFPageTextEvidence:
    """Project one page into the standard textual evidence envelope."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


class _PDFPageAdapter:
    """Resolve and read page Sources without touching a catalog."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    adapter=_PDFPageAdapter(),
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(_PDFPageTextEvidence(),),
)


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Return the stable identity of one page of one document revision."""

    encoded = quote(document_id, safe="")
    identity = f"pdf:{encoded}:{document_sha256[:_SHA_PREFIX_LENGTH]}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:_SHA_PREFIX_LENGTH]
        identity = f"pdf:{digest}:{document_sha256[:_SHA_PREFIX_LENGTH]}:{page_number}"
    return identity


def parse_pdf_document(value: PDFDocumentInput, /) -> tuple[PDFPageInput, ...]:
    """Extract every physical page of the original PDF, in order.

    The whole document is read before anything is returned; a document that
    cannot be fully parsed fails as a whole.
    """

    document_id = value.document_id
    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without surrounding whitespace")
    content = value.content
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise ValueError("PDF content must be non-empty bytes")

    document_sha256 = hashlib.sha256(bytes(content)).hexdigest()
    pages: list[PDFPageInput] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(bytes(content)), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a password that was not supplied")
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=document_sha256,
                    page_number=page_number,
                    text=page.extract_text() or "",
                )
            )
    except ValueError:
        raise
    except Exception as error:  # noqa: BLE001 - every parser failure is a document failure
        raise ValueError("PDF document could not be fully parsed") from error
    return tuple(pages)


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Return up to ``limit`` exact page citations for one natural-language question.

    One lexical query over a whole question has to satisfy a coverage floor that
    scales with its term count, which drops pages that carry only part of the
    question. Retrieving each distinct term on its own and fusing the rankings
    keeps those partial matches without giving up the whole-question ranking.
    """

    from powercontext.builtin.artifacts.search import analyze_fts_query
    from powercontext.builtin.runtime.models import SearchMemoryRequest

    if limit < 1:
        raise ValueError("limit must be positive")
    question = (question or "").strip()
    if not question:
        return ()

    terms = tuple(dict.fromkeys(analyze_fts_query(question).split()))[:_MAX_TERMS]
    rankings: list[list] = []
    page = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=max(limit * 2, 40)))
    rankings.append(list(page.hits))
    for term in terms:
        term_page = await memory.search(
            SearchMemoryRequest(query=term, mode="fts", limit=max(limit * 6, 120))
        )
        rankings.append(list(term_page.hits))

    scores: dict[tuple[str, str, int, str], float] = {}
    hits_by_key: dict[tuple[str, str, int, str], object] = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking):
            key = (hit.memory_ref.family, hit.memory_ref.artifact_id, hit.memory_ref.revision, hit.entry_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_OFFSET + position)
            hits_by_key.setdefault(key, hit)

    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return tuple(
        MemoryCitation(
            memory_ref=hits_by_key[key].memory_ref,
            entry_id=hits_by_key[key].entry_id,
            entry_version_id=hits_by_key[key].entry_version_id,
        )
        for key in ordered[:limit]
    )


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
