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
import math
import re
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
# Each question term is retrieved on its own down to this depth before scoring.
_TERM_DEPTH: Final = 200
# Coverage of the question scales a page score but never zeroes a partial match.
_COVERAGE_FLOOR: Final = 0.25
_ORDER_WEIGHT: Final = 12.0
# A thesaurus spelling counts for less than a word the question actually used.
_EXPANSION_WEIGHT: Final = 0.5
_TOKEN = re.compile(r"[a-z]+|\d+")

# Filing questions name concepts the report spells differently. The map is
# generic 10-K vocabulary, never document or question specific.
_THESAURUS: Final[dict[str, tuple[str, ...]]] = {
    "geographies": ("countries", "regions", "territories"),
    "geography": ("countries", "regions", "territories"),
    "legal": ("litigation", "proceedings", "contingencies"),
    "litigation": ("legal", "proceedings", "contingencies"),
    "battles": ("proceedings", "litigation", "contingencies"),
    "capex": ("capital", "expenditures"),
    "p&l": ("income", "statement", "operations"),
    "ebitda": ("operating", "income", "depreciation", "amortization"),
    "headcount": ("employees", "personnel"),
    "buyback": ("repurchase", "shares"),
    "guidance": ("outlook", "expect"),
    "segments": ("reportable", "segments"),
    "dividends": ("dividend", "share"),
}

_HitKey = tuple[str, str, int, str]
_PageKey = tuple[str, int]
_MAX_RESOLVED_HITS: Final = 220
_MAX_CACHED_PAGES: Final = 8192

# Page identities are unique across scopes, so one process-wide cache is safe.
_PAGE_OF_HIT: dict[_HitKey, _PageKey] = {}


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

    Filing questions also use vocabulary the report itself does not: a question
    asks about "legal battles" where the page is headed "Legal Proceedings", or
    about "geographies" where the text says "countries and territories". A small
    finance thesaurus adds those spellings at a lower weight, so a page can match
    the concept without the exact word.
    """

    from powercontext.builtin.artifacts.search import analyze_fts_query
    from powercontext.builtin.runtime.models import GetMemoryEntryRequest, SearchMemoryRequest

    if limit < 1:
        raise ValueError("limit must be positive")
    question = (question or "").strip()
    if not question:
        return ()

    base = tuple(dict.fromkeys(analyze_fts_query(question).split()))[:_MAX_TERMS]
    weighted = _weighted_terms(question, base)
    hits_by_key: dict[_HitKey, object] = {}
    order_bonus: dict[_HitKey, float] = {}

    whole = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=max(limit * 4, 80)))
    _collect(hits_by_key, order_bonus, list(whole.hits), 1.0)
    for term, _weight in weighted:
        term_page = await memory.search(SearchMemoryRequest(query=term, mode="fts", limit=_TERM_DEPTH))
        _collect(hits_by_key, order_bonus, list(term_page.hits), 0.35)

    counts = {key: _term_counts(hit.text) for key, hit in hits_by_key.items()}
    total = max(len(counts), 1)
    weights = {}
    for term, factor in weighted:
        document_frequency = sum(1 for counter in counts.values() if term in counter)
        weights[term] = factor * math.log(1.0 + total / (1.0 + document_frequency))

    scores: dict[_HitKey, float] = {}
    for key, counter in counts.items():
        matched = 0
        weight = 0.0
        for term, _factor in weighted:
            frequency = counter.get(term, 0)
            if not frequency:
                continue
            matched += 1
            weight += weights[term] * (1.0 + math.log(frequency))
        # Evidence pages carry most of the question, so coverage scales the score.
        coverage = matched / len(weighted) if weighted else 0.0
        scores[key] = weight * (_COVERAGE_FLOOR + (1.0 - _COVERAGE_FLOOR) * coverage)
        scores[key] += order_bonus.get(key, 0.0) * _ORDER_WEIGHT

    page_scores: dict[_PageKey, float] = {}
    page_keys: dict[_PageKey, _HitKey] = {}
    ordered_keys = sorted(scores, key=lambda key: (-scores[key], key))
    for key in ordered_keys[:_MAX_RESOLVED_HITS]:
        page = await _page_of(memory, hits_by_key[key], GetMemoryEntryRequest)
        if page is None:
            continue
        if scores[key] > page_scores.get(page, 0.0):
            page_scores[page] = scores[key]
            page_keys[page] = key

    ordered_pages = sorted(page_scores, key=lambda page: (-page_scores[page], page))
    return tuple(_citation(hits_by_key[page_keys[page]]) for page in ordered_pages[:limit])


def _weighted_terms(question: str, base: tuple[str, ...]) -> list[tuple[str, float]]:
    """Return the question terms plus thesaurus spellings at a lower weight."""

    lowered = question.lower()
    weighted: list[tuple[str, float]] = [(term, 1.0) for term in base]
    seen = set(base)
    for trigger, spellings in _THESAURUS.items():
        if trigger not in lowered:
            continue
        for spelling in spellings:
            if spelling in seen:
                continue
            seen.add(spelling)
            weighted.append((spelling, _EXPANSION_WEIGHT))
    return weighted



def _collect(
    hits_by_key: dict[_HitKey, object],
    order_bonus: dict[_HitKey, float],
    hits: list,
    weight: float,
) -> None:
    """Keep the first hit per page and record where each ranking placed it."""

    for position, hit in enumerate(hits):
        key = _hit_key(hit)
        hits_by_key.setdefault(key, hit)
        order_bonus[key] = order_bonus.get(key, 0.0) + weight / (_RRF_OFFSET + position)


def _term_counts(text: str) -> dict[str, int]:
    """Count the analysable terms of one page, matching the lexical contract."""

    counts: dict[str, int] = {}
    for token in _TOKEN.findall(text.lower()):
        counts[token] = counts.get(token, 0) + 1
    return counts


def _hit_key(hit: object) -> _HitKey:
    reference = hit.memory_ref
    return (reference.family, reference.artifact_id, reference.revision, hit.entry_id)


def _citation(hit: object) -> MemoryCitation:
    return MemoryCitation(
        memory_ref=hit.memory_ref,
        entry_id=hit.entry_id,
        entry_version_id=hit.entry_version_id,
    )


def _page_from_source_id(source_id: str) -> _PageKey | None:
    """Recover ``(document, page)`` from the identity of a page Source."""

    head, _, tail = source_id.rpartition(":")
    if not head.startswith("pdf:"):
        return None
    try:
        number = int(tail)
    except ValueError:
        return None
    document = head[len("pdf:") :].rpartition(":")[0]
    if not document or number < 1:
        return None
    return document, number


async def _page_of(memory: object, hit: object, request_type: type) -> _PageKey | None:
    """Resolve the physical page behind one Memory hit, caching the answer."""

    key = _hit_key(hit)
    cached = _PAGE_OF_HIT.get(key)
    if cached is not None:
        return cached
    entry = (await memory.get(request_type(citation=_citation(hit)))).entry
    for reference in entry.sources:
        page = _page_from_source_id(reference.source_id)
        if page is None:
            continue
        if len(_PAGE_OF_HIT) >= _MAX_CACHED_PAGES:
            _PAGE_OF_HIT.clear()
        _PAGE_OF_HIT[key] = page
        return page
    return None


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
