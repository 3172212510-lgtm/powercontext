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

"""Score Memory candidates through an HTTP reranking service.

The service caps how many candidates one request may carry, so the adapter splits
the pool into batches. Batches keep the original one-based ranks of the whole
pool, and the final selection is decided once over every batch score.
"""

from __future__ import annotations

import math
from typing import Any, Final

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_BATCH_SIZE: Final = 32
DEFAULT_POLICY_ID: Final = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Select Memory hits by scores returned from one HTTP scoring service."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = DEFAULT_BATCH_SIZE,
        policy_id: str = DEFAULT_POLICY_ID,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once, then select the global top ``limit`` ranks."""

        candidate_count = len(candidates)
        if candidate_count < 1:
            raise ValueError("rerank requires at least one candidate")
        if limit < 1 or limit > candidate_count:
            raise ValueError("rerank limit must be between one and the candidate count")

        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, candidate_count, self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = list(range(start + 1, start + len(batch) + 1))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": hit.text} for rank, hit in zip(ranks, batch, strict=True)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            requests += 1
            response.raise_for_status()
            scores.update(_parse_scores(response, ranks))

        ordered = sorted(scores, key=lambda rank: (-scores[rank], rank))
        return MemoryRerankDecision(
            selected_ranks=tuple(ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _parse_scores(response: httpx.Response, expected_ranks: list[int]) -> dict[int, float]:
    try:
        payload: Any = response.json()
    except ValueError as error:
        raise ValueError("rerank service returned invalid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("rerank response must contain exactly the scores field")
    entries = payload["scores"]
    if not isinstance(entries, list):
        raise ValueError("rerank scores must be a list")

    batch = set(expected_ranks)
    observed: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"rank", "score"}:
            raise ValueError("rerank score entries must contain exactly rank and score")
        rank = entry["rank"]
        score = entry["score"]
        if type(rank) is not int:
            raise ValueError("rerank ranks must be integers")
        if rank not in batch:
            raise ValueError("rerank scores must cover exactly the ranks of their batch")
        if rank in observed:
            raise ValueError("rerank scores must cover each rank once")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("rerank scores must be finite numbers")
        observed[rank] = float(score)

    if set(observed) != batch:
        raise ValueError("rerank scores must cover every rank of their batch")
    return observed


__all__ = ["DEFAULT_BATCH_SIZE", "DEFAULT_POLICY_ID", "HTTPMemoryReranker"]
