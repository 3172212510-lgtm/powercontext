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

"""Share in-flight embedding work across callers and cache successful vectors.

Every caller keeps its own ordering and usage accounting, while identical text
inside one coalescing window, or already in flight, is embedded once. A caller
that is cancelled only drops its own wait; the shared provider call keeps going
for everyone else.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

from powercontext.builtin.artifacts.memory import EmbeddingProfile
from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage

DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_COALESCE_MS = 5.0
DEFAULT_CACHE_SIZE = 1024

_Result = tuple[EmbeddingVector, int, InferenceUsage]


class _Batch:
    """One underlying provider call and the texts it carries."""

    __slots__ = ("texts",)

    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts


class CoalescingEmbeddingModel:
    """An EmbeddingModel wrapper that deduplicates text and caches vectors."""

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: float = DEFAULT_COALESCE_MS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")

        self._model = model
        self.profile: EmbeddingProfile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        # Texts queued for, or already executing in, an underlying batch.
        self._futures: dict[str, asyncio.Future[_Result]] = {}
        self._queue: list[str] = []
        self._flusher: asyncio.Task[None] | None = None
        self._batches: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        texts = tuple(texts)
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        unique = list(dict.fromkeys(texts))
        vectors: dict[str, EmbeddingVector] = {}
        waiting: dict[str, asyncio.Future[_Result]] = {}

        for text in unique:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                vectors[text] = cached
                continue
            future = self._futures.get(text)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                self._futures[text] = future
                self._queue.append(text)
            waiting[text] = future

        if self._queue and self._flusher is None:
            self._flusher = asyncio.create_task(self._flush())

        requests = 0
        input_tokens: int | None = None
        output_tokens: int | None = None
        counted: set[int] = set()
        for text, future in waiting.items():
            # A shielded wait keeps the shared batch alive when this caller is cancelled.
            vector, batch_id, usage = await asyncio.shield(future)
            vectors[text] = vector
            if batch_id in counted:
                continue
            counted.add(batch_id)
            requests += usage.requests
            if usage.input_tokens is not None:
                input_tokens = (input_tokens or 0) + usage.input_tokens
            if usage.output_tokens is not None:
                output_tokens = (output_tokens or 0) + usage.output_tokens

        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in texts),
            usage=InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens),
        )

    async def _flush(self) -> None:
        try:
            while self._queue:
                if self._coalesce_seconds:
                    await asyncio.sleep(self._coalesce_seconds)
                    if not self._queue:
                        break
                texts = tuple(self._queue[: self._max_batch_size])
                del self._queue[: self._max_batch_size]
                task = asyncio.create_task(self._run_batch(texts))
                self._batches.add(task)
                task.add_done_callback(self._batches.discard)
        finally:
            self._flusher = None

    async def _run_batch(self, texts: tuple[str, ...]) -> None:
        batch_id = id(texts)
        futures = {text: self._futures.get(text) for text in texts}
        try:
            result = await self._model.embed(texts)
            vectors = tuple(tuple(float(value) for value in vector) for vector in result.vectors)
            if len(vectors) != len(texts):
                raise ValueError("embedding provider returned a different number of vectors than texts")
        except BaseException as error:  # noqa: BLE001 - the batch failure belongs to its waiters
            for text in texts:
                future = futures[text]
                if future is None or future.done():
                    continue
                future.set_exception(error)
                future.add_done_callback(_consume_outcome)
                if self._futures.get(text) is future:
                    del self._futures[text]
            return

        if self._cache_size:
            for text, vector in zip(texts, vectors, strict=True):
                self._cache[text] = vector
                self._cache.move_to_end(text)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        for text, vector in zip(texts, vectors, strict=True):
            future = futures[text]
            if future is None:
                continue
            if self._futures.get(text) is future:
                del self._futures[text]
            if not future.done():
                future.set_result((vector, batch_id, result.usage))


def _consume_outcome(future: asyncio.Future[Any]) -> None:
    """Mark a failed future as retrieved when no waiter is left to observe it."""

    if not future.cancelled():
        future.exception()


__all__ = [
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_COALESCE_MS",
    "DEFAULT_MAX_BATCH_SIZE",
    "CoalescingEmbeddingModel",
]
