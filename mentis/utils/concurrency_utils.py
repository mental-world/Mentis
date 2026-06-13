"""Bounded concurrency helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def gather_bounded(
    items: list[T], worker: Callable[[T], Awaitable[R]], max_concurrency: int
) -> list[R]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(item: T) -> R:
        async with semaphore:
            return await worker(item)

    return await asyncio.gather(*(run(item) for item in items))
