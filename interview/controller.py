"""
Interview controller — tracks question count and time limit.

Signals INTERVIEW_COMPLETE when either:
  - question_count reaches TARGET_QUESTIONS
  - elapsed time exceeds MAX_DURATION_SECONDS

Each exchange (assistant Q + user A) counts as one question.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable, Coroutine, Any

from core.logging import get_logger

log = get_logger(__name__)

TARGET_QUESTIONS = 8
MAX_DURATION_SECONDS = 140  # ~2 min 20 s (slight buffer over 2 min)


class InterviewController:
    def __init__(
        self,
        session_id: str,
        on_complete: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        self.session_id = session_id
        self.question_count = 0
        self._start_time = time.monotonic()
        self._on_complete = on_complete
        self._completed = False
        self._timer_task: asyncio.Task | None = None

    def start_timer(self) -> None:
        self._timer_task = asyncio.create_task(self._timeout_watchdog())
        log.info(
            "Interview timer started",
            session_id=self.session_id,
            max_seconds=MAX_DURATION_SECONDS,
        )

    async def _timeout_watchdog(self) -> None:
        await asyncio.sleep(MAX_DURATION_SECONDS)
        if not self._completed:
            log.info(
                "Interview timed out — triggering completion",
                session_id=self.session_id,
            )
            await self._trigger_complete("timeout")

    def record_question(self) -> None:
        self.question_count += 1
        elapsed = time.monotonic() - self._start_time
        log.debug(
            "Question recorded",
            session_id=self.session_id,
            question_count=self.question_count,
            elapsed_seconds=round(elapsed, 1),
        )

    def is_complete(self) -> bool:
        elapsed = time.monotonic() - self._start_time
        return self.question_count >= TARGET_QUESTIONS or elapsed >= MAX_DURATION_SECONDS

    async def check_and_complete(self) -> bool:
        if not self._completed and self.is_complete():
            await self._trigger_complete("questions_exhausted")
            return True
        return False

    async def _trigger_complete(self, reason: str) -> None:
        if self._completed:
            return
        self._completed = True
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        log.info(
            "Interview completed",
            session_id=self.session_id,
            reason=reason,
            question_count=self.question_count,
            elapsed_seconds=round(time.monotonic() - self._start_time, 1),
        )
        await self._on_complete()

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time
