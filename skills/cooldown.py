"""
Attempt count and cooldown enforcement for skill selection.

Rules (from plan):
  - Max 3 attempts per user per skill
  - 60-day cooldown after each attempt
  - FRAUD verdict: user is_active=False, blocked from all skills
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from cache.redis_client import get_cooldown
from core.exceptions import InCooldownError, MaxAttemptsReachedError, UserInactiveError
from core.logging import get_logger
from db.adapter import DatabaseAdapter
from db.crud import (
    count_attempts,
    get_latest_application_for_skill,
    get_skill_by_id,
)
from db.models import User

log = get_logger(__name__)

MAX_ATTEMPTS = 3
COOLDOWN_DAYS = 60


async def check_skill_eligibility(
    db: DatabaseAdapter,
    user: User,
    skill_id: UUID,
) -> int:
    """
    Validate that user can start a new application for skill_id.
    Returns the next attempt_number (1, 2, or 3) if allowed.
    Raises descriptive exceptions if blocked.
    """
    # 1 — FRAUD check
    if not user.is_active:
        log.warning(
            "Blocked skill attempt — user is inactive (FRAUD)",
            user_id=str(user.id),
            skill_id=str(skill_id),
        )
        raise UserInactiveError()

    skill = await get_skill_by_id(db, skill_id)
    if skill is None:
        from core.exceptions import SkillNotFoundError
        raise SkillNotFoundError(str(skill_id))

    # 2 — Max attempts
    attempt_count = await count_attempts(db, user.id, skill_id)
    if attempt_count >= MAX_ATTEMPTS:
        log.info(
            "Max attempts reached",
            user_id=str(user.id),
            skill_name=skill.name,
            attempts=attempt_count,
        )
        raise MaxAttemptsReachedError(skill.name)

    # 3 — Cooldown (Redis fast path first)
    cooldown_iso = await get_cooldown(str(user.id), str(skill_id))

    if cooldown_iso is None:
        # Redis miss — check DB
        latest = await get_latest_application_for_skill(db, user.id, skill_id)
        if latest and latest.cooldown_until:
            if latest.cooldown_until.tzinfo is None:
                # Naive datetime from DB — assume UTC
                cooldown_ts = latest.cooldown_until.replace(tzinfo=timezone.utc)
            else:
                cooldown_ts = latest.cooldown_until
            cooldown_iso = cooldown_ts.isoformat()

    if cooldown_iso:
        try:
            cooldown_ts = datetime.fromisoformat(cooldown_iso)
            if cooldown_ts.tzinfo is None:
                cooldown_ts = cooldown_ts.replace(tzinfo=timezone.utc)
            if datetime.now(tz=timezone.utc) < cooldown_ts:
                log.info(
                    "Skill attempt blocked — in cooldown",
                    user_id=str(user.id),
                    skill_name=skill.name,
                    cooldown_until=cooldown_iso,
                )
                raise InCooldownError(skill.name, cooldown_iso)
        except InCooldownError:
            raise
        except Exception:
            log.warning("Could not parse cooldown timestamp", value=cooldown_iso)

    next_attempt = attempt_count + 1
    log.info(
        "Skill eligibility confirmed",
        user_id=str(user.id),
        skill_name=skill.name,
        next_attempt=next_attempt,
    )
    return next_attempt
