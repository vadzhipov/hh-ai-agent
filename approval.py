from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from config import Settings
from cover_letter import has_required_portfolio
from database import ClaimResult, Database


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplicationPermission:
    job_id: str
    permit: str
    telegram_user_id: int


@dataclass(frozen=True)
class ApprovalResult:
    ok: bool
    message: str


class PhysicalApplicationSender(Protocol):
    async def submit_application(self, permission: ApplicationPermission) -> bool: ...


class ApprovalGuard:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.settings = settings
        self.database = database
        self.now_factory = now_factory or (lambda: datetime.now().astimezone())

    def claim(self, permission: ApplicationPermission) -> ClaimResult:
        return self.database.claim_application(
            job_id=permission.job_id,
            permit=permission.permit,
            telegram_user_id=permission.telegram_user_id,
            expected_user_id=self.settings.tg_user_id,
            app_mode=self.settings.app_mode,
            enable_real_apply=self.settings.enable_real_apply,
            daily_limit=self.settings.max_applications_per_day,
            now=self.now_factory(),
        )


class ApprovalService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        sender: PhysicalApplicationSender,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.settings = settings
        self.database = database
        self.sender = sender
        self.now_factory = now_factory or (lambda: datetime.now().astimezone())

    async def approve_and_apply(
        self, job_id: str, telegram_user_id: int
    ) -> ApprovalResult:
        if telegram_user_id != self.settings.tg_user_id:
            logger.warning("application_blocked job_id=%s reason=wrong_user", job_id)
            return ApprovalResult(False, "Action is not allowed")
        if self.settings.app_mode != "approval":
            logger.warning("application_blocked job_id=%s reason=app_mode", job_id)
            return ApprovalResult(False, "Real applications are disabled in dry-run")
        if not self.settings.enable_real_apply:
            logger.warning("application_blocked job_id=%s reason=real_apply_disabled", job_id)
            return ApprovalResult(False, "Real applications are disabled")
        vacancy = self.database.get(job_id)
        if vacancy is not None and not has_required_portfolio(
            vacancy.cover_letter, self.settings.profile.cover_letter.required_portfolio_url
        ):
            return ApprovalResult(False, "В письме отсутствует обязательная ссылка на портфолио")
        permit = self.database.approve(
            job_id,
            telegram_user_id,
            self.settings.tg_user_id,
            self.now_factory(),
            self.settings.approval_ttl_minutes,
        )
        if permit is None:
            logger.warning("application_blocked job_id=%s reason=approval_invalid", job_id)
            return ApprovalResult(False, "Approval is missing, expired, or already used")
        logger.info("approval_received job_id=%s", job_id)
        permission = ApplicationPermission(job_id, permit, telegram_user_id)
        sent = await self.sender.submit_application(permission)
        return ApprovalResult(sent, "Application sent" if sent else "Application failed")

    async def auto_apply(self, job_id: str) -> ApprovalResult:
        if not self.settings.auto_apply.enabled:
            return ApprovalResult(False, "Automatic applications are disabled")
        if self.settings.app_mode != "approval" or not self.settings.enable_real_apply:
            logger.warning("auto_application_blocked job_id=%s reason=mode", job_id)
            return ApprovalResult(False, "Real applications are disabled")
        vacancy = self.database.get(job_id)
        if vacancy is None or vacancy.status.value != "pending_approval":
            logger.warning("auto_application_blocked job_id=%s reason=not_pending", job_id)
            return ApprovalResult(False, "Vacancy is not pending")
        if (
            vacancy.confidence is None
            or vacancy.confidence < self.settings.auto_apply.min_confidence
        ):
            logger.info("auto_application_blocked job_id=%s reason=confidence", job_id)
            return ApprovalResult(False, "Vacancy confidence is below the auto-apply threshold")
        if not has_required_portfolio(
            vacancy.cover_letter, self.settings.profile.cover_letter.required_portfolio_url
        ):
            logger.warning("auto_application_blocked job_id=%s reason=portfolio", job_id)
            return ApprovalResult(False, "В письме отсутствует обязательная ссылка на портфолио")
        permit = self.database.approve(
            job_id,
            self.settings.tg_user_id,
            self.settings.tg_user_id,
            self.now_factory(),
            self.settings.approval_ttl_minutes,
        )
        if permit is None:
            logger.warning("auto_application_blocked job_id=%s reason=approval_invalid", job_id)
            return ApprovalResult(False, "Automatic approval is unavailable")
        logger.info("auto_application_received job_id=%s", job_id)
        permission = ApplicationPermission(job_id, permit, self.settings.tg_user_id)
        sent = await self.sender.submit_application(permission)
        return ApprovalResult(sent, "Application sent" if sent else "Application failed")

    def skip(self, job_id: str, telegram_user_id: int) -> ApprovalResult:
        if telegram_user_id != self.settings.tg_user_id:
            return ApprovalResult(False, "Action is not allowed")
        skipped = self.database.skip(
            job_id, telegram_user_id, self.settings.tg_user_id
        )
        return ApprovalResult(skipped, "Vacancy skipped" if skipped else "Vacancy is not pending")
