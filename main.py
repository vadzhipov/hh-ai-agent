from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import random
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from aiogram.exceptions import TelegramAPIError

from ai_analyzer import AnalysisError, VacancyAnalyzer
from approval import ApprovalGuard, ApprovalService
from browser_backend import BrowserLaunchError, create_browser_backend
from config import ConfigError, Settings, load_settings
from database import Database, SearchRun, VacancyStatus
from fit_summary import normalize_fit_summary
from hh_client import HHClient, PageState, VacancySummary
from instance_lock import AlreadyRunningError, single_instance
from llm.base import LLMProvider
from llm.errors import LLMError
from llm.factory import create_llm_provider
from llm.mistral_keys import MistralKeyManager
from llm.types import LLMRequest
from logging_setup import configure_logging
from tg_bot import AgentControl, TelegramService
from vacancy_filter import vacancy_rejection_reason
from version import __version__


logger = logging.getLogger(__name__)
ANALYSIS_MAX_ATTEMPTS = 3
RECOVERABLE_AUTO_CIRCUIT_REASONS = frozenset({"search_errors"})


@dataclass(frozen=True)
class VacancyProcessResult:
    outcome: str
    reason: str = ""
    page_state: PageState | None = None


@dataclass
class SearchRunStats:
    query_count: int = 0
    found_results: int = 0
    new_vacancies: int = 0
    duplicates: int = 0
    rejected_by_filter: int = 0
    rejected_by_llm: int = 0
    telegram_cards: int = 0
    auto_applications: int = 0
    error_count: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    error_reasons: Counter[str] = field(default_factory=Counter)
    last_safe_error: str = ""
    read_attempts: int = 0
    technical_failures: int = 0
    consecutive_page_errors: int = 0

    def record_error(self, reason: str) -> None:
        self.error_count += 1
        self.error_reasons[reason] += 1
        self.last_safe_error = reason

    def record(self, result: VacancyProcessResult) -> None:
        if result.outcome == "rejected_by_filter":
            self.rejected_by_filter += 1
            self.rejection_reasons[result.reason or "filter_rejected"] += 1
        elif result.outcome == "rejected_by_llm":
            self.rejected_by_llm += 1
            self.rejection_reasons["llm_rejected"] += 1
        elif result.outcome == "telegram_card":
            self.telegram_cards += 1
        elif result.outcome == "auto_applied":
            self.auto_applications += 1
        elif result.outcome == "other_error":
            self.record_error(result.reason or "other_error")

        if result.page_state is None:
            return
        self.read_attempts += 1
        if result.page_state is PageState.PAGE_STRUCTURE_CHANGED:
            self.consecutive_page_errors += 1
        else:
            self.consecutive_page_errors = 0
        if result.page_state in {
            PageState.ACCESS_DENIED,
            PageState.CAPTCHA_DETECTED,
            PageState.NETWORK_ERROR,
        }:
            self.technical_failures += 1

    def circuit_reason(self, settings: Settings) -> str:
        if self.consecutive_page_errors >= settings.circuit_breaker_page_errors:
            return "page_structure_changed"
        if (
            self.read_attempts >= settings.circuit_breaker_min_sample
            and self.technical_failures / self.read_attempts
            >= settings.circuit_breaker_unknown_ratio
        ):
            return "technical_failure_ratio"
        return ""


def next_auto_batch_at(
    after: datetime, settings: Settings, *, interval_hours: int = 0
) -> datetime:
    schedule = settings.auto_apply
    local = after.astimezone(ZoneInfo(schedule.timezone))
    window_start = local.replace(
        hour=schedule.start_hour, minute=0, second=0, microsecond=0
    )
    window_end = local.replace(
        hour=schedule.end_hour, minute=0, second=0, microsecond=0
    )
    if local < window_start:
        return window_start
    if local >= window_end:
        return window_start + timedelta(days=1)
    candidate = local + timedelta(hours=interval_hours)
    return candidate if candidate < window_end else window_start + timedelta(days=1)


def next_auto_window_start_after(after: datetime, settings: Settings) -> datetime:
    schedule = settings.auto_apply
    local = after.astimezone(ZoneInfo(schedule.timezone))
    window_start = local.replace(
        hour=schedule.start_hour, minute=0, second=0, microsecond=0
    )
    return window_start if local < window_start else window_start + timedelta(days=1)


def restored_auto_batch_at(
    now: datetime,
    settings: Settings,
    database: Database,
    *,
    interval_hours: int,
) -> datetime:
    """Resume an underfilled batch now; otherwise preserve the normal interval."""
    previous = database.latest_search_run()
    if previous is None:
        return next_auto_batch_at(now, settings)
    try:
        finished_at = datetime.fromisoformat(previous.finished_at)
    except ValueError:
        logger.warning("auto_schedule_restore_failed run_id=%s", previous.id)
        return next_auto_batch_at(now, settings)
    if database.applied_today(now) < settings.auto_apply.min_batch_size:
        return next_auto_batch_at(now, settings)
    return next_auto_batch_at(
        finished_at, settings, interval_hours=interval_hours
    )


def recoverable_auto_retry_at(
    control: AgentControl,
    after: datetime,
    settings: Settings,
) -> datetime | None:
    """Clear a transient search pause and keep the retry inside active hours."""
    if (
        not control.paused
        or control.circuit_reason not in RECOVERABLE_AUTO_CIRCUIT_REASONS
    ):
        return None
    control.paused = False
    control.circuit_reason = ""
    control.consecutive_search_errors = 0
    return next_auto_batch_at(
        after + timedelta(minutes=settings.check_interval_minutes), settings
    )


def next_auto_batch_after_run(
    after: datetime,
    settings: Settings,
    *,
    sent: int,
    target: int,
    interval_hours: int,
) -> datetime:
    """Retry an underfilled batch sooner without leaving active hours."""
    if sent < target:
        return next_auto_batch_at(
            after + timedelta(minutes=settings.check_interval_minutes), settings
        )
    return next_auto_batch_at(after, settings, interval_hours=interval_hours)


def format_auto_batch_report(
    run: SearchRun,
    *,
    auto_applied: int,
    next_batch: datetime | None,
    settings: Settings,
) -> str:
    lines = [
        "Итог автоподачи",
        f"Найдено: {run.found_results}; новых: {run.new_vacancies}.",
        f"Отправлено: {auto_applied}; на ручную проверку: {run.telegram_cards}.",
        (
            "Отсечено фильтрами: "
            f"{run.rejected_by_filter + run.rejected_by_llm}; "
            f"пропущено без отправки: {run.error_count}."
        ),
    ]
    if next_batch is None:
        lines.append("Следующая пачка не назначена: автоподача приостановлена.")
    else:
        local_next = next_batch.astimezone(ZoneInfo(settings.auto_apply.timezone))
        lines.append(
            f"Следующая пачка: {local_next.strftime('%d.%m %H:%M')} "
            f"{settings.auto_apply.timezone}."
        )
    return "\n".join(lines)


async def process_vacancy(
    summary: VacancySummary,
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    *,
    approval_service: ApprovalService | None = None,
    now_factory: Callable[[], datetime] | None = None,
) -> VacancyProcessResult:
    clock = now_factory or (lambda: datetime.now(UTC))
    now = clock()
    existing = database.get(summary.id)
    if summary.previously_sent:
        if existing is None or existing.status is not VacancyStatus.PENDING_APPROVAL:
            return VacancyProcessResult("ignored")
        try:
            await telegram.send_preview(existing, include_actions=True)
        except TelegramAPIError:
            return VacancyProcessResult("other_error", "telegram_error")
        return VacancyProcessResult("telegram_card")
    if existing is not None and (
        existing.status is not VacancyStatus.DISCOVERED
        or existing.llm_decision is not None
    ):
        return VacancyProcessResult("ignored")
    details = await hh_client.read_vacancy(summary, telegram.request_captcha)
    description_hash = (
        hashlib.sha256(details.description.encode()).hexdigest()
        if details.description
        else ""
    )
    if existing is None:
        if not database.discover(
            job_id=summary.id,
            title=summary.title,
            company=details.company,
            company_url=details.company_url,
            url=summary.url,
            description_hash=description_hash,
            search_query=summary.search_query,
            discovered_at=now,
        ):
            return VacancyProcessResult("ignored")
        logger.info("vacancy_discovered job_id=%s", summary.id)
    else:
        logger.info("vacancy_resumed job_id=%s", summary.id)
    if details.state is not PageState.VACANCY_LOADED:
        error = details.error or details.state.value
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.APPLY_FAILED,
            error_text=error,
        )
        if details.state is PageState.CAPTCHA_DETECTED:
            await telegram.notify(f"CAPTCHA was not completed for vacancy {summary.id}.")
        logger.error("vacancy_read_failed job_id=%s state=%s", summary.id, details.state.value)
        return VacancyProcessResult("other_error", details.state.value, details.state)

    rejection = vacancy_rejection_reason(
        title=summary.title,
        company=details.company,
        description=details.description,
        excluded_positions=settings.profile.candidate.excluded_positions,
        excluded_companies=settings.profile.candidate.excluded_companies,
        excluded_keywords=settings.profile.candidate.excluded_keywords,
    )
    if rejection:
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.REJECTED_BY_FILTER,
            llm_reason=f"Excluded by profile filter: {rejection}",
        )
        logger.info("vacancy_rejected job_id=%s source=filter", summary.id)
        return VacancyProcessResult(
            "rejected_by_filter", rejection, PageState.VACANCY_LOADED
        )

    try:
        suitability = await analyzer.assess(summary.title, details.description)
    except AnalysisError as exc:
        database.mark_analysis_failed(
            summary.id,
            error_type=exc.error_type,
            now=clock(),
            retry_after=timedelta(minutes=settings.check_interval_minutes),
            max_attempts=ANALYSIS_MAX_ATTEMPTS,
        )
        await telegram.notify_analysis_failed(summary.title, summary.url, exc.error_type)
        logger.warning("vacancy_analysis_failed job_id=%s error_type=%s", summary.id, exc.error_type)
        return VacancyProcessResult(
            "other_error",
            f"analysis_failed:{exc.error_type}",
            PageState.VACANCY_LOADED,
        )

    if not suitability.suitable:
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.REJECTED_BY_LLM,
            llm_decision=False,
            llm_reason=suitability.reason,
            confidence=suitability.confidence,
            error_text="",
        )
        logger.info("vacancy_rejected job_id=%s source=llm", summary.id)
        return VacancyProcessResult(
            "rejected_by_llm", suitability.reason, PageState.VACANCY_LOADED
        )

    fit_summary = normalize_fit_summary(suitability.fit_points)
    company = await hh_client.read_company_details(details.company_url)
    database.store_company_details(
        summary.id, rating=company.rating, reviews_count=company.reviews_count
    )

    letter = await analyzer.generate_cover_letter(
        summary.title, details.description, company_name=details.company
    )
    if not letter.strip():
        database.transition(
            summary.id,
            VacancyStatus.DISCOVERED,
            VacancyStatus.APPLY_FAILED,
            error_text="cover_letter_failed",
        )
        await telegram.notify_analysis_failed(summary.title, summary.url, "cover_letter_failed")
        return VacancyProcessResult(
            "other_error", "cover_letter_failed", PageState.VACANCY_LOADED
        )

    include_actions = settings.app_mode != "dry_run"
    if settings.app_mode == "dry_run":
        database.store_analysis(
            summary.id,
            cover_letter=letter,
            llm_decision=True,
            llm_reason=suitability.reason,
            confidence=suitability.confidence,
            fit_summary=fit_summary,
        )
    elif not database.request_approval(
        job_id=summary.id,
        cover_letter=letter,
        llm_decision=True,
        llm_reason=suitability.reason,
        confidence=suitability.confidence,
        fit_summary=fit_summary,
        now=now,
    ):
        return VacancyProcessResult(
            "other_error", "approval_transition_failed", PageState.VACANCY_LOADED
        )
    elif (
        settings.auto_apply.enabled
        and suitability.confidence >= settings.auto_apply.min_confidence
    ):
        if approval_service is None:
            return VacancyProcessResult(
                "other_error", "auto_apply_service_unavailable", PageState.VACANCY_LOADED
            )
        auto_result = await approval_service.auto_apply(summary.id)
        vacancy = database.get(summary.id)
        if auto_result.ok:
            try:
                await telegram.notify(
                    f"✓ Автоотклик отправлен: {vacancy.title if vacancy else summary.title}"
                )
            except TelegramAPIError:
                logger.warning("auto_application_notification_failed job_id=%s", summary.id)
            return VacancyProcessResult("auto_applied", page_state=PageState.VACANCY_LOADED)
        if vacancy and vacancy.status is VacancyStatus.REJECTED_BY_FILTER:
            reason = vacancy.llm_reason.removeprefix(
                "Excluded by questionnaire filter: "
            )
            try:
                await telegram.notify(
                    f"↷ Вакансия пропущена по ответу в анкете: {vacancy.title}. {reason}"
                )
            except TelegramAPIError:
                logger.warning("auto_application_notification_failed job_id=%s", summary.id)
            return VacancyProcessResult(
                "rejected_by_filter", reason, PageState.VACANCY_LOADED
            )
        if vacancy and vacancy.error_text.startswith("questionnaire_required"):
            details = vacancy.error_text.partition(":")[2]
            await telegram.notify_questionnaire_required(
                vacancy.title, vacancy.url, details
            )
            return VacancyProcessResult(
                "telegram_card", "questionnaire_required", PageState.VACANCY_LOADED
            )
        try:
            await telegram.notify(
                f"✗ Автоотклик не отправлен: {vacancy.title if vacancy else summary.title}. "
                f"{auto_result.message}"
            )
        except TelegramAPIError:
            logger.warning("auto_application_notification_failed job_id=%s", summary.id)
        return VacancyProcessResult(
            "other_error", "auto_apply_failed", PageState.VACANCY_LOADED
        )
    try:
        await telegram.send_preview(
            database.get(summary.id), include_actions=include_actions
        )
    except TelegramAPIError:
        return VacancyProcessResult(
            "other_error", "telegram_error", PageState.VACANCY_LOADED
        )
    return VacancyProcessResult("telegram_card", page_state=PageState.VACANCY_LOADED)


async def run_search_cycle(
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    control: AgentControl,
    *,
    approval_service: ApprovalService | None = None,
    auto_apply_batch_limit: int | None = None,
    now_factory: Callable[[], datetime] | None = None,
) -> SearchRun:
    if auto_apply_batch_limit is not None and auto_apply_batch_limit < 1:
        raise ValueError("auto_apply_batch_limit must be positive")
    now = now_factory or (lambda: datetime.now(UTC))
    started_at = now()
    stats = SearchRunStats()
    handled_ids: set[str] = set()
    state = "completed"
    circuit_reason = ""
    failure: Exception | None = None
    logger.info(
        "search_cycle_started queries=%s", len(settings.profile.hh.search_queries)
    )

    async def process(summary: VacancySummary) -> None:
        nonlocal circuit_reason, state
        if summary.id in handled_ids:
            return
        handled_ids.add(summary.id)
        result = await process_vacancy(
            summary,
            settings,
            database,
            hh_client,
            analyzer,
            telegram,
            approval_service=approval_service,
            now_factory=now,
        )
        stats.record(result)
        if result.reason == "auto_apply_failed":
            failed_application = database.get(summary.id)
            if (
                failed_application is not None
                and failed_application.submit_attempted_at is not None
            ):
                circuit_reason = "application_delivery_uncertain"
                state = "paused_by_circuit_breaker"
                control.paused = True
                control.circuit_reason = circuit_reason
        if not circuit_reason:
            circuit_reason = stats.circuit_reason(settings)
        if circuit_reason:
            state = "paused_by_circuit_breaker"
            control.paused = True
            control.circuit_reason = circuit_reason

    try:
        if settings.auto_apply.questionnaires_enabled:
            database.requeue_legacy_questionnaire_failures()
        database.expire_approved(now())
        database.requeue_due_analysis_failures(
            now(), max_attempts=ANALYSIS_MAX_ATTEMPTS
        )
        for vacancy in database.unprocessed_discovered():
            if control.paused or (
                auto_apply_batch_limit is not None
                and stats.auto_applications >= auto_apply_batch_limit
            ):
                break
            await process(
                VacancySummary(
                    vacancy.id,
                    vacancy.title,
                    vacancy.url,
                    vacancy.search_query,
                )
            )
        for query in settings.profile.hh.search_queries:
            if control.paused or (
                auto_apply_batch_limit is not None
                and stats.auto_applications >= auto_apply_batch_limit
            ):
                break
            logger.info("search_query_started query=%r", query)
            search = await hh_client.search_vacancies(
                query,
                settings.profile.hh.areas,
                settings.profile.hh.experience_filters,
                remote_only=settings.profile.hh.remote_only,
            )
            stats.query_count += 1
            stats.found_results += search.found_results
            stats.new_vacancies += sum(
                not summary.previously_sent for summary in search.summaries
            )
            stats.duplicates += search.duplicates
            logger.info(
                "search_query_finished query=%r found=%s new=%s duplicates=%s",
                query,
                search.found_results,
                sum(not summary.previously_sent for summary in search.summaries),
                search.duplicates,
            )
            if search.error_reason:
                stats.record_error(search.error_reason)
                control.consecutive_search_errors += 1
            else:
                control.consecutive_search_errors = 0
            if (
                control.consecutive_search_errors
                >= settings.circuit_breaker_page_errors
            ):
                circuit_reason = "search_errors"
                state = "paused_by_circuit_breaker"
                control.paused = True
                control.circuit_reason = circuit_reason
            for summary in search.summaries:
                if control.paused or (
                    auto_apply_batch_limit is not None
                    and stats.auto_applications >= auto_apply_batch_limit
                ):
                    break
                await process(summary)
        if not control.paused:
            await hh_client.check_messages(telegram.notify)
    except Exception as exc:
        state = "failed"
        stats.record_error(type(exc).__name__)
        failure = exc

    if circuit_reason:
        if circuit_reason in RECOVERABLE_AUTO_CIRCUIT_REASONS:
            notification = (
                "Текущий поиск остановлен из-за сетевых ошибок. "
                "В режиме автоподачи повтор будет назначен автоматически."
            )
        else:
            notification = (
                f"Поиск приостановлен: {circuit_reason}. "
                "Проверьте /diagnostics и выполните /resume после устранения причины."
            )
    elif (
        auto_apply_batch_limit is None
        and state == "completed"
        and stats.telegram_cards == 0
    ):
        reasons = (stats.rejection_reasons + stats.error_reasons).most_common(3)
        reason_text = ", ".join(f"{name}={count}" for name, count in reasons)
        notification = (
            f"Цикл завершён: найдено {stats.found_results}, "
            f"новых {stats.new_vacancies}, карточек: 0."
            + (f" Причины: {reason_text}." if reason_text else "")
        )
    else:
        notification = ""

    run_id = database.save_search_run(
        started_at=started_at,
        finished_at=now(),
        state=state,
        query_count=stats.query_count,
        found_results=stats.found_results,
        new_vacancies=stats.new_vacancies,
        duplicates=stats.duplicates,
        rejected_by_filter=stats.rejected_by_filter,
        rejected_by_llm=stats.rejected_by_llm,
        telegram_cards=stats.telegram_cards,
        error_count=stats.error_count,
        rejection_reasons=dict(stats.rejection_reasons),
        error_reasons=dict(stats.error_reasons),
        last_safe_error=stats.last_safe_error,
        circuit_reason=circuit_reason,
    )
    if failure is not None:
        raise failure
    if notification:
        try:
            await telegram.notify(notification)
        except TelegramAPIError:
            database.record_search_run_error(run_id, "telegram_error")
    result = database.latest_search_run()
    if result is None:
        raise RuntimeError("search run was not saved")
    logger.info(
        "search_cycle_finished state=%s cards=%s auto_applied=%s errors=%s",
        result.state,
        result.telegram_cards,
        stats.auto_applications,
        result.error_count,
    )
    return result


async def retry_due_analyses(
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    control: AgentControl,
    *,
    now_factory: Callable[[], datetime] | None = None,
) -> None:
    clock = now_factory or (lambda: datetime.now(UTC))
    now = clock()
    # ponytail: one worker owns this lifecycle; add per-row claims if
    # multi-worker deployments matter.
    database.requeue_due_analysis_failures(
        now, max_attempts=ANALYSIS_MAX_ATTEMPTS
    )
    for vacancy in database.unprocessed_discovered():
        if control.paused:
            break
        await process_vacancy(
            VacancySummary(
                vacancy.id,
                vacancy.title,
                vacancy.url,
                vacancy.search_query,
            ),
            settings,
            database,
            hh_client,
            analyzer,
            telegram,
            now_factory=clock,
        )


async def agent_loop(
    settings: Settings,
    database: Database,
    hh_client: HHClient,
    analyzer: VacancyAnalyzer,
    telegram: TelegramService,
    control: AgentControl,
    approval_service: ApprovalService | None = None,
    *,
    now_factory: Callable[[], datetime] | None = None,
    randint: Callable[[int, int], int] = random.randint,
) -> None:
    clock = now_factory or (lambda: datetime.now(UTC))
    next_auto_batch: datetime | None = None
    while True:
        wake_timeout_seconds = settings.check_interval_minutes * 60
        if control.paused:
            control.next_run_at = None
        elif settings.auto_apply.enabled:
            now = clock()
            next_auto_batch = next_auto_batch or restored_auto_batch_at(
                now,
                settings,
                database,
                interval_hours=randint(
                    settings.auto_apply.min_interval_hours,
                    settings.auto_apply.max_interval_hours,
                ),
            )
            if now >= next_auto_batch:
                remaining = database.available_application_slots(
                    daily_limit=settings.max_applications_per_day, now=now
                )
                if remaining:
                    applications_before = database.applied_today(now)
                    batch_size = min(
                        randint(
                            settings.auto_apply.min_batch_size,
                            settings.auto_apply.max_batch_size,
                        ),
                        remaining,
                    )
                    run = await run_search_cycle(
                        settings,
                        database,
                        hh_client,
                        analyzer,
                        telegram,
                        control,
                        approval_service=approval_service,
                        auto_apply_batch_limit=batch_size,
                        now_factory=clock,
                    )
                    applied_in_batch = max(
                        0,
                        database.applied_today(clock()) - applications_before,
                    )
                    retry_at = recoverable_auto_retry_at(
                        control, clock(), settings
                    )
                    if retry_at is not None:
                        next_auto_batch = retry_at
                        control.next_run_at = next_auto_batch
                    elif control.paused:
                        control.next_run_at = None
                    else:
                        next_auto_batch = next_auto_batch_after_run(
                            clock(),
                            settings,
                            sent=applied_in_batch,
                            target=batch_size,
                            interval_hours=randint(
                                settings.auto_apply.min_interval_hours,
                                settings.auto_apply.max_interval_hours,
                            ),
                        )
                        control.next_run_at = next_auto_batch
                    try:
                        await telegram.notify(
                            format_auto_batch_report(
                                run,
                                auto_applied=applied_in_batch,
                                next_batch=control.next_run_at,
                                settings=settings,
                            )
                        )
                    except TelegramAPIError:
                        logger.warning("auto_batch_report_notification_failed")
                else:
                    next_auto_batch = next_auto_window_start_after(now, settings)
                    control.next_run_at = next_auto_batch
            else:
                control.next_run_at = next_auto_batch
            if control.next_run_at is not None:
                wake_timeout_seconds = max(
                    0.0, (control.next_run_at - clock()).total_seconds()
                )
        else:
            run = await run_search_cycle(
                settings, database, hh_client, analyzer, telegram, control
            )
            control.next_run_at = (
                None
                if control.paused
                else datetime.fromisoformat(run.finished_at)
                + timedelta(minutes=settings.check_interval_minutes)
            )
            wake_timeout_seconds = settings.check_interval_minutes * 60
        wake_received = False
        try:
            await asyncio.wait_for(
                control.wake_event.wait(), wake_timeout_seconds
            )
            wake_received = True
        except TimeoutError:
            pass
        finally:
            control.wake_event.clear()
            control.next_run_at = None
        if wake_received:
            next_auto_batch = None


async def run(settings: Settings) -> None:
    database = Database(settings.database_path)
    database.init()
    backend = create_browser_backend(settings)
    llm_provider = create_llm_provider(settings, database)
    mistral_keys = (
        llm_provider if isinstance(llm_provider, MistralKeyManager) else None
    )
    telegram: TelegramService | None = None
    try:
        context = await backend.start()
        guard = ApprovalGuard(settings, database)
        analyzer = VacancyAnalyzer(settings, llm_provider)
        hh_client = HHClient(context, settings, database, guard)
        hh_client.questionnaire_answerer = getattr(
            analyzer, "generate_questionnaire_answers", None
        )
        if not await hh_client.ensure_login():
            raise RuntimeError(
                "HH.ru login is required. Run with BROWSER_HEADLESS=false and sign in manually."
            )
        control = AgentControl()
        approval_service = ApprovalService(settings, database, hh_client)
        telegram = TelegramService(
            settings, database, approval_service, control, mistral_keys=mistral_keys
        )
        if mistral_keys is not None:
            mistral_keys.set_notifier(telegram.notify)
        if hasattr(telegram, "check_updates"):
            asyncio.create_task(telegram.check_updates(notify=True))
        await asyncio.gather(
            telegram.start_polling(),
            agent_loop(
                settings,
                database,
                hh_client,
                analyzer,
                telegram,
                control,
                approval_service,
            ),
        )
    finally:
        try:
            if telegram is not None:
                await telegram.stop()
        finally:
            try:
                await llm_provider.close()
            finally:
                await backend.close()


async def check_llm(
    settings: Settings,
    provider_factory: Callable[[Settings, Database], LLMProvider] = create_llm_provider,
) -> None:
    database = Database(settings.database_path)
    database.init()
    provider = provider_factory(settings, database)
    try:
        response = await provider.generate_text(
            LLMRequest(
                system_instructions="Return only the word OK.",
                user_content="Reply OK.",
                model=settings.llm.model,
                temperature=0,
                max_output_tokens=min(settings.llm.max_output_tokens, 8),
                timeout_seconds=settings.llm.timeout_seconds,
                operation="healthcheck",
            )
        )
        print(
            f"LLM check: provider={response.provider} model={response.model} "
            f"latency_ms={response.latency_ms} success=true"
        )
    finally:
        await provider.close()


def cli(
    argv: list[str] | None = None,
    *,
    provider_factory: Callable[[Settings, Database], LLMProvider] = create_llm_provider,
) -> int:
    parser = argparse.ArgumentParser(description="Safe personal HH assistant")
    parser.add_argument(
        "--version",
        action="version",
        version=f"HH Agent v{__version__}",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--check-llm", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.env_file, args.profile)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.check_config:
        print(
            f"Configuration valid: mode={settings.app_mode}, "
            f"browser={settings.browser_backend}"
        )
        return 0
    if args.check_llm:
        try:
            asyncio.run(check_llm(settings, provider_factory))
        except LLMError as exc:
            print(
                f"LLM check failed: provider={settings.llm.provider} "
                f"error_type={exc.category}",
                file=sys.stderr,
            )
            return 1
        return 0
    configure_logging(settings.log_path)
    lock_path = settings.database_path.with_name(f"{settings.database_path.name}.lock")
    try:
        with single_instance(lock_path):
            asyncio.run(run(settings))
    except AlreadyRunningError as exc:
        logger.info("startup_skipped reason=already_running")
        print(exc, file=sys.stderr)
        return 0
    except (BrowserLaunchError, RuntimeError) as exc:
        logger.error("startup_failed error=%s", exc)
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.info("agent_stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
