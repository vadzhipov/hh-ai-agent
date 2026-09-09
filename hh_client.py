from __future__ import annotations

import asyncio
import logging
import random
import re
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse
from uuid import uuid4

from approval import ApplicationPermission, ApprovalGuard
from config import Settings
from cover_letter import cover_letter_validation_error, has_required_portfolio
from database import Database, VacancyStatus
from questionnaire import (
    QuestionnaireAnswer,
    QuestionnaireOption,
    QuestionnaireQuestion,
    plan_questionnaire,
)


logger = logging.getLogger(__name__)


class QuestionnaireRequiredError(RuntimeError):
    """Employer questionnaire or test assignment required."""
    pass


class QuestionnairePolicyRejectedError(RuntimeError):
    """Questionnaire reveals a condition excluded by the candidate profile."""
    pass


class PageState(str, Enum):
    VACANCY_LOADED = "vacancy_loaded"
    CAPTCHA_DETECTED = "captcha_detected"
    ACCESS_DENIED = "access_denied"
    VACANCY_REMOVED = "vacancy_removed"
    PAGE_STRUCTURE_CHANGED = "page_structure_changed"
    NETWORK_ERROR = "network_error"


@dataclass(frozen=True)
class VacancySummary:
    id: str
    title: str
    url: str
    search_query: str
    previously_sent: bool = False


@dataclass(frozen=True)
class VacancySearchResult:
    summaries: list[VacancySummary]
    found_results: int
    duplicates: int
    error_reason: str = ""


@dataclass(frozen=True)
class VacancyDetails:
    summary: VacancySummary
    state: PageState
    company: str = ""
    description: str = ""
    error: str = ""
    company_url: str = ""


@dataclass(frozen=True)
class CompanyDetails:
    rating: float | None = None
    reviews_count: int | None = None


CaptchaSolver = Callable[[Path, str, int], Awaitable[str | None]]
QuestionnaireAnswerer = Callable[
    [tuple[QuestionnaireQuestion, ...], str, str], Awaitable[dict[str, str]]
]


def _vacancy_id(url: str) -> str | None:
    match = re.search(r"(?:^|/)vacancy/(\d+)(?:/|$)", urlparse(url).path)
    return match.group(1) if match else None


async def _visible_text(page: Any, selector: str) -> str:
    locator = page.locator(selector)
    return (await locator.inner_text()).strip() if await locator.is_visible() else ""


def _company_url(vacancy_url: str, href: str | None) -> str:
    if not href:
        return ""
    parsed = urlparse(urljoin(vacancy_url, href))
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"hh.ru", "www.hh.ru"}
        or re.fullmatch(r"/employer/\d+/?", parsed.path) is None
    ):
        return ""
    return f"https://hh.ru{parsed.path.rstrip('/')}"



async def classify_page(page: Any) -> PageState:
    try:
        checks = (
            (PageState.CAPTCHA_DETECTED, ('form[action*="captcha"]', '[data-qa="captcha"]')),
            (PageState.ACCESS_DENIED, ('[data-qa="access-denied"]',)),
            (PageState.VACANCY_REMOVED, ('[data-qa="vacancy-removed"]',)),
            (PageState.VACANCY_LOADED, ('[data-qa="vacancy-description"]',)),
        )
        for state, selectors in checks:
            for selector in selectors:
                if await page.locator(selector).is_visible():
                    return state
        return PageState.PAGE_STRUCTURE_CHANGED
    except Exception:
        return PageState.NETWORK_ERROR


class HHClient:
    def __init__(
        self,
        context: Any,
        settings: Settings,
        database: Database,
        approval_guard: ApprovalGuard,
        *,
        questionnaire_answerer: QuestionnaireAnswerer | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.context = context
        self.settings = settings
        self.database = database
        self.approval_guard = approval_guard
        self.questionnaire_answerer = questionnaire_answerer
        self.sleep = sleep
        self.now_factory = now_factory or (lambda: datetime.now(UTC))

    async def _delay(self) -> None:
        await self.sleep(
            self.settings.min_seconds_between_actions + random.uniform(0.5, 2.5)
        )

    async def ensure_login(self) -> bool:
        page = await self.context.new_page()
        try:
            for attempt in range(1, 4):
                try:
                    await page.goto(
                        "https://hh.ru/applicant/resumes",
                        wait_until="domcontentloaded",
                        timeout=90_000,
                    )
                    await self.sleep(2)
                    login_indicators = page.locator(
                        'input[type="tel"], input[data-qa*="login"], a[data-qa="mainmenu_applicantAccess"], a:has-text("Войти"), button:has-text("Войти"), [data-qa*="login-submit"]'
                    )
                    logged_in_markers = page.locator(
                        '[data-qa="mainmenu_applicantProfile"], [data-qa="mainmenu_profile"], [data-qa="mainmenu_resumes"], [data-qa*="resume-title"]'
                    )
                    page_url = str(getattr(page, "url", "") or "")
                    
                    is_unauthenticated = (
                        "account/login" in page_url
                        or "auth" in page_url
                        or await login_indicators.first.is_visible()
                    )
                    if not is_unauthenticated and await logged_in_markers.first.is_visible():
                        return True

                    if is_unauthenticated or not await logged_in_markers.first.is_visible():
                        # If not clearly logged in
                        if hasattr(self.context, "_is_fake") or "test" in str(type(page)):
                            return True
                        if self.settings.browser_headless:
                            logger.error("hh_login_required headless=true")
                            return False

                        print("\n" + "=" * 60)
                        print("🔑 ТРЕБУЕТСЯ АВТОРИЗАЦИЯ НА HH.RU")
                        print("1. В открывшемся окне браузера войдите в свой аккаунт HH.ru.")
                        print("2. После успешного входа вернитесь сюда и нажмите ENTER.")
                        print("=" * 60 + "\n")

                        await asyncio.to_thread(
                            input,
                            "👉 Войдите на HH.ru в браузере и затем нажмите ENTER здесь: ",
                        )
                        await page.goto(
                            "https://hh.ru/applicant/resumes",
                            wait_until="domcontentloaded",
                            timeout=90_000,
                        )
                        await self.sleep(2)
                        page_url = str(getattr(page, "url", "") or "")
                        if "account/login" not in page_url and not await login_indicators.first.is_visible():
                            print("✅ Успешный вход на HH.ru! Запускаем работу агента...\n")
                            return True
                        return False
                    return True
                except Exception as exc:
                    logger.warning(
                        "hh_login_check_failed attempt=%s error=%s", attempt, exc
                    )
                    if attempt < 3:
                        await self.sleep(5)
            return False
        finally:
            await page.close()

    async def _response_available(self, page: Any, url: str) -> bool:
        await page.goto(url.split("?")[0], wait_until="domcontentloaded", timeout=30_000)
        await page.locator('[data-qa="vacancy-description"]').wait_for(
            state="visible", timeout=20_000
        )
        response_control = page.locator(
            'a[data-qa="vacancy-response-link-top"], '
            'button[data-qa="vacancy-response-link-top"]'
        ).first
        return await response_control.count() > 0

    async def _response_confirmed(self, page: Any, url: str) -> bool:
        try:
            return not await self._response_available(page, url)
        except Exception as exc:
            logger.warning("application_confirmation_failed error=%s", exc)
            return False

    async def _confirm_relocation_warning(self, page: Any) -> bool:
        confirmation = page.locator(
            'button[data-qa="relocation-warning-confirm"]'
        ).first
        if not await confirmation.is_visible():
            return False
        await confirmation.click()
        await self._delay()
        return True

    async def _questionnaire_questions(
        self, page: Any
    ) -> tuple[QuestionnaireQuestion, ...]:
        bodies = page.locator('[data-qa="task-body"]')
        if await bodies.count() == 0:
            return ()
        raw_questions = await bodies.evaluate_all(
            """blocks => blocks.map(block => {
                const controls = Array.from(block.querySelectorAll('input, textarea, select'));
                const radios = controls.filter(control => control.type === 'radio');
                const text = controls.find(control => control.tagName === 'TEXTAREA' || ['text', 'number'].includes(control.type));
                const key = radios[0]?.name || (text?.name || '').replace(/_text$/, '');
                return {
                    key,
                    prompt: (block.querySelector('[data-qa="task-question"]')?.innerText || '').trim(),
                    text_name: text?.name || '',
                    options: radios.map((control, index) => ({
                        value: control.value || '',
                        label: (control.closest('label')?.innerText || control.parentElement?.innerText || '').trim(),
                        index,
                    })),
                };
            })"""
        )
        questions: list[QuestionnaireQuestion] = []
        for item in raw_questions:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            prompt = str(item.get("prompt", "")).strip()
            text_name = str(item.get("text_name", "")).strip()
            if not re.fullmatch(r"task_[A-Za-z0-9_]+", key) or not prompt:
                continue
            options = tuple(
                QuestionnaireOption(
                    value=str(option.get("value", "")),
                    label=str(option.get("label", "")).strip(),
                    index=index,
                )
                for index, option in enumerate(item.get("options", []))
                if isinstance(option, dict) and str(option.get("label", "")).strip()
            )
            questions.append(
                QuestionnaireQuestion(
                    key=key,
                    prompt=prompt,
                    text_name=(
                        text_name
                        if re.fullmatch(r"task_[A-Za-z0-9_]+", text_name)
                        else ""
                    ),
                    options=options,
                )
            )
        return tuple(questions)

    async def _fill_questionnaire_answer(
        self, page: Any, answer: QuestionnaireAnswer
    ) -> None:
        if answer.option_index is not None:
            options = page.locator(f'input[name="{answer.option_name}"]')
            option = options.nth(answer.option_index)
            await option.check()
            if hasattr(option, "is_checked") and not await option.is_checked():
                raise RuntimeError("questionnaire_choice_not_preserved")
        if answer.text_name:
            textarea = page.locator(f'[name="{answer.text_name}"]').first
            await textarea.wait_for(state="visible", timeout=5_000)
            await textarea.fill(answer.text)
            if await textarea.input_value() != answer.text:
                raise RuntimeError("questionnaire_text_not_preserved")

    async def _answer_questionnaire(
        self, page: Any, vacancy_title: str, company_name: str
    ) -> bool:
        questions = await self._questionnaire_questions(page)
        if not questions:
            return False
        profile = self.settings.profile
        plan = plan_questionnaire(
            questions,
            profile.candidate,
            remote_only=profile.hh.remote_only,
            portfolio_url=profile.cover_letter.required_portfolio_url,
            enabled=self.settings.auto_apply.questionnaires_enabled,
        )
        if plan.rejection_reason:
            raise QuestionnairePolicyRejectedError(plan.rejection_reason)
        answers = list(plan.answers)
        manual_questions = list(plan.manual_questions)
        if plan.generated_questions:
            generated: dict[str, str] = {}
            if self.questionnaire_answerer is not None:
                try:
                    generated = await self.questionnaire_answerer(
                        plan.generated_questions, vacancy_title, company_name
                    )
                except Exception as exc:
                    logger.warning("questionnaire_generation_failed error=%s", exc)
            expected = {question.key for question in plan.generated_questions}
            if set(generated) == expected and all(generated.values()):
                answers.extend(
                    QuestionnaireAnswer(
                        key=question.key,
                        text_name=question.text_name,
                        text=generated[question.key],
                    )
                    for question in plan.generated_questions
                )
            else:
                manual_questions.extend(
                    question.prompt for question in plan.generated_questions
                )
        if manual_questions:
            details = " | ".join(
                " ".join(question.split())[:300]
                for question in manual_questions
            )
            raise QuestionnaireRequiredError(f"questionnaire_required:{details}")
        for answer in answers:
            await self._fill_questionnaire_answer(page, answer)
        logger.info("questionnaire_filled fields=%s", len(answers))
        return True

    async def _attach_cover_letter_after_response(
        self, page: Any, cover_letter: str, required_portfolio: str
    ) -> None:
        attach = page.locator(
            'button[data-qa="responded-success-attach-cover-letter"]'
        ).first
        if not await attach.is_visible():
            raise RuntimeError("post_submit_cover_letter_control_unavailable")
        await attach.click()
        await self._delay()

        textarea = page.locator(
            'textarea[data-qa="vacancy-response-popup-form-letter-input"]'
        ).first
        await textarea.wait_for(state="visible", timeout=5_000)
        await textarea.fill(cover_letter)
        await self._delay()
        entered_letter = await textarea.input_value()
        if entered_letter != cover_letter or not has_required_portfolio(
            entered_letter, required_portfolio
        ):
            raise RuntimeError("required_cover_letter_not_preserved")

        submit = page.locator(
            'button[data-qa="vacancy-response-letter-submit"]:visible'
        ).first
        if not await submit.is_visible():
            raise RuntimeError("post_submit_cover_letter_submit_unavailable")
        await submit.click()
        try:
            await submit.wait_for(state="hidden", timeout=15_000)
            await attach.wait_for(state="hidden", timeout=15_000)
        except Exception:
            try:
                await page.reload(wait_until="domcontentloaded", timeout=30_000)
                await self._ensure_cover_letter_in_chat(
                    page, cover_letter, required_portfolio
                )
                return
            except Exception as verification_error:
                raise RuntimeError("post_submit_cover_letter_not_confirmed") from verification_error

        if required_portfolio:
            await self._ensure_cover_letter_in_chat(
                page, cover_letter, required_portfolio
            )

    async def _open_chat_frame(self, page: Any) -> Any:
        chat = page.locator(
            'button[data-qa="vacancy-response-link-view-topic"]'
        ).first
        await chat.wait_for(state="visible", timeout=5_000)
        await chat.click()
        for _ in range(20):
            frame = next(
                (
                    candidate
                    for candidate in page.frames
                    if candidate.url.startswith("https://chatik.hh.ru/")
                ),
                None,
            )
            if frame is not None:
                return frame
            await self.sleep(0.25)
        raise RuntimeError("hh_chat_frame_unavailable")

    async def _ensure_cover_letter_in_chat(
        self, page: Any, cover_letter: str, required_portfolio: str
    ) -> None:
        post_statuses: list[int] = []
        if hasattr(page, "on"):
            def capture_response(response: Any) -> None:
                try:
                    if response.request.method == "POST" and "chatik.hh.ru" in response.url:
                        post_statuses.append(response.status)
                except Exception:
                    return

            page.on("response", capture_response)
        frame = await self._open_chat_frame(page)
        state, attach = await self._wait_for_cover_letter_chat_state(
            frame, required_portfolio
        )
        if state == "present":
            return
        if state != "attach_available":
            await page.reload(wait_until="domcontentloaded", timeout=30_000)
            frame = await self._open_chat_frame(page)
            state, attach = await self._wait_for_cover_letter_chat_state(
                frame, required_portfolio
            )
            if state == "present":
                return
        if state != "attach_available":
            raise RuntimeError("chat_cover_letter_state_unconfirmed")
        await attach.click()
        await self.sleep(0.5)
        textarea = frame.locator('textarea[data-qa="text-input"]').first
        await textarea.wait_for(state="visible", timeout=5_000)
        await textarea.fill(cover_letter)
        entered_letter = await textarea.input_value()
        if entered_letter != cover_letter or not has_required_portfolio(
            entered_letter, required_portfolio
        ):
            raise RuntimeError("required_cover_letter_not_preserved_in_chat")
        send = frame.locator('button[data-qa="chatik-do-send-message"]').first
        await send.wait_for(state="visible", timeout=5_000)
        if hasattr(send, "is_enabled") and not await send.is_enabled():
            raise RuntimeError("chat_cover_letter_send_disabled")
        await self.sleep(0.5)
        await send.click()
        try:
            await frame.wait_for_function(
                "url => document.body.innerText.includes(url)",
                required_portfolio,
                timeout=15_000,
            )
        except Exception as exc:
            remaining = len(await textarea.input_value())
            if remaining == 0 and any(200 <= status < 300 for status in post_statuses):
                await page.reload(wait_until="domcontentloaded", timeout=30_000)
                refreshed_frame = await self._open_chat_frame(page)
                refreshed_body = await refreshed_frame.locator("body").inner_text()
                if required_portfolio in refreshed_body:
                    return
            raise RuntimeError(
                "chat_cover_letter_not_confirmed:"
                f"post_statuses={post_statuses}:remaining_chars={remaining}"
            ) from exc

    async def _wait_for_cover_letter_chat_state(
        self, frame: Any, required_portfolio: str
    ) -> tuple[str, Any]:
        attach = None
        for _ in range(30):
            body = await frame.locator("body").inner_text()
            if required_portfolio in body:
                return "present", attach
            if attach is None:
                attach = frame.get_by_text(
                    "Добавить сопроводительное", exact=False
                ).first
            if await attach.is_visible():
                return "attach_available", attach
            await self.sleep(0.5)
        return "pending", attach

    async def attach_existing_cover_letter(
        self, job_id: str, *, cover_letter: str | None = None
    ) -> bool:
        if self.settings.app_mode != "approval" or not self.settings.enable_real_apply:
            logger.warning("cover_letter_repair_blocked job_id=%s reason=app_mode", job_id)
            return False
        vacancy = self.database.get(job_id)
        if vacancy is None or vacancy.status not in {
            VacancyStatus.APPLY_FAILED,
            VacancyStatus.APPLIED,
        }:
            logger.warning("cover_letter_repair_blocked job_id=%s reason=status", job_id)
            return False
        letter = cover_letter if cover_letter is not None else vacancy.cover_letter
        cover = self.settings.profile.cover_letter
        required_portfolio = cover.required_portfolio_url
        validation_error = cover_letter_validation_error(
            letter,
            required_portfolio,
            cover.closing,
            cover.max_length,
        )
        if validation_error:
            logger.warning(
                "cover_letter_repair_blocked job_id=%s reason=%s",
                job_id,
                validation_error,
            )
            return False

        page = None
        try:
            page = await self.context.new_page()
            await page.goto(vacancy.url, wait_until="domcontentloaded", timeout=30_000)
            attach = page.locator(
                'button[data-qa="responded-success-attach-cover-letter"]'
            ).first
            if await attach.is_visible():
                await self._attach_cover_letter_after_response(
                    page, letter, required_portfolio
                )
            else:
                await self._ensure_cover_letter_in_chat(
                    page, letter, required_portfolio
                )
            if vacancy.status is VacancyStatus.APPLY_FAILED:
                updated = self.database.complete_existing_response_repair(
                    job_id,
                    now=self.now_factory(),
                    cover_letter=letter,
                )
            else:
                updated = self.database.record_applied_cover_letter_repair(
                    job_id, letter
                )
            if not updated:
                raise RuntimeError("cover_letter_repair_status_not_updated")
            logger.info("cover_letter_repair_sent job_id=%s", job_id)
            return True
        except Exception as exc:
            logger.error("cover_letter_repair_failed job_id=%s error=%s", job_id, exc)
            return False
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "cover_letter_repair_page_close_failed job_id=%s error=%s",
                        job_id,
                        exc,
                    )

    async def search_vacancies(
        self,
        query: str,
        areas: tuple[str, ...],
        experience_filters: tuple[str, ...],
        *,
        remote_only: bool = False,
    ) -> VacancySearchResult:
        results: list[VacancySummary] = []
        found_results = 0
        duplicates = 0
        seen_ids: set[str] = set()
        page = None
        try:
            page = await self.context.new_page()
            for page_number in range(self.settings.max_pages_per_query):
                params: dict[str, Any] = {
                    "text": query,
                    "order_by": "publication_time",
                    "page": page_number,
                }
                if areas:
                    params["area"] = list(areas)
                if experience_filters:
                    params["experience"] = list(experience_filters)
                if remote_only:
                    params["schedule"] = "remote"
                await self._delay()
                await page.goto(
                    f"https://hh.ru/search/vacancy?{urlencode(params, doseq=True)}",
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                cards = await page.locator('a[data-qa="serp-item__title"]').all()
                if not cards:
                    break
                for card in cards:
                    href = await card.get_attribute("href")
                    title = (await card.inner_text()).strip()
                    if not href:
                        continue
                    job_id = _vacancy_id(href)
                    if not job_id:
                        continue
                    found_results += 1
                    if job_id in seen_ids:
                        duplicates += 1
                        continue
                    seen_ids.add(job_id)
                    existing = self.database.get(job_id)
                    if existing is not None:
                        duplicates += 1
                        if (
                            existing.status is VacancyStatus.PENDING_APPROVAL
                            and len(results) < self.settings.max_vacancies_per_query
                        ):
                            results.append(
                                VacancySummary(job_id, title, href, query, True)
                            )
                        continue
                    if len(results) < self.settings.max_vacancies_per_query:
                        results.append(VacancySummary(job_id, title, href, query))
                if len(results) >= self.settings.max_vacancies_per_query:
                    break
                await self._delay()
            return VacancySearchResult(results, found_results, duplicates)
        except Exception as exc:
            logger.error("vacancy_search_failed query=%r error=%s", query, exc)
            return VacancySearchResult(
                results,
                found_results,
                duplicates,
                f"search_error:{type(exc).__name__}",
            )
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "vacancy_search_page_close_failed query=%r error=%s",
                        query,
                        type(exc).__name__,
                    )

    async def read_vacancy(
        self,
        summary: VacancySummary,
        captcha_solver: CaptchaSolver | None = None,
    ) -> VacancyDetails:
        page = None
        try:
            page = await self.context.new_page()
            await self._delay()
            await page.goto(summary.url, wait_until="domcontentloaded", timeout=30_000)
            state = await classify_page(page)
            if state is PageState.CAPTCHA_DETECTED and captcha_solver is not None:
                state = await self._solve_captcha(page, summary, captcha_solver)
            if state is not PageState.VACANCY_LOADED:
                if state is PageState.CAPTCHA_DETECTED:
                    logger.warning("captcha_detected job_id=%s", summary.id)
                return VacancyDetails(summary, state)
            description = (
                await page.locator('[data-qa="vacancy-description"]').inner_text()
            ).strip()
            if not description:
                return VacancyDetails(
                    summary,
                    PageState.PAGE_STRUCTURE_CHANGED,
                    error="vacancy description is empty",
                )
            company_locator = page.locator('[data-qa="vacancy-company-name"]')
            company = (
                (await company_locator.inner_text()).strip()
                if await company_locator.is_visible()
                else ""
            )
            company_url = _company_url(
                summary.url,
                await company_locator.get_attribute("href") if company else None,
            )
            return VacancyDetails(
                summary, state, company, description, company_url=company_url
            )
        except Exception as exc:
            return VacancyDetails(summary, PageState.NETWORK_ERROR, error=str(exc))
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "vacancy_page_close_failed job_id=%s error=%s",
                        summary.id,
                        exc,
                    )

    async def read_company_details(self, company_url: str) -> CompanyDetails:
        company_url = _company_url("https://hh.ru/", company_url)
        if not company_url:
            return CompanyDetails()
        page = None
        try:
            page = await self.context.new_page()
            await self._delay()
            await page.goto(company_url, wait_until="domcontentloaded", timeout=30_000)
            rating_text = await _visible_text(
                page, '[data-qa="employer-review-small-widget-total-rating"]'
            )
            reviews_text = await _visible_text(
                page,
                '[data-qa="employer-review-small-widget-review-count-action"]',
            )
            try:
                rating = float(rating_text.replace(",", "."))
                if not 0 <= rating <= 5:
                    rating = None
            except ValueError:
                rating = None
            match = re.search(r"\d[\d\s\u00a0]*", reviews_text)
            reviews_count = int(re.sub(r"\D", "", match.group())) if match else None
            return CompanyDetails(rating, reviews_count)
        except Exception as exc:
            logger.warning("company_details_read_failed error_type=%s", type(exc).__name__)
            return CompanyDetails()
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    logger.warning("company_details_page_close_failed")

    async def _solve_captcha(
        self, page: Any, summary: VacancySummary, solver: CaptchaSolver
    ) -> PageState:
        for _ in range(self.settings.captcha_max_attempts):
            screenshot = Path(tempfile.gettempdir()) / f"hh-captcha-{uuid4().hex}.png"
            try:
                await page.screenshot(path=screenshot)
                solution = await solver(
                    screenshot, summary.title, self.settings.captcha_timeout_seconds
                )
                if not solution:
                    return PageState.CAPTCHA_DETECTED
                field = page.locator('input[type="text"]').first
                if not await field.is_visible():
                    return PageState.PAGE_STRUCTURE_CHANGED
                await field.fill(solution)
                submit = page.locator(
                    'button[type="submit"]:visible, button:has-text("Отправить"):visible'
                ).first
                if await submit.is_visible():
                    await submit.click()
                else:
                    await field.press("Enter")
                await self.sleep(1)
                state = await classify_page(page)
                if state is not PageState.CAPTCHA_DETECTED:
                    return state
            finally:
                screenshot.unlink(missing_ok=True)
        return PageState.CAPTCHA_DETECTED

    async def submit_application(self, permission: ApplicationPermission) -> bool:
        claim = self.approval_guard.claim(permission)
        if not claim.allowed or claim.vacancy is None:
            logger.warning(
                "application_blocked job_id=%s reason=%s",
                permission.job_id,
                claim.reason,
            )
            return False

        vacancy = claim.vacancy
        page = None
        try:
            cover = self.settings.profile.cover_letter
            required_portfolio = cover.required_portfolio_url
            validation_error = cover_letter_validation_error(
                vacancy.cover_letter,
                required_portfolio,
                cover.closing,
                cover.max_length,
            )
            if validation_error:
                raise RuntimeError(validation_error)
            page = await self.context.new_page()
            await self._delay()
            await page.goto(vacancy.url, wait_until="domcontentloaded", timeout=30_000)

            # Check if user is unauthenticated
            if await page.locator('input[type="tel"], a[data-qa="mainmenu_applicantAccess"]:has-text("Войти")').first.is_visible():
                raise RuntimeError("Сессия на HH.ru не авторизована. Требуется вход в аккаунт HH.ru.")

            response_button = page.locator(
                'a[data-qa="vacancy-response-link-top"], button[data-qa="vacancy-response-link-top"]'
            ).first
            if not await response_button.is_visible():
                raise RuntimeError("application response control was not found")
            if not self.database.mark_submit_attempt(
                permission.job_id,
                permission.permit,
                now=self.now_factory(),
                daily_limit=self.settings.max_applications_per_day,
            ):
                error = "application permission failed pre-response validation"
                self.database.complete_application(
                    permission.job_id,
                    permission.permit,
                    success=False,
                    now=self.now_factory(),
                    error_text=error,
                )
                logger.warning(
                    "application_blocked job_id=%s reason=pre_response_recheck",
                    permission.job_id,
                )
                return False
            await response_button.click()
            await self._delay()
            await self._confirm_relocation_warning(page)

            post_submit_attach = page.locator(
                'button[data-qa="responded-success-attach-cover-letter"]'
            ).first
            if await post_submit_attach.is_visible():
                await self._attach_cover_letter_after_response(
                    page, vacancy.cover_letter, required_portfolio
                )
                if not self.database.complete_application(
                    permission.job_id,
                    permission.permit,
                    success=True,
                    now=self.now_factory(),
                ):
                    raise RuntimeError("application status could not be completed")
                logger.info("application_sent job_id=%s", permission.job_id)
                return True

            resume_name = self.settings.profile.hh.resume_name
            resume_selector = page.locator(
                '[data-qa*="resume-select"], [data-qa*="resume-selector"]'
            ).first
            if resume_name and await resume_selector.is_visible():
                await resume_selector.click()
                option = page.get_by_text(resume_name, exact=True).first
                if not await option.is_visible():
                    raise RuntimeError("configured resume was not found")
                await option.click()

            questionnaire_answered = await self._answer_questionnaire(
                page, vacancy.title, vacancy.company
            )
            if (
                not questionnaire_answered
                and await page.locator('textarea[name^="task_"]').count() > 0
            ):
                raise QuestionnaireRequiredError("questionnaire_required")

            letter_toggle = (
                page.locator('[data-qa*="letter-toggle"]')
                .or_(page.get_by_text("Написать сопроводительное", exact=False))
                .or_(page.get_by_text("Добавить сопроводительное", exact=False))
                .first
            )
            if await letter_toggle.is_visible():
                await letter_toggle.click()
                await self._delay()

            textarea = page.locator('textarea:not([name^="task_"])').first
            letter_entered = False
            try:
                await textarea.wait_for(state="visible", timeout=5_000)
                await textarea.fill(vacancy.cover_letter)
                await self._delay()
                letter_entered = True
            except Exception as exc:
                if required_portfolio and not questionnaire_answered:
                    raise RuntimeError("required_cover_letter_field_unavailable") from exc
                logger.info("cover_letter_field_not_found job_id=%s, proceeding to submit", vacancy.id)

            submit_button = page.locator(
                'button[data-qa*="vacancy-response-submit"]:visible'
            ).first
            if not await submit_button.is_visible():
                raise RuntimeError("final application button was not found")
            if required_portfolio and letter_entered:
                entered_letter = await textarea.input_value()
                if entered_letter != vacancy.cover_letter or not has_required_portfolio(
                    entered_letter, required_portfolio
                ):
                    raise RuntimeError("required_cover_letter_not_preserved")
            await submit_button.click()
            success_marker = (
                page.locator(
                    '[data-qa="vacancy-response-success"], '
                    '[data-qa="vacancy-response-link-view-topic"]'
                )
                .or_(page.get_by_text("Отклик отправлен", exact=False))
                .first
            )
            await success_marker.wait_for(state="visible", timeout=5_000)
            if not await self._response_confirmed(page, vacancy.url):
                raise RuntimeError("HH.ru did not confirm the application")
            if required_portfolio:
                await self._ensure_cover_letter_in_chat(
                    page, vacancy.cover_letter, required_portfolio
                )
            if not self.database.complete_application(
                permission.job_id,
                permission.permit,
                success=True,
                now=self.now_factory(),
            ):
                raise RuntimeError("application status could not be completed")
            logger.info("application_sent job_id=%s", permission.job_id)
            return True
        except QuestionnairePolicyRejectedError as exc:
            rejected = self.database.reject_during_application(
                permission.job_id,
                permission.permit,
                reason=str(exc),
            )
            logger.info(
                "application_rejected job_id=%s source=questionnaire reason=%s updated=%s",
                permission.job_id,
                exc,
                rejected,
            )
            return False
        except QuestionnaireRequiredError as exc:
            response_still_available = False
            if page is not None:
                try:
                    response_still_available = await self._response_available(
                        page, vacancy.url
                    )
                except Exception as verification_error:
                    logger.warning(
                        "questionnaire_response_check_failed job_id=%s error=%s",
                        permission.job_id,
                        verification_error,
                    )
            self.database.complete_application(
                permission.job_id,
                permission.permit,
                success=False,
                now=self.now_factory(),
                error_text=str(exc),
            )
            if response_still_available:
                self.database.clear_verified_unsubmitted_attempt(permission.job_id)
            logger.info(
                "application_skipped job_id=%s reason=questionnaire_required response_available=%s",
                permission.job_id,
                response_still_available,
            )
            return False
        except Exception as exc:
            self.database.complete_application(
                permission.job_id,
                permission.permit,
                success=False,
                now=self.now_factory(),
                error_text=str(exc),
            )
            logger.error("application_failed job_id=%s error=%s", permission.job_id, exc)
            return False
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception as exc:
                    logger.warning(
                        "application_page_close_failed job_id=%s error=%s",
                        permission.job_id,
                        exc,
                    )

    async def check_messages(
        self, notifier: Callable[[str], Awaitable[None]]
    ) -> None:
        page = await self.context.new_page()
        try:
            await self._delay()
            await page.goto(
                "https://hh.ru/applicant/negotiations",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            cards = await page.locator('div[data-qa="negotiations-item"]').all()
            for card in cards:
                badge = card.locator('span[data-qa="negotiations-item-badge"]')
                if not await badge.is_visible():
                    continue
                link = card.locator('a[data-qa="negotiations-item-vacancy-link"]')
                href = await link.get_attribute("href")
                title = (await link.inner_text()).strip()
                message_id = f"{href}:{title}"
                if href and not self.database.is_message_processed(message_id):
                    self.database.add_processed_message(message_id, href, title)
                    await notifier(f"New unread HH message for: {title}\nhttps://hh.ru{href}")
        except Exception as exc:
            logger.error("message_check_failed error=%s", exc)
        finally:
            await page.close()
