import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from approval import ApplicationPermission, ApprovalGuard, ApprovalService
from config import load_settings
from database import Database, VacancyStatus
from hh_client import HHClient
from tests.test_config import VALID_ENV, write_profile


NOW = datetime(2026, 7, 26, 9, 0, tzinfo=UTC)


class NeverPageContext:
    async def new_page(self):
        raise AssertionError("browser must not be touched for a blocked application")


class FakeLocator:
    def __init__(self, visible: bool = True, clicks: list[str] | None = None, name: str = ""):
        self.visible = visible
        self.clicks = clicks
        self.name = name
        self.first = self
        self.filled = ""

    async def is_visible(self) -> bool:
        return self.visible

    async def click(self) -> None:
        if self.clicks is not None:
            self.clicks.append(self.name)

    async def wait_for(self, **kwargs) -> None:
        if not self.visible:
            raise RuntimeError("not visible")

    async def fill(self, value: str) -> None:
        self.filled = value

    async def input_value(self) -> str:
        return self.filled

    async def count(self) -> int:
        return int(self.visible)

    def or_(self, other: "FakeLocator") -> "FakeLocator":
        return self if self.visible else other


class FakeApplicationPage:
    def __init__(
        self,
        fail_navigation: bool = False,
        success_visible: bool = True,
        employer_questions: bool = False,
        response_still_available: bool = False,
    ):
        self.fail_navigation = fail_navigation
        self.success_visible = success_visible
        self.employer_questions = employer_questions
        self.response_still_available = response_still_available
        self.clicks: list[str] = []
        self.selectors: list[str] = []
        self.closed = False
        self.textarea = FakeLocator(True)
        self.frames = [ConfirmedChatFrame()]

    async def goto(self, url: str, **kwargs) -> None:
        if self.fail_navigation:
            raise RuntimeError("navigation failed")

    def locator(self, selector: str) -> FakeLocator:
        self.selectors.append(selector)
        if selector == 'button[data-qa="vacancy-response-link-view-topic"]':
            return FakeLocator(True, self.clicks, "open_chat")
        if selector == '[data-qa="vacancy-description"]':
            return FakeLocator(True)
        if selector == 'textarea[name^="task_"]':
            return FakeLocator(self.employer_questions)
        if "vacancy-response-success" in selector:
            return FakeLocator(self.success_visible)
        if "vacancy-response-link" in selector:
            submitted = "final_submit" in self.clicks
            visible = not submitted or self.response_still_available
            return FakeLocator(visible, self.clicks, "open_response")
        if "resume-select" in selector:
            return FakeLocator(False)
        if "letter-toggle" in selector or "сопроводительное" in selector:
            return FakeLocator(False)
        if selector in {"textarea", 'textarea:not([name^="task_"])'}:
            return self.textarea
        if "vacancy-response-submit" in selector:
            return FakeLocator(True, self.clicks, "final_submit")
        return FakeLocator(False)

    def get_by_text(self, text: str, *, exact: bool) -> FakeLocator:
        return FakeLocator(
            text == "Отклик отправлен" and self.success_visible
        )

    async def close(self) -> None:
        self.closed = True


class FakeApplicationContext:
    def __init__(self, page: FakeApplicationPage):
        self.page = page

    async def new_page(self) -> FakeApplicationPage:
        return self.page


class ConfirmedChatBody:
    async def inner_text(self) -> str:
        return PORTFOLIO


class ConfirmedChatFrame:
    url = "https://chatik.hh.ru/chat/confirmed"

    def locator(self, selector: str) -> ConfirmedChatBody:
        assert selector == "body"
        return ConfirmedChatBody()


class FailingApplicationContext:
    async def new_page(self):
        raise RuntimeError("browser page creation failed")


async def no_sleep(_: float) -> None:
    return None


PORTFOLIO = "https://portfolio.example/candidate"


def require_portfolio(app_settings):
    return replace(app_settings, profile=replace(
        app_settings.profile, cover_letter=replace(
            app_settings.profile.cover_letter, required_portfolio_url=PORTFOLIO
        ),
    ))


def settings(tmp_path: Path, **environment: str):
    loaded = load_settings(
        profile_path=write_profile(tmp_path),
        environ={**VALID_ENV, **environment},
    )
    return replace(loaded, database_path=tmp_path / "agent.db")


def pending(
    database: Database,
    job_id: str = "job-1",
    letter: str = "Letter",
    confidence: float = 0.9,
) -> None:
    assert database.discover(
        job_id=job_id,
        title="Python developer",
        company="Example",
        url=f"https://example.com/vacancy/{job_id}",
        description_hash="hash",
        search_query="Python",
        discovered_at=NOW,
    )
    assert database.request_approval(
        job_id=job_id,
        cover_letter=letter,
        llm_decision=True,
        llm_reason="Relevant",
        confidence=confidence,
        now=NOW,
    )


@pytest.mark.parametrize("backend", ["cloakbrowser", "playwright"])
def test_dry_run_blocks_low_level_send_independently_of_backend(
    tmp_path: Path, backend: str
) -> None:
    app_settings = settings(tmp_path, BROWSER_BACKEND=backend, APP_MODE="dry_run")
    database = Database(app_settings.database_path)
    database.init()
    pending(database)
    token = database.approve("job-1", 42, 42, NOW + timedelta(minutes=1))
    assert token
    client = HHClient(
        NeverPageContext(),
        app_settings,
        database,
        ApprovalGuard(app_settings, database, now_factory=lambda: NOW + timedelta(minutes=2)),
        sleep=no_sleep,
    )

    sent = asyncio.run(
        client.submit_application(ApplicationPermission("job-1", token, 42))
    )

    assert not sent
    assert database.get("job-1").status is VacancyStatus.APPROVED


def test_direct_low_level_call_without_individual_permission_is_blocked(
    tmp_path: Path,
) -> None:
    app_settings = settings(
        tmp_path, APP_MODE="approval", ENABLE_REAL_APPLY="true"
    )
    database = Database(app_settings.database_path)
    database.init()
    pending(database)
    client = HHClient(
        NeverPageContext(),
        app_settings,
        database,
        ApprovalGuard(app_settings, database, now_factory=lambda: NOW + timedelta(minutes=1)),
        sleep=no_sleep,
    )

    sent = asyncio.run(
        client.submit_application(ApplicationPermission("job-1", "forged", 42))
    )

    assert not sent
    assert database.get("job-1").status is VacancyStatus.PENDING_APPROVAL


def test_approval_service_is_the_valid_path_to_physical_submit(tmp_path: Path) -> None:
    app_settings = settings(
        tmp_path, APP_MODE="approval", ENABLE_REAL_APPLY="true", TG_USER_ID="42"
    )
    database = Database(app_settings.database_path)
    database.init()
    pending(database)
    page = FakeApplicationPage()
    client = HHClient(
        FakeApplicationContext(page),
        app_settings,
        database,
        ApprovalGuard(app_settings, database, now_factory=lambda: NOW + timedelta(minutes=2)),
        sleep=no_sleep,
        now_factory=lambda: NOW + timedelta(minutes=3),
    )
    service = ApprovalService(
        app_settings,
        database,
        client,
        now_factory=lambda: NOW + timedelta(minutes=1),
    )

    result = asyncio.run(service.approve_and_apply("job-1", 42))

    assert result.ok
    assert page.clicks == ["open_response", "final_submit"]
    assert all("text=" not in selector for selector in page.selectors)
    assert database.get("job-1").status is VacancyStatus.APPLIED


def test_browser_error_after_claim_becomes_apply_failed(tmp_path: Path) -> None:
    app_settings = settings(
        tmp_path, APP_MODE="approval", ENABLE_REAL_APPLY="true", TG_USER_ID="42"
    )
    database = Database(app_settings.database_path)
    database.init()
    pending(database)
    token = database.approve("job-1", 42, 42, NOW + timedelta(minutes=1))
    assert token
    page = FakeApplicationPage(fail_navigation=True)
    client = HHClient(
        FakeApplicationContext(page),
        app_settings,
        database,
        ApprovalGuard(app_settings, database, now_factory=lambda: NOW + timedelta(minutes=2)),
        sleep=no_sleep,
        now_factory=lambda: NOW + timedelta(minutes=3),
    )

    sent = asyncio.run(
        client.submit_application(ApplicationPermission("job-1", token, 42))
    )

    vacancy = database.get("job-1")
    assert not sent
    assert vacancy.status is VacancyStatus.APPLY_FAILED
    assert "navigation failed" in vacancy.error_text


def test_page_creation_error_after_claim_becomes_apply_failed(tmp_path: Path) -> None:
    app_settings = settings(
        tmp_path, APP_MODE="approval", ENABLE_REAL_APPLY="true", TG_USER_ID="42"
    )
    database = Database(app_settings.database_path)
    database.init()
    pending(database)
    token = database.approve("job-1", 42, 42, NOW + timedelta(minutes=1))
    assert token
    client = HHClient(
        FailingApplicationContext(),
        app_settings,
        database,
        ApprovalGuard(app_settings, database, now_factory=lambda: NOW + timedelta(minutes=2)),
        sleep=no_sleep,
        now_factory=lambda: NOW + timedelta(minutes=3),
    )

    sent = asyncio.run(
        client.submit_application(ApplicationPermission("job-1", token, 42))
    )

    vacancy = database.get("job-1")
    assert not sent
    assert vacancy.status is VacancyStatus.APPLY_FAILED
    assert "page creation failed" in vacancy.error_text


def test_missing_explicit_success_signal_fails_closed(tmp_path: Path) -> None:
    app_settings = settings(
        tmp_path, APP_MODE="approval", ENABLE_REAL_APPLY="true", TG_USER_ID="42"
    )
    database = Database(app_settings.database_path)
    database.init()
    pending(database)
    token = database.approve("job-1", 42, 42, NOW + timedelta(minutes=1))
    assert token
    page = FakeApplicationPage(success_visible=False)
    client = HHClient(
        FakeApplicationContext(page),
        app_settings,
        database,
        ApprovalGuard(app_settings, database, now_factory=lambda: NOW + timedelta(minutes=2)),
        sleep=no_sleep,
        now_factory=lambda: NOW + timedelta(minutes=3),
    )

    sent = asyncio.run(
        client.submit_application(ApplicationPermission("job-1", token, 42))
    )

    assert not sent
    assert page.clicks == ["open_response", "final_submit"]
    assert database.get("job-1").status is VacancyStatus.APPLY_FAILED


def test_employer_questionnaire_blocks_submit(tmp_path: Path) -> None:
    app_settings = settings(
        tmp_path, APP_MODE="approval", ENABLE_REAL_APPLY="true", TG_USER_ID="42"
    )
    database = Database(app_settings.database_path)
    database.init()
    pending(database)
    token = database.approve("job-1", 42, 42, NOW + timedelta(minutes=1))
    assert token
    page = FakeApplicationPage(employer_questions=True)
    client = HHClient(
        FakeApplicationContext(page),
        app_settings,
        database,
        ApprovalGuard(app_settings, database, now_factory=lambda: NOW + timedelta(minutes=2)),
        sleep=no_sleep,
        now_factory=lambda: NOW + timedelta(minutes=3),
    )

    sent = asyncio.run(
        client.submit_application(ApplicationPermission("job-1", token, 42))
    )

    assert not sent
    assert page.clicks == ["open_response"]
    vacancy = database.get("job-1")
    assert vacancy.status is VacancyStatus.APPLY_FAILED
    assert vacancy.error_text == "questionnaire_required"
    assert vacancy.submit_attempted_at is None
    assert database.available_application_slots(
        daily_limit=5, now=NOW + timedelta(minutes=4)
    ) == 5


def test_success_marker_without_hh_confirmation_fails_closed(tmp_path: Path) -> None:
    app_settings = settings(
        tmp_path, APP_MODE="approval", ENABLE_REAL_APPLY="true", TG_USER_ID="42"
    )
    database = Database(app_settings.database_path)
    database.init()
    pending(database)
    token = database.approve("job-1", 42, 42, NOW + timedelta(minutes=1))
    assert token
    page = FakeApplicationPage(response_still_available=True)
    client = HHClient(
        FakeApplicationContext(page),
        app_settings,
        database,
        ApprovalGuard(app_settings, database, now_factory=lambda: NOW + timedelta(minutes=2)),
        sleep=no_sleep,
        now_factory=lambda: NOW + timedelta(minutes=3),
    )

    sent = asyncio.run(
        client.submit_application(ApplicationPermission("job-1", token, 42))
    )

    assert not sent
    assert page.clicks == ["open_response", "final_submit"]
    assert database.get("job-1").status is VacancyStatus.APPLY_FAILED


def test_permission_is_rechecked_before_first_response_action(
    tmp_path: Path,
) -> None:
    app_settings = settings(
        tmp_path, APP_MODE="approval", ENABLE_REAL_APPLY="true", TG_USER_ID="42"
    )
    database = Database(app_settings.database_path)
    database.init()
    pending(database)
    token = database.approve("job-1", 42, 42, NOW + timedelta(minutes=29))
    assert token
    page = FakeApplicationPage()
    client = HHClient(
        FakeApplicationContext(page),
        app_settings,
        database,
        ApprovalGuard(app_settings, database, now_factory=lambda: NOW + timedelta(minutes=29)),
        sleep=no_sleep,
        now_factory=lambda: NOW + timedelta(minutes=60),
    )

    sent = asyncio.run(
        client.submit_application(ApplicationPermission("job-1", token, 42))
    )

    assert not sent
    assert page.clicks == []
    assert database.get("job-1").status is VacancyStatus.EXPIRED


def test_action_delay_never_drops_below_configured_minimum(tmp_path: Path) -> None:
    app_settings = settings(tmp_path)
    database = Database(app_settings.database_path)
    database.init()
    delays: list[float] = []

    async def capture_delay(seconds: float) -> None:
        delays.append(seconds)

    client = HHClient(
        NeverPageContext(),
        app_settings,
        database,
        ApprovalGuard(app_settings, database),
        sleep=capture_delay,
    )

    asyncio.run(client._delay())

    assert len(delays) == 1
    assert delays[0] >= app_settings.min_seconds_between_actions


def test_missing_portfolio_blocks_approval_without_touching_browser(tmp_path: Path) -> None:
    app_settings = require_portfolio(settings(tmp_path, APP_MODE="approval", ENABLE_REAL_APPLY="true", TG_USER_ID="42"))
    database = Database(app_settings.database_path)
    database.init()
    pending(database, letter="Letter without portfolio")
    client = HHClient(NeverPageContext(), app_settings, database, ApprovalGuard(app_settings, database))
    service = ApprovalService(app_settings, database, client, now_factory=lambda: NOW + timedelta(minutes=1))

    result = asyncio.run(service.approve_and_apply("job-1", 42))

    assert not result.ok
    assert "портфолио" in result.message
    assert database.get("job-1").status is VacancyStatus.PENDING_APPROVAL


def test_auto_apply_reuses_the_guarded_submission_path(tmp_path: Path) -> None:
    app_settings = settings(
        tmp_path,
        APP_MODE="approval",
        ENABLE_REAL_APPLY="true",
        TG_USER_ID="42",
        AUTO_APPLY_ENABLED="true",
        MAX_APPLICATIONS_PER_DAY="20",
    )
    database = Database(app_settings.database_path)
    database.init()
    pending(database, confidence=0.91)
    page = FakeApplicationPage()
    client = HHClient(
        FakeApplicationContext(page),
        app_settings,
        database,
        ApprovalGuard(app_settings, database, now_factory=lambda: NOW + timedelta(minutes=2)),
        sleep=no_sleep,
        now_factory=lambda: NOW + timedelta(minutes=3),
    )
    service = ApprovalService(
        app_settings, database, client, now_factory=lambda: NOW + timedelta(minutes=1)
    )

    result = asyncio.run(service.auto_apply("job-1"))

    assert result.ok
    assert page.clicks == ["open_response", "final_submit"]
    assert database.get("job-1").status is VacancyStatus.APPLIED


def test_auto_apply_keeps_low_confidence_vacancy_pending(tmp_path: Path) -> None:
    app_settings = settings(
        tmp_path,
        APP_MODE="approval",
        ENABLE_REAL_APPLY="true",
        TG_USER_ID="42",
        AUTO_APPLY_ENABLED="true",
        MAX_APPLICATIONS_PER_DAY="20",
    )
    database = Database(app_settings.database_path)
    database.init()
    pending(database, confidence=0.84)
    service = ApprovalService(
        app_settings,
        database,
        HHClient(
            NeverPageContext(), app_settings, database, ApprovalGuard(app_settings, database)
        ),
    )

    result = asyncio.run(service.auto_apply("job-1"))

    assert not result.ok
    assert database.get("job-1").status is VacancyStatus.PENDING_APPROVAL


def test_low_level_send_also_rejects_missing_portfolio(tmp_path: Path) -> None:
    app_settings = require_portfolio(settings(tmp_path, APP_MODE="approval", ENABLE_REAL_APPLY="true", TG_USER_ID="42"))
    database = Database(app_settings.database_path)
    database.init()
    pending(database, letter="Letter without portfolio")
    token = database.approve("job-1", 42, 42, NOW + timedelta(minutes=1))
    client = HHClient(NeverPageContext(), app_settings, database,
                      ApprovalGuard(app_settings, database, now_factory=lambda: NOW + timedelta(minutes=2)),
                      now_factory=lambda: NOW + timedelta(minutes=3))

    assert not asyncio.run(client.submit_application(ApplicationPermission("job-1", token, 42)))
    assert database.get("job-1").error_text == "required_portfolio_missing"


@pytest.mark.parametrize(
    ("letter", "expected_error"),
    [
        (
            f"Complete sentence.\n\n.\n\nПортфолио: {PORTFOLIO}",
            "cover_letter_layout_artifact",
        ),
        (
            f"Incomplete sentence\n\nПортфолио: {PORTFOLIO}",
            "cover_letter_incomplete",
        ),
    ],
)
def test_low_level_send_rejects_malformed_cover_letter_before_browser(
    tmp_path: Path, letter: str, expected_error: str
) -> None:
    app_settings = require_portfolio(
        settings(
            tmp_path,
            APP_MODE="approval",
            ENABLE_REAL_APPLY="true",
            TG_USER_ID="42",
        )
    )
    database = Database(app_settings.database_path)
    database.init()
    pending(database, letter=letter)
    token = database.approve("job-1", 42, 42, NOW + timedelta(minutes=1))
    client = HHClient(
        NeverPageContext(),
        app_settings,
        database,
        ApprovalGuard(
            app_settings,
            database,
            now_factory=lambda: NOW + timedelta(minutes=2),
        ),
        now_factory=lambda: NOW + timedelta(minutes=3),
    )

    assert not asyncio.run(
        client.submit_application(ApplicationPermission("job-1", token, 42))
    )
    assert database.get("job-1").error_text == expected_error


@pytest.mark.parametrize("field_state", ["missing", "fill_failed", "discarded", "correct"])
def test_required_portfolio_must_reach_hh_textarea(tmp_path: Path, field_state: str) -> None:
    app_settings = require_portfolio(settings(tmp_path, APP_MODE="approval", ENABLE_REAL_APPLY="true", TG_USER_ID="42"))
    database = Database(app_settings.database_path)
    database.init()
    letter = f"Привет команде Example!\n\nФинтех и AI.\n\nПортфолио: {PORTFOLIO}"
    pending(database, letter=letter)
    page = FakeApplicationPage()
    if field_state == "missing":
        page.textarea.visible = False
    elif field_state == "fill_failed":
        async def fail_fill(value):
            raise RuntimeError("fill failed")
        page.textarea.fill = fail_fill
    elif field_state == "discarded":
        async def discard_fill(value):
            page.textarea.filled = "Text cleared by the page"
        page.textarea.fill = discard_fill
    client = HHClient(FakeApplicationContext(page), app_settings, database,
                      ApprovalGuard(app_settings, database, now_factory=lambda: NOW + timedelta(minutes=2)),
                      sleep=no_sleep, now_factory=lambda: NOW + timedelta(minutes=3))
    service = ApprovalService(app_settings, database, client, now_factory=lambda: NOW + timedelta(minutes=1))

    result = asyncio.run(service.approve_and_apply("job-1", 42))

    assert result.ok is (field_state == "correct")
    assert ("final_submit" in page.clicks) is (field_state == "correct")
    if field_state == "correct":
        assert page.textarea.filled == letter
    else:
        assert database.get("job-1").status is VacancyStatus.APPLY_FAILED
