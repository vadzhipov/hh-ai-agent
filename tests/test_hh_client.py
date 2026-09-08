import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from approval import ApprovalGuard
from config import load_settings
from database import Database
from hh_client import HHClient, PageState, VacancySummary, classify_page
from tests.test_config import VALID_ENV, write_profile
from vacancy_filter import title_rejection_reason, vacancy_rejection_reason


class FakeLocator:
    def __init__(self, visible: bool):
        self.visible = visible
        self.first = self

    async def is_visible(self) -> bool:
        return self.visible

    async def inner_text(self) -> str:
        return ""


class FakePage:
    def __init__(self, visible_selector: str = "", broken: bool = False):
        self.visible_selector = visible_selector
        self.broken = broken

    def locator(self, selector: str) -> FakeLocator:
        if self.broken:
            raise RuntimeError("page closed")
        return FakeLocator(selector == self.visible_selector)

    async def goto(self, url: str, **kwargs) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeContext:
    def __init__(self, page: FakePage):
        self.page = page

    async def new_page(self) -> FakePage:
        return self.page


class SearchCard:
    def __init__(self, href: str, title: str):
        self.href = href
        self.title = title

    async def get_attribute(self, name: str) -> str | None:
        return self.href if name == "href" else None

    async def inner_text(self) -> str:
        return self.title


class TextLocator(FakeLocator):
    def __init__(self, text: str = "", href: str | None = None):
        super().__init__(bool(text))
        self.text = text
        self.href = href

    async def inner_text(self) -> str:
        return self.text

    async def get_attribute(self, name: str) -> str | None:
        return self.href if name == "href" else None


class SearchLocator:
    def __init__(self, cards: list[SearchCard]):
        self.cards = cards

    async def all(self) -> list[SearchCard]:
        return self.cards


class SearchPage(FakePage):
    def __init__(self, cards: list[SearchCard] | None = None):
        super().__init__()
        self.cards = cards or []
        self.urls: list[str] = []

    async def goto(self, url: str, **kwargs) -> None:
        self.urls.append(url)

    def locator(self, selector: str) -> SearchLocator:
        return SearchLocator(self.cards)


class VacancyPage(FakePage):
    def locator(self, selector: str) -> TextLocator:
        values = {
            '[data-qa="vacancy-description"]': "Описание вакансии.",
            '[data-qa="vacancy-company-name"]': "Компания",
        }
        return TextLocator(
            values.get(selector, ""),
            "/employer/123?hhtmFrom=vacancy"
            if selector == '[data-qa="vacancy-company-name"]'
            else None,
        )


class EmployerPage(FakePage):
    def __init__(self, rating: str = "", reviews: str = ""):
        super().__init__()
        self.values = {
            '[data-qa="employer-review-small-widget-total-rating"]': rating,
            '[data-qa="employer-review-small-widget-review-count-action"]': reviews,
        }

    def locator(self, selector: str) -> TextLocator:
        return TextLocator(self.values.get(selector, ""))


class FailingContext:
    async def new_page(self):
        raise RuntimeError("page creation failed")


class RelocationWarningLocator:
    def __init__(self, visible: bool):
        self.visible = visible
        self.clicked = False
        self.first = self

    async def is_visible(self) -> bool:
        return self.visible

    async def click(self) -> None:
        self.clicked = True


class RelocationWarningPage:
    def __init__(self, visible: bool):
        self.confirmation = RelocationWarningLocator(visible)

    def locator(self, selector: str) -> RelocationWarningLocator:
        assert selector == 'button[data-qa="relocation-warning-confirm"]'
        return self.confirmation


class PostSubmitCoverLetterLocator:
    def __init__(self, page: "PostSubmitCoverLetterPage", kind: str):
        self.page = page
        self.kind = kind
        self.first = self

    async def is_visible(self) -> bool:
        if self.kind == "attach":
            return not self.page.form_open and not self.page.submitted
        return self.page.form_open

    async def click(self) -> None:
        if self.kind == "attach":
            self.page.form_open = True
        elif self.kind == "submit":
            self.page.form_open = False
            self.page.submitted = True

    async def wait_for(self, *, state: str, timeout: int) -> None:
        visible = await self.is_visible()
        if (state == "visible" and not visible) or (state == "hidden" and visible):
            raise RuntimeError("unexpected locator state")

    async def fill(self, value: str) -> None:
        self.page.value = value

    async def input_value(self) -> str:
        return self.page.value


class PostSubmitCoverLetterPage:
    def __init__(self):
        self.form_open = False
        self.submitted = False
        self.value = ""

    def locator(self, selector: str) -> PostSubmitCoverLetterLocator:
        if selector == 'button[data-qa="responded-success-attach-cover-letter"]':
            return PostSubmitCoverLetterLocator(self, "attach")
        if selector == 'textarea[data-qa="vacancy-response-popup-form-letter-input"]':
            return PostSubmitCoverLetterLocator(self, "textarea")
        if selector == 'button[data-qa="vacancy-response-letter-submit"]:visible':
            return PostSubmitCoverLetterLocator(self, "submit")
        raise AssertionError(selector)


class ChatCoverLetterLocator:
    def __init__(self, frame: "ChatCoverLetterFrame", kind: str):
        self.frame = frame
        self.kind = kind
        self.first = self

    async def wait_for(self, **kwargs) -> None:
        return None

    async def click(self) -> None:
        if self.kind == "attach":
            self.frame.form_open = True
        elif self.kind == "send":
            self.frame.sent = True

    async def fill(self, value: str) -> None:
        self.frame.value = value

    async def input_value(self) -> str:
        return self.frame.value

    async def inner_text(self) -> str:
        if self.kind == "body":
            return self.frame.value if self.frame.sent else "Без сопроводительного письма"
        return ""


class ChatCoverLetterFrame:
    url = "https://chatik.hh.ru/chat/123"

    def __init__(self):
        self.form_open = False
        self.sent = False
        self.value = ""

    def locator(self, selector: str) -> ChatCoverLetterLocator:
        if selector == "body":
            return ChatCoverLetterLocator(self, "body")
        if selector == 'textarea[data-qa="text-input"]':
            return ChatCoverLetterLocator(self, "textarea")
        if selector == 'button[data-qa="chatik-do-send-message"]':
            return ChatCoverLetterLocator(self, "send")
        raise AssertionError(selector)

    def get_by_text(self, text: str, *, exact: bool) -> ChatCoverLetterLocator:
        assert text == "Добавить сопроводительное"
        assert not exact
        return ChatCoverLetterLocator(self, "attach")

    async def wait_for_function(self, expression: str, arg: str, **kwargs) -> None:
        assert "innerText.includes" in expression
        if not self.sent or arg not in self.value:
            raise RuntimeError("cover letter not visible")


class ChatCoverLetterPage:
    def __init__(self):
        self.frame = ChatCoverLetterFrame()
        self.frames = [self.frame]

    def locator(self, selector: str) -> ChatCoverLetterLocator:
        assert selector == 'button[data-qa="vacancy-response-link-view-topic"]'
        return ChatCoverLetterLocator(self.frame, "chat")


class RetryLoginPage(FakePage):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    async def goto(self, url: str, **kwargs) -> None:
        self.attempts += 1
        if self.attempts < 3:
            raise RuntimeError("temporary navigation failure")


class CaptchaLocator:
    def __init__(self, page: "CaptchaPage"):
        self.page = page
        self.first = self

    async def is_visible(self) -> bool:
        return True

    async def fill(self, value: str) -> None:
        self.page.solution = value

    async def press(self, key: str) -> None:
        self.page.actions.append(f"press:{key}")

    async def click(self) -> None:
        self.page.actions.append("click:submit")
        self.page.solved = True


class CaptchaPage:
    def __init__(self):
        self.actions: list[str] = []
        self.solution = ""
        self.solved = False

    def locator(self, selector: str):
        if selector == 'input[type="text"]' or "button" in selector:
            return CaptchaLocator(self)
        return FakeLocator(
            (self.solved and selector == '[data-qa="vacancy-description"]')
            or (not self.solved and selector == 'form[action*="captcha"]')
        )

    async def screenshot(self, *, path: Path) -> None:
        path.touch()


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ('[data-qa="vacancy-description"]', PageState.VACANCY_LOADED),
        ('form[action*="captcha"]', PageState.CAPTCHA_DETECTED),
        ('[data-qa="access-denied"]', PageState.ACCESS_DENIED),
        ('[data-qa="vacancy-removed"]', PageState.VACANCY_REMOVED),
        ("", PageState.PAGE_STRUCTURE_CHANGED),
    ],
)
def test_page_state_uses_explicit_signals(selector: str, expected: PageState) -> None:
    assert asyncio.run(classify_page(FakePage(selector))) is expected


def test_page_state_reports_network_error() -> None:
    assert asyncio.run(classify_page(FakePage(broken=True))) is PageState.NETWORK_ERROR


def test_visible_but_empty_description_reports_structure_change(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FakeContext(FakePage('[data-qa="vacancy-description"]')),
        settings,
        database,
        ApprovalGuard(settings, database),
    )
    summary = VacancySummary(
        "job-1", "Developer", "https://example.com/vacancy/job-1", "Python"
    )

    result = asyncio.run(client.read_vacancy(summary))

    assert result.state is PageState.PAGE_STRUCTURE_CHANGED


def test_page_creation_failure_reports_network_error(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FailingContext(), settings, database, ApprovalGuard(settings, database)
    )
    summary = VacancySummary(
        "job-1", "Developer", "https://example.com/vacancy/job-1", "Python"
    )

    result = asyncio.run(client.read_vacancy(summary))

    assert result.state is PageState.NETWORK_ERROR
    assert "page creation failed" in result.error


def test_search_page_creation_failure_returns_safe_error(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FailingContext(), settings, database, ApprovalGuard(settings, database)
    )

    result = asyncio.run(client.search_vacancies("Python", (), ()))

    assert result.summaries == []
    assert result.error_reason == "search_error:RuntimeError"


def test_search_ignores_ad_redirects_without_numeric_vacancy_id(
    tmp_path: Path,
) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        max_pages_per_query=1,
    )
    database = Database(settings.database_path)
    database.init()
    page = SearchPage(
        [
            SearchCard(
                "https://adsrv.hh.ru/click?clickType=link_to_vacancy",
                "Ad",
            ),
            SearchCard("https://hh.ru/vacancy/135006927?from=search", "DevOps"),
        ]
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=no_sleep,
    )

    result = asyncio.run(client.search_vacancies("DevOps", (), ()))

    assert [(item.id, item.title) for item in result.summaries] == [
        ("135006927", "DevOps")
    ]


def test_search_counts_found_new_and_existing_results(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        max_pages_per_query=1,
    )
    database = Database(settings.database_path)
    database.init()
    assert database.discover(
        job_id="135006927",
        title="Existing",
        company="Example",
        url="https://hh.ru/vacancy/135006927",
        description_hash="hash",
        search_query="DevOps",
        discovered_at=datetime(2026, 7, 29, tzinfo=UTC),
    )
    page = SearchPage(
        [
            SearchCard("https://hh.ru/vacancy/135006927", "Existing"),
            SearchCard("https://hh.ru/vacancy/135006928", "New"),
            SearchCard("https://example.com/ad", "Ad"),
        ]
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=no_sleep,
    )

    result = asyncio.run(client.search_vacancies("DevOps", (), ()))

    assert result.found_results == 2
    assert result.duplicates == 1
    assert [item.id for item in result.summaries] == ["135006928"]


def test_search_adds_hh_remote_schedule_only_when_requested(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        max_pages_per_query=1,
    )
    database = Database(settings.database_path)
    database.init()
    page = SearchPage([])

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page), settings, database, ApprovalGuard(settings, database), sleep=no_sleep
    )
    asyncio.run(client.search_vacancies("DevOps", (), (), remote_only=True))

    assert parse_qs(urlparse(page.urls[0]).query)["schedule"] == ["remote"]


def test_search_returns_only_pending_existing_vacancies_as_repeats(
    tmp_path: Path,
) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
        max_pages_per_query=1,
    )
    database = Database(settings.database_path)
    database.init()
    now = datetime(2026, 7, 30, tzinfo=UTC)
    for job_id in ("135006927", "135006928"):
        assert database.discover(
            job_id=job_id,
            title="Existing",
            company="Example",
            url=f"https://hh.ru/vacancy/{job_id}",
            description_hash="hash",
            search_query="DevOps",
            discovered_at=now,
        )
        assert database.request_approval(
            job_id=job_id,
            cover_letter="Letter",
            llm_decision=True,
            llm_reason="Relevant",
            confidence=0.9,
            now=now,
        )
    assert database.skip("135006928", 42, 42)
    page = SearchPage(
        [
            SearchCard("https://hh.ru/vacancy/135006927", "Pending"),
            SearchCard("https://hh.ru/vacancy/135006928", "Skipped"),
            SearchCard("https://hh.ru/vacancy/135006929", "New"),
        ]
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=no_sleep,
    )

    result = asyncio.run(client.search_vacancies("DevOps", (), ()))

    assert [(item.id, item.previously_sent) for item in result.summaries] == [
        ("135006927", True),
        ("135006929", False),
    ]
    assert result.duplicates == 2


def test_read_vacancy_collects_safe_company_url(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FakeContext(VacancyPage()),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=lambda _: asyncio.sleep(0),
    )

    result = asyncio.run(
        client.read_vacancy(
            VacancySummary("job-1", "Developer", "https://hh.ru/vacancy/1", "Python")
        )
    )

    assert result.company_url == "https://hh.ru/employer/123"


def test_read_company_details_parses_rating_and_review_count(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FakeContext(EmployerPage("3,4", "14 отзывов")),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=lambda _: asyncio.sleep(0),
    )

    result = asyncio.run(client.read_company_details("https://hh.ru/employer/123"))

    assert result.rating == 3.4
    assert result.reviews_count == 14


def test_read_company_details_rejects_non_hh_url(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
    )
    database = Database(settings.database_path)
    database.init()
    client = HHClient(
        FakeContext(EmployerPage("5", "100 отзывов")),
        settings,
        database,
        ApprovalGuard(settings, database),
    )

    result = asyncio.run(client.read_company_details("https://example.com/employer/1"))

    assert result.rating is None
    assert result.reviews_count is None


def test_login_check_retries_temporary_navigation_errors(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    page = RetryLoginPage()

    async def no_sleep(_seconds: float) -> None:
        return None

    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=no_sleep,
    )

    assert asyncio.run(client.ensure_login())
    assert page.attempts == 3


def test_captcha_solution_uses_submit_button(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    page = CaptchaPage()
    client = HHClient(
        FakeContext(page), settings, database, ApprovalGuard(settings, database)
    )
    summary = VacancySummary(
        "job-1", "Developer", "https://example.com/vacancy/job-1", "Python"
    )

    async def solve(*_args):
        return "abcd"

    result = asyncio.run(client._solve_captcha(page, summary, solve))

    assert result is PageState.VACANCY_LOADED
    assert page.solution == "abcd"
    assert page.actions == ["click:submit"]


@pytest.mark.parametrize("visible", [True, False])
def test_relocation_warning_is_confirmed_only_when_hh_shows_it(
    tmp_path: Path, visible: bool
) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    page = RelocationWarningPage(visible)
    client = HHClient(
        FakeContext(FakePage()),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=lambda _: asyncio.sleep(0),
    )

    confirmed = asyncio.run(client._confirm_relocation_warning(page))

    assert confirmed is visible
    assert page.confirmation.clicked is visible


def test_post_submit_cover_letter_is_filled_verified_and_sent(tmp_path: Path) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    page = PostSubmitCoverLetterPage()
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=lambda _: asyncio.sleep(0),
    )
    portfolio = settings.profile.cover_letter.required_portfolio_url
    letter = f"Relevant experience.\n\n{portfolio}"

    asyncio.run(client._attach_cover_letter_after_response(page, letter, portfolio))

    assert page.value == letter
    assert page.submitted


def test_missing_cover_letter_is_attached_and_verified_inside_chat_frame(
    tmp_path: Path,
) -> None:
    settings = replace(
        load_settings(profile_path=write_profile(tmp_path), environ=VALID_ENV),
        database_path=tmp_path / "agent.db",
        min_seconds_between_actions=0,
    )
    database = Database(settings.database_path)
    database.init()
    page = ChatCoverLetterPage()
    client = HHClient(
        FakeContext(page),
        settings,
        database,
        ApprovalGuard(settings, database),
        sleep=lambda _: asyncio.sleep(0),
    )
    portfolio = "https://portfolio.example/candidate"
    letter = f"Relevant experience.\n\nПортфолио: {portfolio}"

    asyncio.run(client._ensure_cover_letter_in_chat(page, letter, portfolio))

    assert page.frame.sent
    assert page.frame.value == letter


@pytest.mark.parametrize(
    ("title", "excluded", "expected"),
    [
        ("Senior Python developer", (), None),
        ("Senior Product Designer", (), None),
        ("Продуктовый дизайнер", (), None),
        ("Lead UX/UI Designer", ("lead",), "lead"),
        ("Python sales engineer", ("sales",), "sales"),
        ("Python developer", (), None),
    ],
)
def test_title_filter_returns_the_matched_reason(
    title: str, excluded: tuple[str, ...], expected: str | None
) -> None:
    assert title_rejection_reason(title, excluded) == expected


@pytest.mark.parametrize(
    ("title", "company", "description", "expected"),
    [
        ("Team Lead Product Designer", "Example", "Product work", "position:team lead"),
        ("Product Designer", "Blocked Corp", "Product work", "company:blocked"),
        ("Product Designer", "Example", "Разработка мобильных игр", "keyword:мобильных игр"),
        ("Product Designer", "Example", "Product work", None),
    ],
)
def test_vacancy_filter_checks_position_company_and_industry(
    title: str, company: str, description: str, expected: str | None
) -> None:
    assert vacancy_rejection_reason(
        title=title,
        company=company,
        description=description,
        excluded_positions=("team lead",),
        excluded_companies=("blocked",),
        excluded_keywords=("разработка игр", "мобильных игр"),
    ) == expected
