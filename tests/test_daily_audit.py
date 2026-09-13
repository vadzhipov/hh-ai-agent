import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import daily_audit
from config import load_settings
from daily_audit import (
    ServiceState,
    _day_bounds,
    build_report,
    recoverable_runtime_stall,
)
from database import Database
from tests.test_config import VALID_ENV, VALID_PROFILE, write_profile


def test_report_marks_missing_service_and_search(tmp_path: Path) -> None:
    profile = write_profile(tmp_path)
    env = {
        **VALID_ENV,
        "DATABASE_PATH": str(tmp_path / "agent.db"),
        "AUTO_APPLY_TIMEZONE": "Europe/Berlin",
    }
    settings = load_settings(environ=env, profile_path=profile)
    Database(settings.database_path).init()

    text, healthy, day = build_report(
        settings,
        now=datetime(2026, 7, 27, 21, 0, tzinfo=UTC),
        state=ServiceState(False, False),
    )

    assert day.isoformat() == "2026-07-27"
    assert healthy is False
    assert "основной сервис не работает" in text
    assert "за день не было поисковых прогонов" in text


def test_report_accepts_valid_applied_letter(tmp_path: Path) -> None:
    profile = write_profile(
        tmp_path,
        {
            **VALID_PROFILE,
            "cover_letter": {
                **VALID_PROFILE["cover_letter"],
                "required_portfolio_url": "https://portfolio.example/profile",
                "closing": "Буду рад пообщаться.",
            },
        },
    )
    env = {
        **VALID_ENV,
        "DATABASE_PATH": str(tmp_path / "agent.db"),
        "MAX_APPLICATIONS_PER_DAY": "20",
        "AUTO_APPLY_TIMEZONE": "Europe/Berlin",
    }
    settings = load_settings(environ=env, profile_path=profile)
    database = Database(settings.database_path)
    database.init()
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            """
            INSERT INTO vacancies (
                id, title, company, url, status, discovered_at, applied_at,
                cover_letter
            ) VALUES (?, ?, ?, ?, 'applied', ?, ?, ?)
            """,
            (
                "job-1",
                "Product Designer",
                "Example",
                "https://example.com/job-1",
                "2026-07-27T10:00:00+00:00",
                "2026-07-27T10:30:00+00:00",
                "Здравствуйте! Готов обсудить задачи.\n\n"
                f"Портфолио: {settings.profile.cover_letter.required_portfolio_url}\n\n"
                f"{settings.profile.cover_letter.closing}",
            ),
        )
        connection.execute(
            """
            INSERT INTO search_runs (
                started_at, finished_at, state, query_count, found_results,
                new_vacancies, duplicates, rejected_by_filter, rejected_by_llm,
                telegram_cards, error_count, rejection_reasons_json,
                error_reasons_json, last_safe_error, circuit_reason
            ) VALUES (?, ?, 'completed', 1, 20, 1, 0, 0, 0, 0, 0, '{}', '{}', '', '')
            """,
            ("2026-07-27T10:00:00+00:00", "2026-07-27T11:00:00+00:00"),
        )

    text, healthy, _ = build_report(
        settings,
        now=datetime(2026, 7, 27, 21, 0, tzinfo=UTC),
        state=ServiceState(True, True, 123),
    )

    assert healthy is True
    assert "Сопроводительные: корректных 1/1" in text
    assert "Отклики: 1/20" in text


def test_report_can_skip_platform_service_check(tmp_path: Path) -> None:
    profile = write_profile(tmp_path)
    settings = load_settings(
        environ={**VALID_ENV, "DATABASE_PATH": str(tmp_path / "agent.db")},
        profile_path=profile,
    )
    Database(settings.database_path).init()

    text, healthy, _ = build_report(
        settings,
        now=datetime(2026, 7, 27, 23, 0, tzinfo=UTC),
    )

    assert healthy is False
    assert "Сервис:" not in text
    assert "основной сервис не работает" not in text


def test_pre_submit_page_failure_is_not_reported_as_uncertain_delivery(
    tmp_path: Path,
) -> None:
    profile = write_profile(tmp_path)
    settings = load_settings(
        environ={
            **VALID_ENV,
            "DATABASE_PATH": str(tmp_path / "agent.db"),
            "AUTO_APPLY_TIMEZONE": "Europe/Berlin",
        },
        profile_path=profile,
    )
    Database(settings.database_path).init()
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            """
            INSERT INTO vacancies (
                id, title, company, url, status, discovered_at, error_text
            ) VALUES (?, ?, ?, ?, 'apply_failed', ?, 'page_structure_changed')
            """,
            (
                "job-pre-submit",
                "Product Designer",
                "Example",
                "https://example.com/job-pre-submit",
                "2026-07-27T10:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO search_runs (
                started_at, finished_at, state, query_count, found_results,
                new_vacancies, duplicates, rejected_by_filter, rejected_by_llm,
                telegram_cards, error_count, rejection_reasons_json,
                error_reasons_json, last_safe_error, circuit_reason
            ) VALUES (?, ?, 'completed', 1, 1, 1, 0, 0, 0, 0, 1, '{}', '{}', '', '')
            """,
            ("2026-07-27T10:00:00+00:00", "2026-07-27T11:00:00+00:00"),
        )

    text, _, _ = build_report(
        settings,
        now=datetime(2026, 7, 27, 21, 0, tzinfo=UTC),
        state=ServiceState(True, True, 123),
    )

    assert "неопределённых отправок" not in text


def test_day_bounds_respect_daylight_saving_transition() -> None:
    start, end = _day_bounds(date(2026, 3, 29), ZoneInfo("Europe/Berlin"))

    duration = datetime.fromisoformat(end) - datetime.fromisoformat(start)

    assert duration.total_seconds() == 23 * 60 * 60


def test_runtime_stall_detects_network_pause_but_not_delivery_uncertainty(
    tmp_path: Path,
) -> None:
    profile = write_profile(tmp_path)
    settings = load_settings(
        environ={
            **VALID_ENV,
            "DATABASE_PATH": str(tmp_path / "agent.db"),
            "AUTO_APPLY_TIMEZONE": "Europe/Berlin",
        },
        profile_path=profile,
    )
    Database(settings.database_path).init()
    with sqlite3.connect(settings.database_path) as connection:
        values = (
            "2026-07-27T10:00:00+00:00",
            "2026-07-27T11:00:00+00:00",
        )
        connection.execute(
            """
            INSERT INTO search_runs (
                started_at, finished_at, state, query_count, found_results,
                new_vacancies, duplicates, rejected_by_filter, rejected_by_llm,
                telegram_cards, error_count, rejection_reasons_json,
                error_reasons_json, last_safe_error, circuit_reason
            ) VALUES (?, ?, 'paused_by_circuit_breaker', 1, 0, 0, 0, 0, 0,
                      0, 1, '{}', '{}', '', 'search_errors')
            """,
            values,
        )

    now = datetime(2026, 7, 27, 21, 0, tzinfo=UTC)
    assert recoverable_runtime_stall(settings, now=now) is True

    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            "UPDATE search_runs SET circuit_reason = 'application_delivery_uncertain'"
        )

    assert recoverable_runtime_stall(settings, now=now) is False


def test_cli_rechecks_same_launchd_service_after_repair(
    tmp_path: Path, monkeypatch
) -> None:
    settings = load_settings(
        environ={**VALID_ENV, "DATABASE_PATH": str(tmp_path / "agent.db")},
        profile_path=write_profile(tmp_path),
    )
    calls: list[str] = []
    states = iter([ServiceState(True, False), ServiceState(True, True, 42)])

    def fake_service_state(label: str) -> ServiceState:
        calls.append(f"check:{label}")
        return next(states)

    def fake_repair_service(label: str) -> bool:
        calls.append(f"repair:{label}")
        return True

    monkeypatch.setattr(daily_audit, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(daily_audit, "load_settings", lambda _path: settings)
    monkeypatch.setattr(daily_audit, "service_state", fake_service_state)
    monkeypatch.setattr(daily_audit, "repair_service", fake_repair_service)
    monkeypatch.setattr(daily_audit, "build_report", lambda *args, **kwargs: (
        "OK", True, date(2026, 7, 27)
    ))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = daily_audit.cli(
        [
            "--env-file",
            str(tmp_path / ".env"),
            "--launchd-label",
            "com.example.hh-agent",
            "--repair-service",
        ]
    )

    assert result == 0
    assert calls == [
        "check:com.example.hh-agent",
        "repair:com.example.hh-agent",
        "check:com.example.hh-agent",
    ]
