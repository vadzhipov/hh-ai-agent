from pathlib import Path

import pytest
import yaml

from config import ConfigError, load_settings


VALID_ENV = {
    "TG_BOT_TOKEN": "test-token",
    "TG_USER_ID": "123456",
    "OLLAMA_URL": "http://localhost:11434/api/generate",
    "OLLAMA_MODEL": "llama3",
    "ENABLE_REAL_APPLY": "false",
    "BROWSER_HEADLESS": "false",
    "CHECK_INTERVAL_MINUTES": "30",
    "MAX_APPLICATIONS_PER_DAY": "5",
    "MAX_VACANCIES_PER_QUERY": "20",
    "MAX_PAGES_PER_QUERY": "2",
    "MIN_SECONDS_BETWEEN_ACTIONS": "5",
}

VALID_PROFILE = {
    "candidate": {
        "name": "Test Candidate",
        "location": "Moscow",
        "desired_positions": ["Python developer"],
        "experience_summary": "Built internal Python services.",
        "education": "",
        "technologies": ["Python", "SQLite"],
        "projects": [],
        "github_url": "",
        "salary_expectation": "",
        "work_format": ["remote"],
        "excluded_positions": ["sales"],
        "excluded_companies": [],
        "excluded_keywords": [],
        "additional_information": "",
    },
    "hh": {
        "resume_name": "Python developer",
        "search_queries": ["Python developer"],
        "areas": ["1"],
        "experience_filters": ["between1And3"],
        "remote_only": False,
    },
    "cover_letter": {
        "language": "ru",
        "max_length": 1800,
        "style": "professional",
    },
}


def write_profile(tmp_path: Path, data: dict | None = None) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(
        yaml.safe_dump(data or VALID_PROFILE, allow_unicode=True), encoding="utf-8"
    )
    return path


def test_load_settings_uses_safe_defaults(tmp_path: Path) -> None:
    settings = load_settings(
        profile_path=write_profile(tmp_path), environ=VALID_ENV
    )

    assert settings.app_mode == "dry_run"
    assert settings.enable_real_apply is False
    assert settings.browser_backend == "cloakbrowser"
    assert settings.browser_headless is False
    assert settings.browser_profile_dir.name == ".browser-profile"
    assert settings.tg_user_id == 123456
    assert settings.profile.hh.remote_only is False
    assert settings.profile.candidate.name == "Test Candidate"
    assert settings.profile.hh.search_queries == ("Python developer",)
    assert settings.circuit_breaker_min_sample == 5
    assert settings.circuit_breaker_unknown_ratio == 0.8
    assert settings.circuit_breaker_page_errors == 3
    assert settings.auto_apply.enabled is False
    assert settings.auto_apply.min_confidence == 0.85
    assert settings.auto_apply.timezone == "UTC"


def test_auto_apply_settings_require_live_approval_and_valid_window(
    tmp_path: Path,
) -> None:
    enabled = {
        **VALID_ENV,
        "APP_MODE": "approval",
        "ENABLE_REAL_APPLY": "true",
        "MAX_APPLICATIONS_PER_DAY": "20",
        "AUTO_APPLY_ENABLED": "true",
        "AUTO_APPLY_MIN_CONFIDENCE": "0.85",
        "AUTO_APPLY_MIN_BATCH_SIZE": "5",
        "AUTO_APPLY_MAX_BATCH_SIZE": "6",
        "AUTO_APPLY_MIN_INTERVAL_HOURS": "3",
        "AUTO_APPLY_MAX_INTERVAL_HOURS": "5",
        "AUTO_APPLY_START_HOUR": "10",
        "AUTO_APPLY_END_HOUR": "22",
        "AUTO_APPLY_TIMEZONE": "Europe/Berlin",
    }

    settings = load_settings(profile_path=write_profile(tmp_path), environ=enabled)

    assert settings.auto_apply.enabled is True
    assert settings.auto_apply.min_batch_size == 5
    assert settings.auto_apply.max_batch_size == 6
    assert settings.auto_apply.min_interval_hours == 3
    assert settings.auto_apply.max_interval_hours == 5


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("AUTO_APPLY_ENABLED", "true", "AUTO_APPLY_ENABLED requires APP_MODE=approval"),
        ("AUTO_APPLY_START_HOUR", "22", "AUTO_APPLY_START_HOUR must be before AUTO_APPLY_END_HOUR"),
        ("AUTO_APPLY_TIMEZONE", "", "AUTO_APPLY_TIMEZONE must be an IANA timezone"),
        ("AUTO_APPLY_TIMEZONE", "Not/AZone", "AUTO_APPLY_TIMEZONE must be an IANA timezone"),
    ],
)
def test_auto_apply_settings_reject_unsafe_configuration(
    tmp_path: Path, key: str, value: str, message: str
) -> None:
    environ = {**VALID_ENV, key: value}
    if key == "AUTO_APPLY_START_HOUR":
        environ.update(APP_MODE="approval", ENABLE_REAL_APPLY="true", AUTO_APPLY_ENABLED="true")

    with pytest.raises(ConfigError, match=message):
        load_settings(profile_path=write_profile(tmp_path), environ=environ)


def test_profile_remote_only_must_be_boolean(tmp_path: Path) -> None:
    profile = {**VALID_PROFILE, "hh": {**VALID_PROFILE["hh"], "remote_only": "yes"}}

    with pytest.raises(ConfigError, match="hh.remote_only must be true or false"):
        load_settings(profile_path=write_profile(tmp_path, profile), environ=VALID_ENV)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("APP_MODE", "automatic", "APP_MODE must be dry_run or approval"),
        (
            "BROWSER_BACKEND",
            "unknown",
            "BROWSER_BACKEND must be cloakbrowser or playwright",
        ),
        (
            "ENABLE_REAL_APPLY",
            "yes",
            "ENABLE_REAL_APPLY must be true or false",
        ),
        ("TG_USER_ID", "abc", "TG_USER_ID must be an integer"),
        (
            "MAX_APPLICATIONS_PER_DAY",
            "0",
            "MAX_APPLICATIONS_PER_DAY must be a positive integer",
        ),
    ],
)
def test_load_settings_rejects_invalid_environment(
    tmp_path: Path, key: str, value: str, message: str
) -> None:
    environ = {**VALID_ENV, key: value}

    with pytest.raises(ConfigError, match=message):
        load_settings(profile_path=write_profile(tmp_path), environ=environ)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "CIRCUIT_BREAKER_MIN_SAMPLE",
            "0",
            "CIRCUIT_BREAKER_MIN_SAMPLE must be a positive integer",
        ),
        (
            "CIRCUIT_BREAKER_UNKNOWN_RATIO",
            "0",
            "CIRCUIT_BREAKER_UNKNOWN_RATIO must be greater than 0 and at most 1",
        ),
        (
            "CIRCUIT_BREAKER_UNKNOWN_RATIO",
            "1.1",
            "CIRCUIT_BREAKER_UNKNOWN_RATIO must be greater than 0 and at most 1",
        ),
        (
            "CIRCUIT_BREAKER_PAGE_ERRORS",
            "no",
            "CIRCUIT_BREAKER_PAGE_ERRORS must be an integer",
        ),
    ],
)
def test_load_settings_rejects_invalid_circuit_breaker_values(
    tmp_path: Path, key: str, value: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_settings(
            profile_path=write_profile(tmp_path),
            environ={**VALID_ENV, key: value},
        )


def test_load_settings_reports_missing_secret_without_traceback_text(
    tmp_path: Path,
) -> None:
    environ = {**VALID_ENV, "TG_BOT_TOKEN": ""}

    with pytest.raises(ConfigError) as raised:
        load_settings(profile_path=write_profile(tmp_path), environ=environ)

    assert str(raised.value).startswith("Configuration error:")
    assert "TG_BOT_TOKEN is required" in str(raised.value)
    assert "Traceback" not in str(raised.value)


def test_load_settings_aggregates_required_profile_errors(tmp_path: Path) -> None:
    profile = {
        **VALID_PROFILE,
        "candidate": {
            **VALID_PROFILE["candidate"],
            "name": "",
            "desired_positions": [],
            "experience_summary": "",
        },
        "hh": {**VALID_PROFILE["hh"], "resume_name": "", "search_queries": []},
    }

    with pytest.raises(ConfigError) as raised:
        load_settings(profile_path=write_profile(tmp_path, profile), environ=VALID_ENV)

    message = str(raised.value)
    assert "candidate.name is required" in message
    assert "candidate.desired_positions must not be empty" in message
    assert "candidate.experience_summary is required" in message
    assert "hh.resume_name is required" in message
    assert "hh.search_queries must not be empty" in message


def test_load_settings_rejects_missing_profile_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="profile file not found"):
        load_settings(profile_path=tmp_path / "missing.yaml", environ=VALID_ENV)


@pytest.mark.parametrize("url,max_length,expected", [
    ("http://portfolio.example/profile", 1200, "must use HTTPS"),
    ("not-a-url", 1200, "valid HTTP"),
    ("https://portfolio.example/profile", 20, "leave room"),
])
def test_required_portfolio_configuration_is_validated(tmp_path, url, max_length, expected):
    profile = {**VALID_PROFILE, "cover_letter": {
        **VALID_PROFILE["cover_letter"], "required_portfolio_url": url, "max_length": max_length,
    }}
    with pytest.raises(ConfigError, match=expected):
        load_settings(profile_path=write_profile(tmp_path, profile), environ=VALID_ENV)


def test_closing_configuration_must_leave_room_for_letter(tmp_path: Path) -> None:
    profile = {
        **VALID_PROFILE,
        "cover_letter": {
            **VALID_PROFILE["cover_letter"],
            "closing": "A closing longer than the configured limit",
            "max_length": 20,
        },
    }

    with pytest.raises(ConfigError, match="leave room"):
        load_settings(profile_path=write_profile(tmp_path, profile), environ=VALID_ENV)
