from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from aiogram import Bot

from config import Settings, load_settings
from cover_letter import cover_letter_validation_error


@dataclass(frozen=True)
class ServiceState:
    loaded: bool
    running: bool
    pid: int | None = None


def service_state(label: str) -> ServiceState:
    result = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return ServiceState(False, False)
    state = ""
    pid = None
    for line in result.stdout.splitlines():
        key, separator, value = line.strip().partition(" = ")
        if not separator:
            continue
        if key == "state" and not state:
            state = value
        elif key == "pid" and pid is None and value.isdigit():
            pid = int(value)
    return ServiceState(True, state == "running" and pid is not None, pid)


def repair_service(label: str) -> bool:
    result = subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{label}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _day_bounds(day: date, timezone: ZoneInfo) -> tuple[str, str]:
    start = datetime.combine(day, time.min, tzinfo=timezone).astimezone(UTC)
    end = datetime.combine(
        day + timedelta(days=1), time.min, tzinfo=timezone
    ).astimezone(UTC)
    return start.isoformat(), end.isoformat()


def _audit_day(now: datetime, settings: Settings) -> date:
    local = now.astimezone(ZoneInfo(settings.auto_apply.timezone))
    if local.hour < settings.auto_apply.end_hour:
        return local.date() - timedelta(days=1)
    return local.date()


def build_report(
    settings: Settings,
    *,
    now: datetime,
    state: ServiceState | None = None,
    restarted: bool = False,
) -> tuple[str, bool, date]:
    timezone = ZoneInfo(settings.auto_apply.timezone)
    day = _audit_day(now, settings)
    start, end = _day_bounds(day, timezone)
    connection = sqlite3.connect(f"file:{settings.database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        applied_rows = connection.execute(
            """
            SELECT id, cover_letter FROM vacancies
            WHERE status = 'applied' AND applied_at >= ? AND applied_at < ?
            """,
            (start, end),
        ).fetchall()
        runs = connection.execute(
            """
            SELECT * FROM search_runs
            WHERE started_at >= ? AND started_at < ? ORDER BY id
            """,
            (start, end),
        ).fetchall()
        llm = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failed
            FROM llm_requests WHERE started_at >= ? AND started_at < ?
            """,
            (start, end),
        ).fetchone()
        uncertain = connection.execute(
            """
            SELECT COUNT(*) FROM vacancies
            WHERE status = 'apply_failed'
              AND COALESCE(submit_attempted_at, applying_at, discovered_at) >= ?
              AND COALESCE(submit_attempted_at, applying_at, discovered_at) < ?
              AND error_text NOT LIKE 'questionnaire_required%'
            """,
            (start, end),
        ).fetchone()[0]
        questionnaires = connection.execute(
            """
            SELECT COUNT(*) FROM vacancies
            WHERE status = 'apply_failed'
              AND COALESCE(submit_attempted_at, applying_at, discovered_at) >= ?
              AND COALESCE(submit_attempted_at, applying_at, discovered_at) < ?
              AND error_text LIKE 'questionnaire_required%'
            """,
            (start, end),
        ).fetchone()[0]
    finally:
        connection.close()

    cover = settings.profile.cover_letter
    invalid_letters = sum(
        bool(
            cover_letter_validation_error(
                row["cover_letter"],
                cover.required_portfolio_url,
                cover.closing,
                cover.max_length,
            )
        )
        for row in applied_rows
    )
    paused_runs = sum(row["state"] == "paused_by_circuit_breaker" for row in runs)
    failed_runs = sum(row["state"] == "failed" for row in runs)
    run_errors = sum(row["error_count"] for row in runs)
    found = sum(row["found_results"] for row in runs)
    last = runs[-1] if runs else None

    problems: list[str] = []
    if state is not None and not state.running:
        problems.append("основной сервис не работает")
    if not runs:
        problems.append("за день не было поисковых прогонов")
    if failed_runs:
        problems.append(f"аварийных прогонов: {failed_runs}")
    if uncertain:
        problems.append(f"неопределённых отправок: {uncertain}")
    if invalid_letters:
        problems.append(f"писем с ошибкой качества в базе: {invalid_letters}")

    status = "OK" if not problems else "НУЖНО ВНИМАНИЕ"
    lines = [
        f"Ежедневный аудит HH · {day.strftime('%d.%m.%Y')}",
        f"Статус: {status}",
        f"Отклики: {len(applied_rows)}/{settings.max_applications_per_day}",
        f"Поиск: прогонов {len(runs)}, просмотрено {found}, ошибок {run_errors}",
        f"LLM: запросов {int(llm['total'] or 0)}, ошибок {int(llm['failed'] or 0)}",
        f"Сопроводительные: корректных {len(applied_rows) - invalid_letters}/{len(applied_rows)}",
    ]
    if state is not None:
        service_text = "работает"
        if restarted and state.running:
            service_text = "перезапущен и работает"
        elif not state.running:
            service_text = "остановлен"
        lines.insert(
            2,
            f"Сервис: {service_text}"
            + (f" (pid {state.pid})" if state.pid else ""),
        )
    if questionnaires:
        lines.append(f"Анкеты работодателя: {questionnaires} — оставлены без автоответа")
    if paused_runs:
        lines.append(f"Circuit breaker срабатывал: {paused_runs}")
    if last is not None:
        last_time = datetime.fromisoformat(last["finished_at"]).astimezone(timezone)
        lines.append(
            f"Последний прогон: {last_time.strftime('%H:%M')}, {last['state']}"
        )
    if len(applied_rows) < settings.max_applications_per_day:
        lines.append(
            f"До дневного лимита не дошло: {settings.max_applications_per_day - len(applied_rows)}"
        )
    if problems:
        lines.append("Проблемы: " + "; ".join(problems))
    return "\n".join(lines), not problems, day


async def send_once(settings: Settings, text: str, day: date, *, force: bool) -> bool:
    reports_dir = settings.database_path.parent / ".local" / "audit-reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    marker = reports_dir / f"{day.isoformat()}.json"
    digest = hashlib.sha256(text.encode()).hexdigest()
    if marker.exists() and not force:
        return False
    bot = Bot(token=settings.tg_bot_token)
    try:
        await bot.send_message(chat_id=settings.tg_user_id, text=text)
    finally:
        await bot.session.close()
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "sent_at": datetime.now(UTC).isoformat(),
                "report_sha256": digest,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(marker)
    return True


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily HH agent audit")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repair-service", action="store_true")
    parser.add_argument(
        "--launchd-label",
        default="",
        help="Optional macOS launchd service label to check and repair",
    )
    args = parser.parse_args(argv)
    if args.repair_service and not args.launchd_label:
        parser.error("--repair-service requires --launchd-label")
    if args.launchd_label and sys.platform != "darwin":
        parser.error("--launchd-label is supported only on macOS")
    settings = load_settings(args.env_file)
    state = service_state(args.launchd_label) if args.launchd_label else None
    restarted = False
    if args.repair_service and state is not None and not state.running and state.loaded:
        restarted = repair_service(args.launchd_label)
        if restarted:
            __import__("time").sleep(2)
            state = service_state(args.launchd_label)
    text, healthy, day = build_report(
        settings, now=datetime.now(UTC), state=state, restarted=restarted
    )
    print(text)
    if args.send:
        sent = asyncio.run(send_once(settings, text, day, force=args.force))
        print("Telegram report: sent" if sent else "Telegram report: already sent")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(cli())
