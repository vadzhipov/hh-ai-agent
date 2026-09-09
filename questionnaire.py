from __future__ import annotations

from dataclasses import dataclass

from config import CandidateProfile


@dataclass(frozen=True)
class QuestionnaireOption:
    value: str
    label: str
    index: int


@dataclass(frozen=True)
class QuestionnaireQuestion:
    key: str
    prompt: str
    text_name: str = ""
    options: tuple[QuestionnaireOption, ...] = ()


@dataclass(frozen=True)
class QuestionnaireAnswer:
    key: str
    text_name: str = ""
    text: str = ""
    option_name: str = ""
    option_index: int | None = None


@dataclass(frozen=True)
class QuestionnairePlan:
    answers: tuple[QuestionnaireAnswer, ...] = ()
    generated_questions: tuple[QuestionnaireQuestion, ...] = ()
    manual_questions: tuple[str, ...] = ()
    rejection_reason: str = ""


_SALARY_TERMS = ("зарплат", "доход", "оплат", "компенсац")
_NO_THRESHOLD_TERMS = ("без минимального порога", "без порога")
_REMOTE_TERMS = ("удален", "удалён", "remote", "дистанцион")
_OFFICE_TERMS = ("офис", "гибрид", "очно", "on-site", "onsite")
_PORTFOLIO_TERMS = ("портфолио", "portfolio")
_YES_LABELS = ("да", "yes", "готов", "подходит")
_NO_LABELS = ("нет", "no", "не готов", "не подходит")
_PROFESSIONAL_TERMS = (
    "опыт",
    "дизайн",
    "figma",
    "интерфейс",
    "продукт",
    "проектир",
    "исследован",
    "прототип",
    "сценари",
    "пользовател",
    "разработ",
    "аналитик",
    "метрик",
    "ux",
    "ui",
    "saas",
    "crm",
    "финтех",
    "нейросет",
)
_SENSITIVE_TERMS = (
    "граждан",
    "разрешение на работу",
    "виз",
    "паспорт",
    "возраст",
    "семейн",
    "детей",
    "здоров",
    "инвалид",
    "судим",
    "военн",
    "национальн",
    "религи",
    "политичес",
    "телефон",
    "почт",
    "email",
    "e-mail",
    "адрес",
    "переезд",
    "релокац",
    "дата выхода",
    "когда готовы",
    "отработка",
)


def _match_term(text: str, terms: tuple[str, ...]) -> str | None:
    normalized = text.casefold()
    return next((term for term in terms if term.casefold() in normalized), None)


def questionnaire_rejection_reason(
    questions: tuple[QuestionnaireQuestion, ...], candidate: CandidateProfile
) -> str | None:
    text = "\n".join(question.prompt for question in questions)
    groups = (
        ("company", candidate.excluded_companies),
        ("position", candidate.excluded_positions),
        ("keyword", candidate.excluded_keywords),
    )
    for prefix, terms in groups:
        if match := _match_term(text, terms):
            return f"questionnaire_{prefix}:{match}"
    return None


def _choice(
    question: QuestionnaireQuestion, accepted_labels: tuple[str, ...]
) -> QuestionnaireAnswer | None:
    for option in question.options:
        normalized = option.label.casefold().strip()
        if any(
            normalized == label
            or normalized.startswith(f"{label},")
            or normalized.startswith(f"{label} ")
            for label in accepted_labels
        ):
            return QuestionnaireAnswer(
                key=question.key,
                option_name=question.key,
                option_index=option.index,
            )
    return None


def _salary_answer(
    question: QuestionnaireQuestion, salary_expectation: str
) -> QuestionnaireAnswer | None:
    no_threshold = bool(_match_term(salary_expectation, _NO_THRESHOLD_TERMS))
    if question.options and no_threshold:
        if answer := _choice(question, _YES_LABELS):
            return answer
    if question.text_name:
        expectation = salary_expectation.strip()
        if not expectation or no_threshold:
            expectation = "Готов обсудить уровень дохода по итогам интервью."
        custom = _choice(question, ("свой вариант", "другое", "other"))
        return QuestionnaireAnswer(
            key=question.key,
            text_name=question.text_name,
            text=expectation,
            option_name=custom.option_name if custom else "",
            option_index=custom.option_index if custom else None,
        )
    return None


def _remote_answer(
    question: QuestionnaireQuestion, remote_only: bool
) -> QuestionnaireAnswer | None:
    if not remote_only:
        return None
    prompt = question.prompt.casefold()
    if question.options:
        if any(term in prompt for term in _OFFICE_TERMS):
            return _choice(question, _NO_LABELS)
        return _choice(question, _YES_LABELS)
    if not question.text_name:
        return None
    return QuestionnaireAnswer(
        key=question.key,
        text_name=question.text_name,
        text="Рассматриваю только полностью удалённый формат работы.",
    )


def _is_professional_text_question(question: QuestionnaireQuestion) -> bool:
    normalized = question.prompt.casefold()
    return bool(
        question.text_name
        and not question.options
        and not any(term in normalized for term in _SENSITIVE_TERMS)
        and any(term in normalized for term in _PROFESSIONAL_TERMS)
    )


def plan_questionnaire(
    questions: tuple[QuestionnaireQuestion, ...],
    candidate: CandidateProfile,
    *,
    remote_only: bool,
    portfolio_url: str,
    enabled: bool,
) -> QuestionnairePlan:
    if rejection := questionnaire_rejection_reason(questions, candidate):
        return QuestionnairePlan(rejection_reason=rejection)
    if not enabled:
        return QuestionnairePlan(
            manual_questions=tuple(question.prompt for question in questions)
        )

    answers: list[QuestionnaireAnswer] = []
    generated: list[QuestionnaireQuestion] = []
    manual: list[str] = []
    for question in questions:
        normalized = question.prompt.casefold()
        answer: QuestionnaireAnswer | None = None
        if any(term in normalized for term in _SALARY_TERMS):
            answer = _salary_answer(question, candidate.salary_expectation)
        elif any(term in normalized for term in _REMOTE_TERMS + _OFFICE_TERMS):
            answer = _remote_answer(question, remote_only)
        elif any(term in normalized for term in _PORTFOLIO_TERMS) and question.text_name:
            if portfolio_url:
                answer = QuestionnaireAnswer(
                    key=question.key,
                    text_name=question.text_name,
                    text=portfolio_url,
                )
        elif _is_professional_text_question(question):
            generated.append(question)
            continue
        if answer is None:
            manual.append(question.prompt)
        else:
            answers.append(answer)

    if manual:
        return QuestionnairePlan(
            manual_questions=tuple(
                [*manual, *(question.prompt for question in generated)]
            )
        )
    return QuestionnairePlan(
        answers=tuple(answers), generated_questions=tuple(generated)
    )
