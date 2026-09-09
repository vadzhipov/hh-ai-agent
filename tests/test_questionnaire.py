from dataclasses import replace

from config import load_settings
from questionnaire import (
    QuestionnaireOption,
    QuestionnaireQuestion,
    plan_questionnaire,
)
from tests.test_config import VALID_ENV, write_profile


def candidate(tmp_path):
    return load_settings(
        profile_path=write_profile(tmp_path), environ=VALID_ENV
    ).profile.candidate


def test_questionnaire_rejects_condition_hidden_behind_blocked_company(
    tmp_path,
) -> None:
    profile = replace(candidate(tmp_path), excluded_companies=("сбер",))
    question = QuestionnaireQuestion(
        key="task_1",
        prompt="Мы ищем сотрудника в Сбер. Вам подойдет?",
        options=(QuestionnaireOption("1", "Да", 0),),
    )

    plan = plan_questionnaire(
        (question,),
        profile,
        remote_only=True,
        portfolio_url="https://portfolio.example/candidate",
        enabled=True,
    )

    assert plan.rejection_reason == "questionnaire_company:сбер"
    assert plan.answers == ()


def test_questionnaire_answers_open_salary_without_inventing_a_number(
    tmp_path,
) -> None:
    profile = replace(candidate(tmp_path), salary_expectation="Без минимального порога")
    question = QuestionnaireQuestion(
        key="task_2",
        prompt="Укажите, пожалуйста, желаемый уровень дохода.",
        text_name="task_2_text",
    )

    plan = plan_questionnaire(
        (question,),
        profile,
        remote_only=True,
        portfolio_url="",
        enabled=True,
    )

    assert plan.manual_questions == ()
    assert plan.answers[0].text == "Готов обсудить уровень дохода по итогам интервью."


def test_questionnaire_uses_explicit_remote_only_preference(tmp_path) -> None:
    question = QuestionnaireQuestion(
        key="task_3",
        prompt="Готовы регулярно работать из офиса?",
        options=(
            QuestionnaireOption("yes", "Да", 0),
            QuestionnaireOption("no", "Нет", 1),
        ),
    )

    plan = plan_questionnaire(
        (question,),
        candidate(tmp_path),
        remote_only=True,
        portfolio_url="",
        enabled=True,
    )

    assert plan.answers[0].option_index == 1


def test_unknown_question_requires_manual_answer_and_does_not_partially_fill(
    tmp_path,
) -> None:
    salary = QuestionnaireQuestion(
        key="task_1",
        prompt="Какой доход вы ожидаете?",
        text_name="task_1_text",
    )
    legal = QuestionnaireQuestion(
        key="task_2",
        prompt="Есть ли у вас разрешение на работу в стране?",
        options=(
            QuestionnaireOption("yes", "Да", 0),
            QuestionnaireOption("no", "Нет", 1),
        ),
    )

    plan = plan_questionnaire(
        (salary, legal),
        candidate(tmp_path),
        remote_only=True,
        portfolio_url="",
        enabled=True,
    )

    assert plan.answers == ()
    assert plan.manual_questions == (legal.prompt,)


def test_questionnaire_auto_answers_are_opt_in(tmp_path) -> None:
    question = QuestionnaireQuestion(
        key="task_1",
        prompt="Какой доход вы ожидаете?",
        text_name="task_1_text",
    )

    plan = plan_questionnaire(
        (question,),
        candidate(tmp_path),
        remote_only=True,
        portfolio_url="",
        enabled=False,
    )

    assert plan.answers == ()
    assert plan.manual_questions == (question.prompt,)


def test_professional_text_questions_are_marked_for_grounded_generation(
    tmp_path,
) -> None:
    question = QuestionnaireQuestion(
        key="task_4",
        prompt="Как вы выстраиваете и поддерживаете дизайн-систему в Figma?",
        text_name="task_4_text",
    )

    plan = plan_questionnaire(
        (question,),
        candidate(tmp_path),
        remote_only=True,
        portfolio_url="",
        enabled=True,
    )

    assert plan.manual_questions == ()
    assert plan.generated_questions == (question,)


def test_sensitive_text_question_is_never_sent_for_generation(tmp_path) -> None:
    question = QuestionnaireQuestion(
        key="task_5",
        prompt="Укажите гражданство и есть ли разрешение на работу.",
        text_name="task_5_text",
    )

    plan = plan_questionnaire(
        (question,),
        candidate(tmp_path),
        remote_only=True,
        portfolio_url="",
        enabled=True,
    )

    assert plan.generated_questions == ()
    assert plan.manual_questions == (question.prompt,)
