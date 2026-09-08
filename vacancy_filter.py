def title_rejection_reason(
    title: str, excluded_positions: tuple[str, ...]
) -> str | None:
    normalized = title.casefold()
    return next(
        (
            word
            for word in excluded_positions
            if word.casefold() in normalized
        ),
        None,
    )


def vacancy_rejection_reason(
    *,
    title: str,
    company: str,
    description: str,
    excluded_positions: tuple[str, ...],
    excluded_companies: tuple[str, ...],
    excluded_keywords: tuple[str, ...],
) -> str | None:
    title_match = title_rejection_reason(title, excluded_positions)
    if title_match:
        return f"position:{title_match}"
    normalized_company = company.casefold()
    company_match = next(
        (term for term in excluded_companies if term.casefold() in normalized_company),
        None,
    )
    if company_match:
        return f"company:{company_match}"
    normalized_description = description.casefold()
    keyword_match = next(
        (term for term in excluded_keywords if term.casefold() in normalized_description),
        None,
    )
    return f"keyword:{keyword_match}" if keyword_match else None
