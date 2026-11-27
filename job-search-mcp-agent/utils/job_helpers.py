"""Shared job filtering, experience extraction, deduplication, and MCP helpers."""

from __future__ import annotations

import json
import re
from typing import Any

_RANGE_PATTERNS = (
    re.compile(
        r"\b(\d{1,2})\s*(?:-|–|—|to)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2})\s*(?:-|–|—|to)\s*(\d{1,2})\s*(?:years?|yrs?)\s+(?:of\s+)?experience\b",
        re.IGNORECASE,
    ),
)

_MIN_PATTERNS = (
    re.compile(
        r"\b(?:min(?:imum)?\.?\s*|at\s+least\s+)?(\d{1,2})\s*\+\s*(?:years?|yrs?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:min(?:imum)?\.?\s*|at\s+least\s+)(\d{1,2})\s*(?:years?|yrs?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2})\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+|work\s+|hands[- ]on\s+)?experience\b",
        re.IGNORECASE,
    ),
)

_FRESHER_PATTERN = re.compile(
    r"\b(?:fresher|fresh\s+graduate|entry[- ]level|junior|jr\.?|trainee|intern(?:ship)?)\b",
    re.IGNORECASE,
)

_SENIOR_PATTERN = re.compile(
    r"\b(?:senior|sr\.?|lead|principal|staff|architect)\b",
    re.IGNORECASE,
)


def extract_experience_range(
    title: str,
    description: str = "",
) -> tuple[int | None, int | None, str]:
    """Extracts (min_years, max_years, display_label) from job title and description."""
    combined = f"{title} {description}".strip()
    if not combined:
        return None, None, ""

    for pattern in _RANGE_PATTERNS:
        match = pattern.search(combined)
        if match:
            low = int(match.group(1))
            high = int(match.group(2))
            if low > high:
                low, high = high, low
            if high <= 30:
                return low, high, f"{low}-{high} yrs"

    for pattern in _MIN_PATTERNS:
        match = pattern.search(combined)
        if match:
            years = int(match.group(1))
            if years <= 30:
                return years, None, f"{years}+ yrs"

    if _FRESHER_PATTERN.search(title) or _FRESHER_PATTERN.search(description):
        return 0, 1, "Fresher (0-1 yr)"

    if _SENIOR_PATTERN.search(title):
        return 5, None, "Senior (5+ yrs est.)"

    return None, None, ""


def matches_experience(
    job: dict[str, Any],
    target_years: int,
    strict: bool = False,
) -> bool:
    """Checks whether a job listing fits the candidate's years of experience."""
    exp_min = job.get("experience_min_years")
    exp_max = job.get("experience_max_years")

    if exp_min is None and exp_max is None:
        exp_min, exp_max, _ = extract_experience_range(
            str(job.get("title") or ""),
            str(job.get("description") or ""),
        )

    if exp_min is None and exp_max is None:
        return not strict

    if exp_min is not None and exp_max is not None:
        return exp_min <= target_years <= exp_max

    if exp_min is not None:
        if target_years < exp_min:
            return False
        # Avoid treating a 0-1 yr entry query as matching a 10-year senior if strict
        if strict and target_years > exp_min + 5:
            return False
        return True

    assert exp_max is not None
    return target_years <= exp_max


def rank_and_filter_by_experience(
    jobs: list[dict[str, Any]],
    experience_years: int | None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Filters incompatible experience requirements and prioritizes explicit matches."""
    if experience_years is None:
        return jobs

    explicit_matches: list[dict[str, Any]] = []
    unspecified_matches: list[dict[str, Any]] = []

    for job in jobs:
        if not isinstance(job, dict):
            continue
        if not matches_experience(job, experience_years, strict=strict):
            continue

        has_explicit = (
            job.get("experience_min_years") is not None
            or job.get("experience_max_years") is not None
            or bool(job.get("experience_label"))
        )
        if has_explicit:
            explicit_matches.append(job)
        elif not strict:
            unspecified_matches.append(job)

    return explicit_matches + unspecified_matches


def deduplicate_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Removes duplicate job listings while preserving first-seen order."""
    unique: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            continue
        url = str(job.get("url") or "").strip()
        if url:
            key = f"url:{url}"
        else:
            title = str(job.get("title") or "").strip().lower()
            company = str(job.get("company") or "").strip().lower()
            location = str(job.get("location") or "").strip().lower()
            key = f"meta:{title}|{company}|{location}"
        if key not in unique:
            unique[key] = job
    return list(unique.values())


def filter_jobs(
    jobs: list[dict[str, Any]],
    keyword: str = "",
    location: str = "",
    min_salary: float | None = None,
    experience_years: int | None = None,
    strict_experience: bool = False,
) -> list[dict[str, Any]]:
    """Filters normalized job listings by keyword, location, salary, and experience."""
    clean_kw = keyword.strip().lower()
    clean_loc = location.strip().lower()

    filtered: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue

        if clean_kw:
            haystack = " ".join(
                str(job.get(field) or "")
                for field in ("title", "company", "description", "experience_label")
            ).lower()
            if clean_kw not in haystack:
                continue

        if clean_loc:
            job_loc = str(job.get("location") or "").lower()
            if clean_loc not in job_loc:
                continue

        if min_salary is not None:
            sal_max = job.get("salary_max")
            sal_min = job.get("salary_min")
            best_salary = sal_max if isinstance(sal_max, (int, float)) else sal_min
            if not isinstance(best_salary, (int, float)) or best_salary < min_salary:
                continue

        filtered.append(job)

    if experience_years is not None:
        return rank_and_filter_by_experience(
            filtered,
            experience_years=experience_years,
            strict=strict_experience,
        )

    return filtered


def parse_mcp_tool_result(result: Any) -> list[dict[str, Any]]:
    """Extracts a list of job dicts from an MCP CallToolResult."""
    jobs: list[dict[str, Any]] = []

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and isinstance(structured.get("result"), list):
        for item in structured["result"]:
            if isinstance(item, dict):
                jobs.append(item)
        if jobs:
            return jobs
    elif isinstance(structured, list):
        for item in structured:
            if isinstance(item, dict):
                jobs.append(item)
        if jobs:
            return jobs

    for content in getattr(result, "content", []) or []:
        text = getattr(content, "text", None)
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, list):
            jobs.extend(item for item in parsed if isinstance(item, dict))
        elif isinstance(parsed, dict):
            jobs.append(parsed)

    return jobs
