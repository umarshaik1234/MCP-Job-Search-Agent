"""Adzuna Jobs API client and response normalization helpers."""

from __future__ import annotations

import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

from utils.job_helpers import extract_experience_range, rank_and_filter_by_experience

load_dotenv()

ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs"
DEFAULT_COUNTRY = "in"
SUPPORTED_COUNTRIES: dict[str, str] = {
    "in": "India",
    "us": "United States",
    "gb": "United Kingdom",
    "ca": "Canada",
    "au": "Australia",
    "de": "Germany",
    "sg": "Singapore",
}


class AdzunaConfigError(RuntimeError):
    """Raised when required Adzuna API credentials are missing."""


def _format_salary(salary_min: Any, salary_max: Any) -> str:
    """Formats numeric min/max salary values into a compact display string."""
    low = int(round(float(salary_min))) if isinstance(salary_min, (int, float)) else None
    high = int(round(float(salary_max))) if isinstance(salary_max, (int, float)) else None

    if low is not None and high is not None:
        if low == high:
            return f"{low:,}"
        return f"{low:,} - {high:,}"
    if low is not None:
        return f"From {low:,}"
    if high is not None:
        return f"Up to {high:,}"
    return ""


def _clean_snippet(text: Any, max_len: int = 260) -> str:
    """Collapses whitespace and truncates a job description for display."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def normalize_adzuna_job(raw_job: dict[str, Any]) -> dict[str, Any]:
    """Normalizes a single Adzuna job payload into the shared app schema."""
    company_info = raw_job.get("company")
    location_info = raw_job.get("location")
    company_name = (
        company_info.get("display_name")
        if isinstance(company_info, dict)
        else company_info
    )
    location_name = (
        location_info.get("display_name")
        if isinstance(location_info, dict)
        else location_info
    )

    title = str(raw_job.get("title") or "Untitled role").strip()
    full_description = str(raw_job.get("description") or "")
    exp_min, exp_max, exp_label = extract_experience_range(title, full_description)

    salary_min = raw_job.get("salary_min")
    salary_max = raw_job.get("salary_max")

    return {
        "title": title,
        "company": str(company_name or "Unknown company").strip(),
        "location": str(location_name or "Unknown location").strip(),
        "url": str(raw_job.get("redirect_url") or raw_job.get("url") or "").strip(),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary": _format_salary(salary_min, salary_max),
        "contract_time": str(raw_job.get("contract_time") or "").strip(),
        "contract_type": str(raw_job.get("contract_type") or "").strip(),
        "created": str(raw_job.get("created") or "").strip(),
        "description": _clean_snippet(full_description),
        "experience_min_years": exp_min,
        "experience_max_years": exp_max,
        "experience_label": exp_label,
    }


def search_jobs(
    keyword: str,
    location: str,
    country: str = DEFAULT_COUNTRY,
    results_per_page: int = 10,
    page: int = 1,
    timeout: int = 15,
    experience_years: int | None = None,
    strict_experience: bool = False,
) -> list[dict[str, Any]]:
    """Queries the Adzuna Jobs API and returns normalized job listings."""
    clean_keyword = keyword.strip()
    clean_location = location.strip()
    if not clean_keyword:
        raise ValueError("Job search keyword must not be empty.")
    if not clean_location:
        raise ValueError("Job search location must not be empty.")

    if experience_years is not None and (experience_years < 0 or experience_years > 40):
        raise ValueError("Experience years must be between 0 and 40.")

    app_id = os.getenv("ADZUNA_APP_ID", "").strip()
    app_key = os.getenv("ADZUNA_APP_KEY", "").strip()
    if not app_id or not app_key:
        raise AdzunaConfigError(
            "ADZUNA_APP_ID and ADZUNA_APP_KEY must be set in your .env file."
        )

    country_code = country.strip().lower() or DEFAULT_COUNTRY
    if country_code not in SUPPORTED_COUNTRIES:
        raise ValueError(
            f"Unsupported country '{country}'. Choose from: {', '.join(SUPPORTED_COUNTRIES)}."
        )

    safe_limit = max(1, min(int(results_per_page), 50))
    fetch_limit = (
        min(50, max(safe_limit * 2, 20)) if experience_years is not None else safe_limit
    )
    safe_page = max(1, int(page))

    url = f"{ADZUNA_BASE_URL}/{country_code}/search/{safe_page}"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": clean_keyword,
        "where": clean_location,
        "results_per_page": fetch_limit,
        "content-type": "application/json",
    }

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    raw_results = data.get("results", [])
    if not isinstance(raw_results, list):
        return []

    normalized = [
        normalize_adzuna_job(item) for item in raw_results if isinstance(item, dict)
    ]

    if experience_years is not None:
        normalized = rank_and_filter_by_experience(
            normalized,
            experience_years=experience_years,
            strict=strict_experience,
        )

    return normalized[:safe_limit]
