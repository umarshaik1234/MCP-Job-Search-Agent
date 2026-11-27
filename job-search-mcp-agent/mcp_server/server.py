"""MCP server exposing job search, filtering, and deduplication tools."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.mcpserver import MCPServer

from services.adzuna_service import DEFAULT_COUNTRY, search_jobs
from utils.job_helpers import deduplicate_jobs, filter_jobs

mcp = MCPServer("Job Search MCP Server")


@mcp.tool()
def search_jobs_mcp(
    keyword: str,
    location: str,
    country: str = DEFAULT_COUNTRY,
    results_per_page: int = 10,
    experience_years: int | None = None,
    strict_experience: bool = False,
) -> list[dict[str, Any]]:
    """Search for live job listings using the Adzuna API, optionally matching years of experience."""
    return search_jobs(
        keyword=keyword,
        location=location,
        country=country,
        results_per_page=results_per_page,
        experience_years=experience_years,
        strict_experience=strict_experience,
    )


@mcp.tool()
def filter_jobs_mcp(
    jobs: list[dict[str, Any]],
    keyword: str = "",
    location: str = "",
    min_salary: float | None = None,
    experience_years: int | None = None,
    strict_experience: bool = False,
) -> list[dict[str, Any]]:
    """Filter job listings by keyword, location, minimum salary, and years of experience."""
    return filter_jobs(
        jobs=jobs,
        keyword=keyword,
        location=location,
        min_salary=min_salary,
        experience_years=experience_years,
        strict_experience=strict_experience,
    )


@mcp.tool()
def deduplicate_jobs_mcp(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate job listings by URL or title, company, and location."""
    return deduplicate_jobs(jobs)


if __name__ == "__main__":
    mcp.run()
