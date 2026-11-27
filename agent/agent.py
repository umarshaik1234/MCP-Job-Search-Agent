"""Gemini-powered MCP job search agent and resume analyzer."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.job_helpers import deduplicate_jobs, parse_mcp_tool_result

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
SERVER_SCRIPT = PROJECT_ROOT / "mcp_server" / "server.py"

AGENT_SYSTEM_INSTRUCTION = (
    "You are an AI job search assistant connected to an MCP job-search server. "
    "Whenever the user asks to find, search, or filter jobs, call the available MCP tools "
    "(such as search_jobs_mcp, filter_jobs_mcp, and deduplicate_jobs_mcp). "
    "When the user specifies years of experience (or 'fresher' -> 0), pass experience_years "
    "as an integer to search_jobs_mcp or filter_jobs_mcp. "
    "Never invent job titles, companies, salaries, or URLs. "
    "After gathering results from tools, summarize the most relevant opportunities clearly "
    "with role, company, location, experience fit, and apply link."
)


@dataclass(slots=True)
class AgentToolStep:
    """Trace record for a single MCP tool invocation inside the agent loop."""

    tool_name: str
    arguments: dict[str, Any]
    result_count: int
    is_error: bool = False
    error_message: str = ""


@dataclass(slots=True)
class AgentResult:
    """Final output returned by the MCP job search agent."""

    answer: str
    jobs: list[dict[str, Any]] = field(default_factory=list)
    steps: list[AgentToolStep] = field(default_factory=list)


def get_gemini_client() -> genai.Client:
    """Creates a configured Gemini client from GEMINI_API_KEY."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from your .env file.")
    return genai.Client(api_key=api_key)


def get_server_params() -> StdioServerParameters:
    """Builds stdio server launch parameters using the active Python interpreter."""
    forwarded_env = {
        key: os.environ[key]
        for key in ("ADZUNA_APP_ID", "ADZUNA_APP_KEY", "PYTHONPATH", "PYTHONUTF8")
        if key in os.environ
    }
    forwarded_env.setdefault("PYTHONUTF8", "1")

    return StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        env=forwarded_env,
        cwd=str(PROJECT_ROOT),
    )


def _sanitize_json_schema(schema: Any) -> dict[str, Any]:
    """Converts MCP JSON Schema into a Gemini-compatible parameter schema."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    blocked_keys = {"$schema", "additionalProperties", "$defs", "definitions", "default"}
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in blocked_keys:
            continue
        if key == "anyOf" and isinstance(value, list):
            non_null = [
                item
                for item in value
                if isinstance(item, dict) and item.get("type") != "null"
            ]
            if len(non_null) == 1:
                cleaned.update(_sanitize_json_schema(non_null[0]))
                continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                prop_name: _sanitize_json_schema(prop_schema)
                for prop_name, prop_schema in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            cleaned[key] = _sanitize_json_schema(value)
        else:
            cleaned[key] = value

    if "type" not in cleaned and "properties" in cleaned:
        cleaned["type"] = "object"
    return cleaned


def mcp_tools_to_gemini_tool(mcp_tools: list[Any]) -> types.Tool:
    """Builds a Gemini Tool object from MCP server tool definitions."""
    declarations = [
        types.FunctionDeclaration(
            name=tool.name,
            description=getattr(tool, "description", "") or f"MCP tool {tool.name}",
            parameters_json_schema=_sanitize_json_schema(
                getattr(tool, "inputSchema", None)
            ),
        )
        for tool in mcp_tools
    ]
    return types.Tool(function_declarations=declarations)


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Parses JSON from model text, tolerating optional markdown code fences."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from resume analysis.")
    return parsed


def _parse_experience_years(value: Any) -> int | None:
    """Coerces a model-returned experience_years value to a bounded integer."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        years = int(round(value))
        return years if 0 <= years <= 40 else None
    match = re.search(r"\b(\d{1,2})\b", str(value))
    if match:
        years = int(match.group(1))
        return years if 0 <= years <= 40 else None
    return None


def analyze_resume(
    resume_text: str,
    model_name: str = DEFAULT_MODEL,
    client: genai.Client | None = None,
) -> dict[str, Any]:
    """Analyzes resume text with Gemini and returns structured search metadata."""
    clean_resume = resume_text.strip()
    if not clean_resume:
        raise ValueError("Resume text is empty.")

    gemini_client = client or get_gemini_client()
    prompt = f"""Analyze this resume and extract job search parameters.

Resume:
{clean_resume[:12000]}

Return a JSON object with this exact shape:
{{
    "job_titles": ["Python Developer", "Backend Engineer"],
    "skills": ["Python", "SQL"],
    "technologies": ["FastAPI", "Docker"],
    "location": "Hyderabad",
    "experience_level": "Fresher",
    "experience_years": 0
}}

Rules:
- Extract only skills and technologies supported by the resume.
- Suggest 3 to 5 realistic job titles suited to the candidate.
- If location is missing, default to "Hyderabad".
- Estimate total professional years of experience as an integer in experience_years (0 for freshers/students).
- If experience level is unclear, infer conservatively (e.g., "Fresher", "Junior", "Mid-level", "Senior").
"""

    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    data = _extract_json_payload(response.text or "")

    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    experience_level = str(data.get("experience_level") or "Not specified").strip()
    experience_years = _parse_experience_years(data.get("experience_years"))
    if experience_years is None and "fresher" in experience_level.lower():
        experience_years = 0

    return {
        "job_titles": _string_list(data.get("job_titles")),
        "skills": _string_list(data.get("skills")),
        "technologies": _string_list(data.get("technologies")),
        "location": str(data.get("location") or "Hyderabad").strip() or "Hyderabad",
        "experience_level": experience_level,
        "experience_years": experience_years,
    }


async def search_jobs_via_mcp(
    keyword: str,
    location: str,
    country: str = "in",
    results_per_page: int = 10,
    experience_years: int | None = None,
    strict_experience: bool = False,
) -> list[dict[str, Any]]:
    """Executes a single job search through the MCP server."""
    return await search_jobs_for_roles(
        job_titles=[keyword],
        location=location,
        country=country,
        results_per_page=results_per_page,
        experience_years=experience_years,
        strict_experience=strict_experience,
    )


async def search_jobs_for_roles(
    job_titles: list[str],
    location: str,
    country: str = "in",
    results_per_page: int = 10,
    experience_years: int | None = None,
    strict_experience: bool = False,
) -> list[dict[str, Any]]:
    """Searches multiple job titles in one MCP session and returns deduplicated jobs."""
    clean_titles = [title.strip() for title in job_titles if title and title.strip()]
    if not clean_titles:
        return []

    all_jobs: list[dict[str, Any]] = []
    errors: list[str] = []

    async with stdio_client(get_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for title in clean_titles:
                arguments: dict[str, Any] = {
                    "keyword": title,
                    "location": location.strip(),
                    "country": country,
                    "results_per_page": results_per_page,
                    "strict_experience": strict_experience,
                }
                if experience_years is not None:
                    arguments["experience_years"] = experience_years

                result = await session.call_tool("search_jobs_mcp", arguments=arguments)
                if getattr(result, "isError", False):
                    err_parts = [
                        getattr(c, "text", "")
                        for c in getattr(result, "content", []) or []
                        if getattr(c, "text", None)
                    ]
                    errors.append(f"{title}: {' '.join(err_parts) or 'MCP tool error'}")
                    continue
                all_jobs.extend(parse_mcp_tool_result(result))

    if not all_jobs and errors:
        raise RuntimeError("; ".join(errors))

    return deduplicate_jobs(all_jobs)


def _tool_result_text(result: Any) -> str:
    """Extracts plain text error or fallback message from an MCP CallToolResult."""
    return " ".join(
        str(getattr(item, "text", ""))
        for item in getattr(result, "content", []) or []
        if getattr(item, "text", None)
    ).strip()


def _normalize_tool_arguments(
    tool_name: str,
    raw_args: dict[str, Any],
    collected_jobs: list[dict[str, Any]],
    default_experience_years: int | None = None,
    default_strict_experience: bool = False,
) -> dict[str, Any]:
    """Normalizes common LLM argument variations before calling MCP tools."""
    args = dict(raw_args)

    if tool_name == "search_jobs_mcp":
        if not args.get("keyword"):
            for alias in ("query", "role", "job_title", "title"):
                if args.get(alias):
                    args["keyword"] = str(args.pop(alias))
                    break
            else:
                args["keyword"] = "Software Developer"
        if not args.get("location"):
            for alias in ("city", "place", "region"):
                if args.get(alias):
                    args["location"] = str(args.pop(alias))
                    break
            else:
                args["location"] = "India"
        if "experience_years" in args:
            parsed_exp = _parse_experience_years(args["experience_years"])
            if parsed_exp is None:
                args.pop("experience_years")
            else:
                args["experience_years"] = parsed_exp
        if "experience_years" not in args and default_experience_years is not None:
            args["experience_years"] = default_experience_years
        if default_strict_experience and "strict_experience" not in args:
            args["strict_experience"] = True

    elif tool_name in {"filter_jobs_mcp", "deduplicate_jobs_mcp"}:
        jobs_arg = args.get("jobs")
        if not isinstance(jobs_arg, list) or not all(isinstance(j, dict) for j in jobs_arg):
            args["jobs"] = list(collected_jobs)
        if tool_name == "filter_jobs_mcp":
            if "experience_years" in args:
                parsed_exp = _parse_experience_years(args["experience_years"])
                if parsed_exp is None:
                    args.pop("experience_years")
                else:
                    args["experience_years"] = parsed_exp
            if "experience_years" not in args and default_experience_years is not None:
                args["experience_years"] = default_experience_years
            if default_strict_experience and "strict_experience" not in args:
                args["strict_experience"] = True

    return args


async def run_job_agent(
    user_request: str,
    model_name: str = DEFAULT_MODEL,
    max_steps: int = 6,
    client: genai.Client | None = None,
    on_step: Callable[[AgentToolStep], None] | None = None,
    experience_years: int | None = None,
    strict_experience: bool = False,
) -> AgentResult:
    """Runs a multi-step Gemini + MCP tool-calling loop for a natural-language query."""
    clean_request = user_request.strip()
    if not clean_request:
        raise ValueError("User request must not be empty.")

    effective_prompt = clean_request
    if experience_years is not None:
        effective_prompt = (
            f"{clean_request}\n"
            f"(Candidate preference: {experience_years} year(s) of experience.)"
        )

    gemini_client = client or get_gemini_client()
    collected_jobs: list[dict[str, Any]] = []
    steps: list[AgentToolStep] = []

    async with stdio_client(get_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            gemini_tool = mcp_tools_to_gemini_tool(tools_response.tools)

            config = types.GenerateContentConfig(
                system_instruction=AGENT_SYSTEM_INSTRUCTION,
                tools=[gemini_tool],
                temperature=0.2,
            )
            contents: list[Any] = [effective_prompt]

            for _ in range(max_steps):
                response = await asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=model_name,
                    contents=contents,
                    config=config,
                )

                function_calls = getattr(response, "function_calls", None) or []
                if not function_calls:
                    answer = (getattr(response, "text", None) or "").strip()
                    if not answer and collected_jobs:
                        answer = f"Found {len(deduplicate_jobs(collected_jobs))} matching jobs."
                    return AgentResult(
                        answer=answer or "No response generated.",
                        jobs=deduplicate_jobs(collected_jobs),
                        steps=steps,
                    )

                candidates = getattr(response, "candidates", None) or []
                if candidates and getattr(candidates[0], "content", None) is not None:
                    contents.append(candidates[0].content)

                response_parts: list[types.Part] = []
                for fn_call in function_calls:
                    args = _normalize_tool_arguments(
                        fn_call.name,
                        dict(fn_call.args or {}),
                        collected_jobs,
                        default_experience_years=experience_years,
                        default_strict_experience=strict_experience,
                    )
                    tool_result = await session.call_tool(fn_call.name, arguments=args)
                    is_error = bool(getattr(tool_result, "isError", False))

                    if is_error:
                        err_msg = _tool_result_text(tool_result) or "MCP tool failed."
                        payload: dict[str, Any] = {"error": err_msg}
                        step = AgentToolStep(
                            tool_name=fn_call.name,
                            arguments=args,
                            result_count=0,
                            is_error=True,
                            error_message=err_msg,
                        )
                    else:
                        parsed_jobs = parse_mcp_tool_result(tool_result)
                        if parsed_jobs:
                            collected_jobs.extend(parsed_jobs)
                            payload = {"jobs": parsed_jobs, "count": len(parsed_jobs)}
                        else:
                            payload = {"result": _tool_result_text(tool_result)}
                        step = AgentToolStep(
                            tool_name=fn_call.name,
                            arguments=args,
                            result_count=len(parsed_jobs),
                        )

                    steps.append(step)
                    if on_step is not None:
                        on_step(step)

                    response_parts.append(
                        types.Part.from_function_response(
                            name=fn_call.name,
                            response=payload,
                        )
                    )

                contents.append(types.Content(role="user", parts=response_parts))

    unique_jobs = deduplicate_jobs(collected_jobs)
    return AgentResult(
        answer=(
            f"Completed {len(steps)} MCP tool step(s) and collected "
            f"{len(unique_jobs)} unique job listing(s)."
        ),
        jobs=unique_jobs,
        steps=steps,
    )


def _print_cli_results(result: AgentResult) -> None:
    """Formats agent output cleanly for terminal usage."""
    print("\n" + "=" * 70)
    print("AGENT SUMMARY")
    print("=" * 70)
    print(result.answer)

    if result.jobs:
        print("\n" + "=" * 70)
        print(f"JOB LISTINGS ({len(result.jobs)})")
        print("=" * 70)
        for idx, job in enumerate(result.jobs, start=1):
            print(f"\n{idx}. {job.get('title', 'N/A')}")
            print(f"   Company : {job.get('company', 'N/A')}")
            print(f"   Location: {job.get('location', 'N/A')}")
            if job.get("experience_label"):
                print(f"   Exp     : {job['experience_label']}")
            if job.get("salary"):
                print(f"   Salary  : {job['salary']}")
            if job.get("url"):
                print(f"   Apply   : {job['url']}")


async def main() -> None:
    """Interactive CLI entrypoint for the MCP job search agent."""
    print("Job Search MCP Agent (type 'exit' to quit)")
    user_request = input("\nWhat jobs are you looking for?\n> ").strip()
    if not user_request or user_request.lower() in {"exit", "quit"}:
        return

    exp_raw = input("Years of experience (press Enter for any):\n> ").strip()
    exp_years = int(exp_raw) if exp_raw.isdigit() else None

    result = await run_job_agent(user_request, experience_years=exp_years)
    _print_cli_results(result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent stopped.")
