"""Unit and Streamlit AppTest suite for the Job Search MCP Agent."""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import (
    _normalize_tool_arguments,
    _sanitize_json_schema,
    analyze_resume,
    get_server_params,
    mcp_tools_to_gemini_tool,
    run_job_agent,
)
from mcp_server.server import mcp as mcp_server
from services.adzuna_service import AdzunaConfigError, normalize_adzuna_job, search_jobs
from utils.job_helpers import (
    deduplicate_jobs,
    extract_experience_range,
    filter_jobs,
    matches_experience,
    parse_mcp_tool_result,
)
from utils.resume_parser import extract_resume_text


class TestAdzunaService(unittest.TestCase):
    def test_normalize_adzuna_job(self) -> None:
        raw = {
            "title": "  Python Developer ",
            "company": {"display_name": "Acme Corp"},
            "location": {"display_name": "Hyderabad, Telangana"},
            "redirect_url": "https://example.com/job/1",
            "salary_min": 600000,
            "salary_max": 1200000,
            "contract_time": "full_time",
            "description": "Build   APIs\nwith Python  and FastAPI. Requires 2-4 years experience.",
        }
        job = normalize_adzuna_job(raw)
        self.assertEqual(job["title"], "Python Developer")
        self.assertEqual(job["company"], "Acme Corp")
        self.assertEqual(job["location"], "Hyderabad, Telangana")
        self.assertEqual(job["url"], "https://example.com/job/1")
        self.assertEqual(job["salary"], "600,000 - 1,200,000")
        self.assertEqual(job["experience_min_years"], 2)
        self.assertEqual(job["experience_max_years"], 4)
        self.assertEqual(job["experience_label"], "2-4 yrs")
        self.assertEqual(
            job["description"],
            "Build APIs with Python and FastAPI. Requires 2-4 years experience.",
        )

    def test_search_jobs_validation(self) -> None:
        with self.assertRaises(ValueError):
            search_jobs("", "Hyderabad")
        with self.assertRaises(ValueError):
            search_jobs("Python", " ")
        with self.assertRaises(ValueError):
            search_jobs("Python", "Hyderabad", experience_years=-1)

        with patch.dict(os.environ, {"ADZUNA_APP_ID": "", "ADZUNA_APP_KEY": ""}, clear=False):
            with self.assertRaises(AdzunaConfigError):
                search_jobs("Python", "Hyderabad")

    def test_search_jobs_mocked_http_with_experience(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Senior AI Engineer",
                    "company": {"display_name": "BigTech"},
                    "location": {"display_name": "Bengaluru"},
                    "redirect_url": "https://example.com/sr",
                    "description": "Requires 6+ years of production ML experience.",
                },
                {
                    "title": "AI Engineer",
                    "company": {"display_name": "Future Labs"},
                    "location": {"display_name": "Bengaluru"},
                    "redirect_url": "https://example.com/ai",
                    "description": "Looking for 1-3 years of Python and ML experience.",
                },
                {
                    "title": "ML Specialist",
                    "company": {"display_name": "General Co"},
                    "location": {"display_name": "Bengaluru"},
                    "redirect_url": "https://example.com/gen",
                    "description": "Work on applied NLP pipelines.",
                },
            ]
        }
        with (
            patch.dict(os.environ, {"ADZUNA_APP_ID": "id", "ADZUNA_APP_KEY": "key"}, clear=False),
            patch("services.adzuna_service.requests.get", return_value=mock_response) as mock_get,
        ):
            jobs = search_jobs(
                "AI Engineer",
                "Bengaluru",
                country="IN",
                results_per_page=5,
                experience_years=2,
            )

        mock_get.assert_called_once()
        # Senior 6+ yrs filtered out; explicit 1-3 yrs ranked ahead of unspecified
        self.assertEqual([j["title"] for j in jobs], ["AI Engineer", "ML Specialist"])

        with (
            patch.dict(os.environ, {"ADZUNA_APP_ID": "id", "ADZUNA_APP_KEY": "key"}, clear=False),
            patch("services.adzuna_service.requests.get", return_value=mock_response),
        ):
            strict_jobs = search_jobs(
                "AI Engineer",
                "Bengaluru",
                experience_years=2,
                strict_experience=True,
            )
        self.assertEqual([j["title"] for j in strict_jobs], ["AI Engineer"])


class TestJobHelpers(unittest.TestCase):
    def test_extract_experience_range_patterns(self) -> None:
        self.assertEqual(
            extract_experience_range("Python Dev", "2 to 5 years of experience"),
            (2, 5, "2-5 yrs"),
        )
        self.assertEqual(
            extract_experience_range("Backend Engineer", "Minimum 3 years in Python"),
            (3, None, "3+ yrs"),
        )
        self.assertEqual(
            extract_experience_range("Graduate Trainee", "Open to freshers"),
            (0, 1, "Fresher (0-1 yr)"),
        )
        self.assertEqual(
            extract_experience_range("Senior Platform Engineer", "Build services"),
            (5, None, "Senior (5+ yrs est.)"),
        )

    def test_matches_experience(self) -> None:
        job_range = {"experience_min_years": 2, "experience_max_years": 4}
        self.assertTrue(matches_experience(job_range, 3))
        self.assertFalse(matches_experience(job_range, 1))
        self.assertFalse(matches_experience(job_range, 6))

        job_unspecified = {"title": "Developer", "description": "Python role"}
        self.assertTrue(matches_experience(job_unspecified, 2, strict=False))
        self.assertFalse(matches_experience(job_unspecified, 2, strict=True))

    def test_deduplicate_jobs(self) -> None:
        jobs = [
            {"title": "Dev", "company": "A", "location": "Hyd", "url": "https://x/1"},
            {"title": "Dev dup", "company": "A", "location": "Hyd", "url": "https://x/1"},
            {"title": "ML", "company": "B", "location": "Blr", "url": ""},
            {"title": "ml", "company": "b", "location": "blr", "url": ""},
        ]
        deduped = deduplicate_jobs(jobs)
        self.assertEqual(len(deduped), 2)

    def test_filter_jobs(self) -> None:
        jobs = [
            {
                "title": "Python Backend",
                "company": "Alpha",
                "location": "Hyderabad",
                "description": "FastAPI microservices (2-4 years experience)",
                "salary_min": 900000,
                "salary_max": 1200000,
                "experience_min_years": 2,
                "experience_max_years": 4,
                "experience_label": "2-4 yrs",
            },
            {
                "title": "Java Developer",
                "company": "Beta",
                "location": "Pune",
                "description": "Spring Boot",
                "salary_min": 500000,
                "salary_max": 700000,
                "experience_min_years": 5,
                "experience_max_years": None,
                "experience_label": "5+ yrs",
            },
        ]
        self.assertEqual(len(filter_jobs(jobs, keyword="fastapi")), 1)
        self.assertEqual(len(filter_jobs(jobs, location="pune")), 1)
        self.assertEqual(len(filter_jobs(jobs, min_salary=800000)), 1)
        self.assertEqual(len(filter_jobs(jobs, experience_years=3)), 1)
        self.assertEqual(filter_jobs(jobs, experience_years=3)[0]["title"], "Python Backend")

    def test_parse_mcp_tool_result(self) -> None:
        text_res = SimpleNamespace(
            content=[SimpleNamespace(text=json.dumps([{"title": "Role A"}]))],
            structuredContent=None,
        )
        self.assertEqual(parse_mcp_tool_result(text_res), [{"title": "Role A"}])

        structured_res = SimpleNamespace(
            content=[],
            structuredContent={"result": [{"title": "Role B"}]},
        )
        self.assertEqual(parse_mcp_tool_result(structured_res), [{"title": "Role B"}])


class TestResumeParser(unittest.TestCase):
    def test_extract_resume_text_reuses_stream(self) -> None:
        fake_page = MagicMock()
        fake_page.extract_text.return_value = "Python, LangChain, Hyderabad"
        fake_reader = MagicMock()
        fake_reader.pages = [fake_page]

        buf = io.BytesIO(b"%PDF-1.4 mock")
        with patch("utils.resume_parser.PdfReader", return_value=fake_reader):
            first = extract_resume_text(buf)
            second = extract_resume_text(buf)

        self.assertEqual(first, "Python, LangChain, Hyderabad")
        self.assertEqual(second, "Python, LangChain, Hyderabad")


class TestMcpServerTools(unittest.TestCase):
    def test_server_registers_and_runs_tools(self) -> None:
        tools = asyncio.run(mcp_server.list_tools())
        tool_names = {tool.name for tool in tools}
        self.assertEqual(
            tool_names,
            {"search_jobs_mcp", "filter_jobs_mcp", "deduplicate_jobs_mcp"},
        )

        filter_res = asyncio.run(
            mcp_server.call_tool(
                "filter_jobs_mcp",
                {
                    "jobs": [
                        {
                            "title": "Python Engineer",
                            "company": "Acme",
                            "location": "Hyderabad",
                            "experience_min_years": 1,
                            "experience_max_years": 3,
                            "experience_label": "1-3 yrs",
                        },
                        {
                            "title": "Staff Python Engineer",
                            "company": "Beta",
                            "location": "Hyderabad",
                            "experience_min_years": 8,
                            "experience_max_years": None,
                            "experience_label": "8+ yrs",
                        },
                    ],
                    "keyword": "python",
                    "experience_years": 2,
                },
            )
        )
        if isinstance(filter_res, tuple):
            parsed = parse_mcp_tool_result(
                SimpleNamespace(content=filter_res[0], structuredContent=filter_res[1])
            )
        else:
            parsed = parse_mcp_tool_result(filter_res)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["title"], "Python Engineer")

    def test_stdio_mcp_roundtrip(self) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async def _roundtrip() -> tuple[set[str], list[dict]]:
            async with stdio_client(get_server_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_resp = await session.list_tools()
                    names = {t.name for t in tools_resp.tools}
                    res = await session.call_tool(
                        "deduplicate_jobs_mcp",
                        arguments={
                            "jobs": [
                                {"title": "Dev", "company": "A", "location": "Hyd", "url": "https://a/1"},
                                {"title": "Dev", "company": "A", "location": "Hyd", "url": "https://a/1"},
                            ]
                        },
                    )
                    return names, parse_mcp_tool_result(res)

        names, deduped = asyncio.run(_roundtrip())
        self.assertEqual(names, {"search_jobs_mcp", "filter_jobs_mcp", "deduplicate_jobs_mcp"})
        self.assertEqual(len(deduped), 1)


class TestAgentLogic(unittest.TestCase):
    def test_sanitize_and_convert_mcp_tools(self) -> None:
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keyword": {"type": "string", "default": ""},
                "experience_years": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
            },
            "required": ["keyword"],
        }
        cleaned = _sanitize_json_schema(schema)
        self.assertNotIn("$schema", cleaned)
        self.assertNotIn("additionalProperties", cleaned)
        self.assertEqual(cleaned["properties"]["experience_years"]["type"], "integer")

        mcp_tool = SimpleNamespace(name="search_jobs_mcp", description="Search", inputSchema=schema)
        gemini_tool = mcp_tools_to_gemini_tool([mcp_tool])
        self.assertEqual(len(gemini_tool.function_declarations), 1)
        self.assertEqual(gemini_tool.function_declarations[0].name, "search_jobs_mcp")

    def test_analyze_resume(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = SimpleNamespace(
            text=(
                '```json\n{"job_titles": ["Python Developer"], "skills": ["Python"], '
                '"technologies": ["FastAPI"], "location": "Hyderabad", '
                '"experience_level": "Junior", "experience_years": 2}\n```'
            )
        )
        result = analyze_resume("Python FastAPI developer in Hyderabad, 2 years exp", client=mock_client)
        self.assertEqual(result["job_titles"], ["Python Developer"])
        self.assertEqual(result["skills"], ["Python"])
        self.assertEqual(result["technologies"], ["FastAPI"])
        self.assertEqual(result["experience_years"], 2)

    def test_run_job_agent_tool_loop(self) -> None:
        mock_client = MagicMock()
        fn_call = SimpleNamespace(
            name="search_jobs_mcp",
            args={"keyword": "Python", "location": "Hyderabad", "experience_years": 2},
        )
        first_resp = SimpleNamespace(
            function_calls=[fn_call],
            candidates=[SimpleNamespace(content={"role": "model", "parts": []})],
            text=None,
        )
        second_resp = SimpleNamespace(
            function_calls=[],
            candidates=[],
            text="Found 1 Python role in Hyderabad for 2 years experience.",
        )
        mock_client.models.generate_content.side_effect = [first_resp, second_resp]

        fake_session = MagicMock()
        fake_session.initialize = AsyncMock()
        fake_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="search_jobs_mcp",
                        description="Search jobs",
                        inputSchema={"type": "object", "properties": {"keyword": {"type": "string"}}},
                    )
                ]
            )
        )
        fake_session.call_tool = AsyncMock(
            return_value=SimpleNamespace(
                isError=False,
                content=[
                    SimpleNamespace(
                        text=json.dumps(
                            [
                                {
                                    "title": "Python Dev",
                                    "company": "Acme",
                                    "location": "Hyderabad",
                                    "url": "https://a/1",
                                    "experience_label": "1-3 yrs",
                                }
                            ]
                        )
                    )
                ],
                structuredContent=None,
            )
        )

        class _FakeCM:
            def __init__(self, value):
                self._value = value

            async def __aenter__(self):
                return self._value

            async def __aexit__(self, *args):
                return False

        with (
            patch("agent.agent.stdio_client", return_value=_FakeCM((None, None))),
            patch("agent.agent.ClientSession", return_value=_FakeCM(fake_session)),
        ):
            res = asyncio.run(
                run_job_agent(
                    "Find Python jobs in Hyderabad",
                    client=mock_client,
                    experience_years=2,
                )
            )

        self.assertEqual(res.answer, "Found 1 Python role in Hyderabad for 2 years experience.")
        self.assertEqual(len(res.jobs), 1)
        self.assertEqual(res.jobs[0]["title"], "Python Dev")
        self.assertEqual(len(res.steps), 1)
        self.assertEqual(res.steps[0].tool_name, "search_jobs_mcp")

    def test_normalize_tool_arguments(self) -> None:
        search_args = _normalize_tool_arguments(
            "search_jobs_mcp",
            {"role": "Backend Engineer", "city": "Pune", "experience_years": "3 years"},
            [],
        )
        self.assertEqual(
            search_args,
            {"keyword": "Backend Engineer", "location": "Pune", "experience_years": 3},
        )

        default_args = _normalize_tool_arguments(
            "search_jobs_mcp",
            {"keyword": "Python Developer", "location": "Hyderabad"},
            [],
            default_experience_years=2,
            default_strict_experience=True,
        )
        self.assertEqual(default_args["experience_years"], 2)
        self.assertTrue(default_args["strict_experience"])

        sample_jobs = [{"title": "Dev", "company": "Acme"}]
        filter_args = _normalize_tool_arguments(
            "filter_jobs_mcp",
            {"keyword": "Dev"},
            sample_jobs,
            default_experience_years=2,
        )
        self.assertEqual(filter_args["jobs"], sample_jobs)
        self.assertEqual(filter_args["experience_years"], 2)


class TestStreamlitApp(unittest.TestCase):
    def test_app_renders_and_searches_role_with_experience(self) -> None:
        from streamlit.testing.v1 import AppTest

        mock_jobs = [
            {
                "title": "Python Developer",
                "company": "CloudTech",
                "location": "Hyderabad",
                "url": "https://example.com/jobs/123",
                "salary": "1,200,000 - 1,800,000",
                "contract_time": "full_time",
                "experience_min_years": 1,
                "experience_max_years": 3,
                "experience_label": "1-3 yrs",
                "description": "Work on MCP agents and APIs.",
            }
        ]

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=10)
        at.run()
        self.assertFalse(at.exception)

        # Experience selector is directly available in AI agent chat too
        at.selectbox(key="agent_experience_input").select(2).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.selectbox(key="agent_experience_input").value, 2)

        mock_search = AsyncMock(return_value=mock_jobs)
        with patch("agent.agent.search_jobs_via_mcp", new=mock_search):
            at.segmented_control(key="active_view").set_value("Search by role").run()
            at.text_input(key="role_query_input").set_value("Python Developer")
            at.text_input(key="role_location_input").set_value("Hyderabad")
            at.selectbox(key="role_experience_input").select(2)
            at.form_submit_button[0].click().run()

        self.assertFalse(at.exception)
        mock_search.assert_awaited_once()
        self.assertEqual(mock_search.await_args.kwargs["experience_years"], 2)
        self.assertEqual(at.session_state["role_jobs"], mock_jobs)
        self.assertEqual(len(at.link_button), 1)
        self.assertEqual(at.link_button[0].url, "https://example.com/jobs/123")

        # Refine client-side by incompatible experience (e.g. 8 years)
        at.selectbox(key="role_results_experience_filter").select(8).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.link_button), 0)
        self.assertTrue(any("No jobs matched" in item.value for item in at.info))


if __name__ == "__main__":
    unittest.main()
