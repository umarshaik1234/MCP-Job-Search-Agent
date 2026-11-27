"""Streamlit UI for the AI Job Search MCP Agent."""

from __future__ import annotations

import asyncio
import html
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agent.agent import (
    DEFAULT_MODEL,
    AgentToolStep,
    analyze_resume,
    run_job_agent,
    search_jobs_for_roles,
    search_jobs_via_mcp,
)
from services.adzuna_service import DEFAULT_COUNTRY, SUPPORTED_COUNTRIES
from utils.job_helpers import filter_jobs
from utils.resume_parser import extract_resume_text

load_dotenv()

st.set_page_config(
    page_title="Job Dekho · AI job search MCP agent",
    page_icon=":material/work:",
    layout="wide",
)

MODEL_OPTIONS = list(
    dict.fromkeys(
        [
            DEFAULT_MODEL,
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]
    )
)

EXPERIENCE_OPTIONS: list[int | None] = [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]

QUICK_PROMPTS = [
    "Python Developer jobs in Hyderabad",
    "AI or ML Engineer roles in Bengaluru",
    "Backend FastAPI jobs in Mumbai",
    "Data Analyst roles in Pune",
]


def format_experience_option(years: int | None) -> str:
    """Formats an experience-years option for selectbox display."""
    if years is None:
        return "Any experience"
    if years == 0:
        return "0 yrs (Fresher)"
    if years == 1:
        return "1 year"
    return f"{years} years"


def init_session_state() -> None:
    """Initializes session state keys in a single predictable place."""
    defaults: dict[str, Any] = {
        "agent_messages": [],
        "agent_jobs": [],
        "role_jobs": None,
        "resume_analysis": None,
        "resume_jobs": None,
        "resume_text": "",
        "resume_filename": "",
        "pending_prompt": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(max_entries=10, show_spinner=False)
def cached_extract_resume(pdf_bytes: bytes) -> str:
    """Caches PDF text extraction across Streamlit reruns."""
    return extract_resume_text(pdf_bytes)


def _inject_scoped_styles() -> None:
    """Applies lightweight, key-scoped styling for hero and card polish."""
    st.html(
        """
        <style>
        .jd-hero {
            padding: 28px 32px;
            border-radius: 20px;
            border: 1px solid rgba(124, 92, 255, 0.32);
            background:
                radial-gradient(circle at 8% 15%, rgba(124, 92, 255, 0.24), transparent 42%),
                radial-gradient(circle at 92% 18%, rgba(56, 189, 248, 0.16), transparent 36%),
                linear-gradient(145deg, rgba(20, 24, 43, 0.96), rgba(12, 15, 29, 0.96));
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.30);
            margin-bottom: 18px;
        }
        .jd-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 5px 12px;
            border-radius: 999px;
            background: rgba(124, 92, 255, 0.16);
            border: 1px solid rgba(124, 92, 255, 0.42);
            color: #d4ccff;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .jd-hero h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(28px, 4vw, 44px);
            line-height: 1.08;
            letter-spacing: -0.03em;
            margin: 14px 0 10px;
            color: #ffffff;
        }
        .jd-hero h1 span {
            background: linear-gradient(90deg, #ffffff 0%, #c4b5fd 48%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .jd-hero p {
            margin: 0;
            max-width: 720px;
            color: #b4bcd0;
            font-size: 15px;
            line-height: 1.6;
        }
        .jd-sidebar-brand {
            padding: 14px 16px;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(124, 92, 255, 0.20), rgba(56, 189, 248, 0.10));
            border: 1px solid rgba(124, 92, 255, 0.30);
            margin-bottom: 12px;
        }
        .jd-sidebar-brand-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #ffffff;
        }
        .jd-sidebar-brand-sub {
            font-size: 12px;
            color: #a5b4fc;
            margin-top: 2px;
        }
        .jd-job-header {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 6px;
        }
        .jd-avatar {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            flex: none;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 14px;
            color: #ffffff;
            background: linear-gradient(135deg, #7c5cff, #0ea5e9);
            box-shadow: 0 6px 18px rgba(124, 92, 255, 0.28);
        }
        .jd-job-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 17px;
            font-weight: 600;
            line-height: 1.3;
            color: #f8fafc;
        }
        .jd-job-company {
            font-size: 13px;
            color: #94a3b8;
            margin-top: 2px;
        }
        .st-key-role_search_form,
        .st-key-resume_search_form {
            background: rgba(18, 21, 38, 0.72);
            border-radius: 16px;
        }
        </style>
        """
    )


def _company_initials(company: str) -> str:
    """Builds a 1-2 character monogram for a company card avatar."""
    words = [part for part in company.strip().split() if part and part[0].isalnum()]
    if not words:
        return "JB"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[1][0]).upper()


def _render_job_metrics(jobs: list[dict[str, Any]]) -> None:
    """Displays a compact KPI summary strip above job search results."""
    total_count = len(jobs)
    companies_count = len(
        {
            str(job.get("company") or "").strip().lower()
            for job in jobs
            if str(job.get("company") or "").strip()
        }
    )
    salary_count = sum(1 for job in jobs if str(job.get("salary") or "").strip())

    with st.grid(3, gap="small"):
        st.metric("Matching roles", f"{total_count}", border=True, height="stretch")
        st.metric("Hiring companies", f"{companies_count}", border=True, height="stretch")
        st.metric("With salary info", f"{salary_count}", border=True, height="stretch")


def render_job_cards(jobs: list[dict[str, Any]], empty_message: str) -> None:
    """Renders normalized job results in a responsive 2-column card grid."""
    if not jobs:
        st.info(empty_message, icon=":material/info:")
        return

    _render_job_metrics(jobs)
    st.space("small")

    with st.grid(2, gap="medium"):
        for index, job in enumerate(jobs, start=1):
            title = str(job.get("title") or "Untitled role")
            company = str(job.get("company") or "Unknown company")
            location = str(job.get("location") or "Unknown location")
            salary = str(job.get("salary") or "")
            contract_time = str(job.get("contract_time") or "").replace("_", " ").title()
            experience_label = str(job.get("experience_label") or "")
            description = str(job.get("description") or "")
            url = str(job.get("url") or "")
            initials = _company_initials(company)

            with st.container(border=True, height="stretch", key=f"job_card_{index}"):
                st.html(
                    f"""
                    <div class="jd-job-header">
                        <div class="jd-avatar">{html.escape(initials)}</div>
                        <div>
                            <div class="jd-job-title">{index}. {html.escape(title)}</div>
                            <div class="jd-job-company">{html.escape(company)}</div>
                        </div>
                    </div>
                    """
                )

                with st.container(horizontal=True):
                    st.badge(location, icon=":material/location_on:", color="blue")
                    if experience_label:
                        st.badge(
                            experience_label,
                            icon=":material/workspace_premium:",
                            color="orange",
                        )
                    if salary:
                        st.badge(salary, icon=":material/payments:", color="green")
                    if contract_time:
                        st.badge(contract_time, icon=":material/schedule:", color="violet")

                if description:
                    st.caption(description)

                if url:
                    st.link_button(
                        "Apply now",
                        url,
                        icon=":material/open_in_new:",
                        width="stretch",
                    )


def _render_tool_step(step: AgentToolStep) -> None:
    """Displays a single MCP tool call inside a chat thinking block."""
    args_summary = ", ".join(f"{k}={v!r}" for k, v in step.arguments.items() if k != "jobs")
    label = f"{step.tool_name}({args_summary})"
    with st.status(label, state="error" if step.is_error else "complete", type="step"):
        if step.is_error:
            st.error(step.error_message)
        else:
            st.caption(f"Returned {step.result_count} job record(s)")


def _execute_agent_turn(
    prompt: str,
    model_name: str,
    experience_years: int | None,
    strict_experience: bool = False,
) -> None:
    """Runs one conversational agent turn and appends results to chat history."""
    st.session_state["agent_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status(":shimmer[Running MCP tools]", type="compact") as status:
            try:
                agent_result = asyncio.run(
                    run_job_agent(
                        prompt,
                        model_name=model_name,
                        experience_years=experience_years,
                        strict_experience=strict_experience,
                    )
                )
                for step in agent_result.steps:
                    _render_tool_step(step)
                status.update(
                    label=f"Completed {len(agent_result.steps)} MCP tool step(s)",
                    state="complete",
                )
            except Exception as exc:
                status.update(label="Agent run failed", state="error")
                st.error(f"Agent execution failed: {exc}", icon=":material/error:")
                agent_result = None

        if agent_result is not None:
            st.markdown(agent_result.answer)
            st.session_state["agent_messages"].append(
                {
                    "role": "assistant",
                    "content": agent_result.answer,
                    "steps": agent_result.steps,
                }
            )
            st.session_state["agent_jobs"] = agent_result.jobs


init_session_state()
_inject_scoped_styles()

has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip())
has_adzuna = bool(
    os.getenv("ADZUNA_APP_ID", "").strip() and os.getenv("ADZUNA_APP_KEY", "").strip()
)

with st.sidebar:
    st.html(
        """
        <div class="jd-sidebar-brand">
            <div class="jd-sidebar-brand-title">JOB DEKHO</div>
            <div class="jd-sidebar-brand-sub">Gemini + MCP career agent</div>
        </div>
        """
    )

    with st.container(border=True):
        st.caption("Search preferences")
        selected_model = st.selectbox(
            "Gemini model",
            options=MODEL_OPTIONS,
            key="selected_model",
        )
        country_code = st.selectbox(
            "Job market",
            options=list(SUPPORTED_COUNTRIES.keys()),
            index=list(SUPPORTED_COUNTRIES.keys()).index(DEFAULT_COUNTRY),
            format_func=lambda code: f"{SUPPORTED_COUNTRIES[code]} ({code.upper()})",
            key="country_code",
        )
        sidebar_experience = st.selectbox(
            "Default experience",
            options=EXPERIENCE_OPTIONS,
            format_func=format_experience_option,
            key="sidebar_experience",
            help="Used by AI chat and as the default filter in role and resume searches.",
        )
        strict_experience = st.checkbox(
            "Only show jobs mentioning experience",
            value=False,
            key="strict_experience",
            help="Hide listings that omit required years of experience when filtering.",
        )
        results_per_page = st.slider(
            "Results per query",
            min_value=5,
            max_value=25,
            value=10,
            step=5,
            key="results_per_page",
        )

    with st.container(border=True):
        st.caption("Resume upload")
        uploaded_file = st.file_uploader(
            "Upload PDF resume",
            type=["pdf"],
            key="resume_uploader",
            help="Used in the Search with resume mode.",
        )

        if uploaded_file is not None:
            try:
                pdf_bytes = uploaded_file.getvalue()
                extracted = cached_extract_resume(pdf_bytes)
                if extracted:
                    st.session_state["resume_text"] = extracted
                    st.session_state["resume_filename"] = uploaded_file.name
                    st.success(
                        f"Loaded {uploaded_file.name} ({len(extracted):,} chars)",
                        icon=":material/check_circle:",
                    )
                else:
                    st.session_state["resume_text"] = ""
                    st.error(
                        "No selectable text was found in this PDF.",
                        icon=":material/error:",
                    )
            except Exception as exc:
                st.session_state["resume_text"] = ""
                st.error(f"Could not read PDF: {exc}", icon=":material/error:")
        else:
            st.session_state["resume_text"] = ""
            st.session_state["resume_filename"] = ""

    st.caption("Service status")
    with st.container(horizontal=True):
        st.badge(
            "Gemini ready" if has_gemini else "GEMINI_API_KEY missing",
            icon=":material/smart_toy:",
            color="green" if has_gemini else "red",
        )
        st.badge(
            "Adzuna ready" if has_adzuna else "Adzuna keys missing",
            icon=":material/work:",
            color="green" if has_adzuna else "red",
        )

st.html(
    """
    <div class="jd-hero">
        <div class="jd-pill">✦ MCP-powered AI career search</div>
        <h1>Find your next <span>opportunity.</span></h1>
        <p>
            Chat with a Gemini agent wired to live Model Context Protocol tools, run role and
            experience searches, or upload your PDF resume for automated skill matching.
        </p>
    </div>
    """
)

active_view = st.segmented_control(
    "Search mode",
    options=[
        "AI agent chat",
        "Search by role",
        "Search with resume",
    ],
    default="AI agent chat",
    key="active_view",
    label_visibility="collapsed",
)

if active_view == "AI agent chat":
    if not has_gemini or not has_adzuna:
        st.warning(
            "Set GEMINI_API_KEY, ADZUNA_APP_ID, and ADZUNA_APP_KEY in your .env file to run the AI agent.",
            title="Configuration required",
            icon=":material/warning:",
        )

    with st.container(border=True):
        exp_col1, exp_col2 = st.columns([1.4, 1], vertical_alignment="bottom")
        with exp_col1:
            agent_experience = st.selectbox(
                "Years of experience",
                options=EXPERIENCE_OPTIONS,
                index=EXPERIENCE_OPTIONS.index(sidebar_experience),
                format_func=format_experience_option,
                key="agent_experience_input",
                help="Applied automatically when you search in chat.",
            )
        with exp_col2:
            agent_strict_exp = st.toggle(
                "Only show jobs mentioning experience",
                value=strict_experience,
                key="agent_strict_experience_toggle",
            )

    if not st.session_state["agent_messages"]:
        with st.container(border=True):
            st.caption("Try a starter prompt")
            with st.container(horizontal=True):
                for idx, starter in enumerate(QUICK_PROMPTS):
                    if st.button(
                        starter,
                        key=f"starter_prompt_{idx}",
                        icon=":material/bolt:",
                        disabled=not (has_gemini and has_adzuna),
                    ):
                        st.session_state["pending_prompt"] = starter

    for message in st.session_state["agent_messages"]:
        with st.chat_message(message["role"]):
            if message.get("steps"):
                with st.status(
                    f"Used {len(message['steps'])} MCP tool step(s)",
                    state="complete",
                    type="compact",
                ):
                    for step in message["steps"]:
                        _render_tool_step(step)
            st.markdown(message["content"])

    queued_prompt = st.session_state.pop("pending_prompt", None)
    typed_prompt = st.chat_input(
        "Ask e.g. 'Find Python Developer jobs in Hyderabad'",
        submit_mode="disable",
        key="agent_chat_input",
        disabled=not (has_gemini and has_adzuna),
    )
    active_prompt = queued_prompt or typed_prompt
    if active_prompt:
        _execute_agent_turn(
            active_prompt,
            selected_model,
            agent_experience,
            agent_strict_exp,
        )

    if st.session_state["agent_jobs"]:
        st.subheader("Matched listings", icon=":material/work:")
        render_job_cards(
            st.session_state["agent_jobs"],
            empty_message="No job listings were returned for this conversation.",
        )

elif active_view == "Search by role":
    with st.form("role_search_form", border=True):
        col_role, col_loc, col_exp = st.columns([1.3, 0.9, 0.8], vertical_alignment="bottom")
        with col_role:
            role_input = st.text_input(
                "Job role or keywords",
                placeholder="Python Developer, Data Scientist, AI Engineer",
                key="role_query_input",
            )
        with col_loc:
            location_input = st.text_input(
                "Location",
                value="Hyderabad",
                placeholder="Hyderabad, Bengaluru, Remote",
                key="role_location_input",
            )
        with col_exp:
            role_experience = st.selectbox(
                "Years of experience",
                options=EXPERIENCE_OPTIONS,
                index=EXPERIENCE_OPTIONS.index(sidebar_experience),
                format_func=format_experience_option,
                key="role_experience_input",
            )

        submitted_role = st.form_submit_button(
            "Search jobs via MCP",
            type="primary",
            icon=":material/search:",
            width="stretch",
        )

    if submitted_role:
        if not role_input.strip() or not location_input.strip():
            st.warning(
                "Enter both a job role and a location before searching.",
                icon=":material/warning:",
            )
        elif not has_adzuna:
            st.error(
                "ADZUNA_APP_ID and ADZUNA_APP_KEY are missing from your .env file.",
                icon=":material/error:",
            )
        else:
            with st.status("Querying MCP job search server", expanded=False) as status:
                try:
                    jobs = asyncio.run(
                        search_jobs_via_mcp(
                            keyword=role_input.strip(),
                            location=location_input.strip(),
                            country=country_code,
                            results_per_page=results_per_page,
                            experience_years=role_experience,
                            strict_experience=strict_experience,
                        )
                    )
                    st.session_state["role_jobs"] = jobs
                    status.update(
                        label=f"Found {len(jobs)} job listing(s)",
                        state="complete",
                    )
                except Exception as exc:
                    status.update(label="Search failed", state="error")
                    st.error(f"Job search failed: {exc}", icon=":material/error:")

    if st.session_state["role_jobs"] is not None:
        filter_col1, filter_col2 = st.columns([1.5, 1], vertical_alignment="bottom")
        with filter_col1:
            role_filter = st.text_input(
                "Filter results",
                placeholder="Filter by title, company, or description keyword",
                key="role_results_filter",
            )
        with filter_col2:
            client_exp_filter = st.selectbox(
                "Refine by years of experience",
                options=EXPERIENCE_OPTIONS,
                format_func=format_experience_option,
                key="role_results_experience_filter",
            )

        filtered_role_jobs = filter_jobs(
            st.session_state["role_jobs"],
            keyword=role_filter,
            experience_years=client_exp_filter,
            strict_experience=strict_experience,
        )
        render_job_cards(
            filtered_role_jobs,
            empty_message="No jobs matched your search or experience filter.",
        )

elif active_view == "Search with resume":
    resume_text = st.session_state["resume_text"]
    if not resume_text:
        st.info(
            "Upload a PDF resume in the sidebar so Gemini can extract your skills and search matching jobs through MCP.",
            title="Resume required",
            icon=":material/upload_file:",
        )
    else:
        with st.form("resume_search_form", border=True):
            res_col1, res_col2 = st.columns([1.3, 1], vertical_alignment="bottom")
            with res_col1:
                resume_location = st.text_input(
                    "Preferred search location",
                    value="Hyderabad",
                    key="resume_location_input",
                )
            with res_col2:
                resume_exp_override = st.selectbox(
                    "Years of experience",
                    options=EXPERIENCE_OPTIONS,
                    index=EXPERIENCE_OPTIONS.index(sidebar_experience),
                    format_func=lambda y: (
                        "Auto-detect from resume"
                        if y is None
                        else format_experience_option(y)
                    ),
                    key="resume_experience_input",
                )
            submitted_resume = st.form_submit_button(
                "Analyze resume and find jobs",
                type="primary",
                icon=":material/auto_awesome:",
                width="stretch",
            )

        if submitted_resume:
            if not has_gemini or not has_adzuna:
                st.error(
                    "GEMINI_API_KEY, ADZUNA_APP_ID, and ADZUNA_APP_KEY are required for resume matching.",
                    icon=":material/error:",
                )
            elif not resume_location.strip():
                st.warning("Enter a preferred search location.", icon=":material/warning:")
            else:
                with st.status("Analyzing resume and searching MCP jobs", expanded=True) as status:
                    try:
                        analysis = analyze_resume(resume_text, model_name=selected_model)
                        st.session_state["resume_analysis"] = analysis
                        job_titles = analysis.get("job_titles", [])
                        target_years = (
                            resume_exp_override
                            if resume_exp_override is not None
                            else analysis.get("experience_years")
                        )

                        if not job_titles:
                            status.update(
                                label="No suitable roles detected",
                                state="error",
                            )
                            st.warning("Gemini could not identify target job titles.")
                        else:
                            status.update(
                                label=f"Searching MCP for {len(job_titles)} role(s)",
                                state="running",
                            )
                            jobs = asyncio.run(
                                search_jobs_for_roles(
                                    job_titles=job_titles,
                                    location=resume_location.strip(),
                                    country=country_code,
                                    results_per_page=results_per_page,
                                    experience_years=target_years,
                                    strict_experience=strict_experience,
                                )
                            )
                            st.session_state["resume_jobs"] = jobs
                            status.update(
                                label=f"Found {len(jobs)} unique job listing(s)",
                                state="complete",
                            )
                    except Exception as exc:
                        status.update(label="Resume search failed", state="error")
                        st.error(f"Resume search failed: {exc}", icon=":material/error:")

        analysis = st.session_state["resume_analysis"]
        if analysis:
            with st.container(border=True):
                st.subheader("Resume profile", icon=":material/person_search:")
                detected_years = analysis.get("experience_years")
                years_text = (
                    format_experience_option(detected_years)
                    if detected_years is not None
                    else "Not specified"
                )
                st.caption(
                    f"Experience level: {analysis.get('experience_level', 'Not specified')} "
                    f"({years_text}) · Detected location: {analysis.get('location', 'N/A')}"
                )

                roles = analysis.get("job_titles", [])
                if roles:
                    st.markdown("**Target roles**")
                    with st.container(horizontal=True):
                        for role in roles:
                            st.badge(role, color="violet")

                skills = analysis.get("skills", []) + analysis.get("technologies", [])
                if skills:
                    st.markdown("**Extracted skills and technologies**")
                    with st.container(horizontal=True):
                        for item in dict.fromkeys(skills):
                            st.badge(item, color="gray")

        if st.session_state["resume_jobs"] is not None:
            resume_filter = st.text_input(
                "Filter resume matches",
                placeholder="Filter by title, company, or keyword",
                key="resume_results_filter",
            )
            filtered_resume_jobs = filter_jobs(
                st.session_state["resume_jobs"],
                keyword=resume_filter,
            )
            render_job_cards(
                filtered_resume_jobs,
                empty_message="No matching jobs found for the detected resume roles.",
            )
