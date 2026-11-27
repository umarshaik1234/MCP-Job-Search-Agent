"""Standalone MCP client smoke-test script for the job search server."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from agent.agent import get_server_params
from utils.job_helpers import parse_mcp_tool_result


async def main() -> None:
    """Connects to the MCP server, lists tools, and runs a sample job search."""
    async with stdio_client(get_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

            print("Available MCP tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            print("\nSearching for Python Developer jobs (2 years exp) in Hyderabad...")
            result = await session.call_tool(
                "search_jobs_mcp",
                arguments={
                    "keyword": "Python Developer",
                    "location": "Hyderabad",
                    "experience_years": 2,
                },
            )
            if getattr(result, "isError", False):
                print("MCP tool returned an error:", result.content)
                return

            jobs = parse_mcp_tool_result(result)
            print(f"\nFound {len(jobs)} job(s):")
            for index, job in enumerate(jobs, start=1):
                exp = f" [{job['experience_label']}]" if job.get("experience_label") else ""
                print(
                    f"{index}. {job.get('title')} — {job.get('company')} "
                    f"({job.get('location')}){exp}"
                )


if __name__ == "__main__":
    asyncio.run(main())
