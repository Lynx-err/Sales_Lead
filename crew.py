from crewai import Crew, Process

from agents import build_bant_scorer_agent
from tasks import build_scoring_task


def score_lead(lead: dict) -> dict:
    """Runs Agent 1 on a single lead and returns the BANT score as a dict."""
    agent = build_bant_scorer_agent()
    task = build_scoring_task(agent, lead)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return result.pydantic.model_dump()
