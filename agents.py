import os
from crewai import Agent, LLM


def get_llm() -> LLM:
    """Gemini LLM used by Agent 1. Requires GEMINI_API_KEY in the environment."""
    return LLM(
        model="gemini/gemini-3.6-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2,  # low temperature: consistent, less "creative" scoring
    )


def build_bant_scorer_agent() -> Agent:
    return Agent(
        role="BANT Lead Qualification Specialist",
        goal=(
            "Read a raw sales lead record and extract Budget, Authority, Need, "
            "and Timeline signals from the free-text notes and structured fields, "
            "then assign an objective 0-100 BANT score and a Qualified/Unqualified label."
        ),
        backstory=(
            "You are an experienced B2B sales operations analyst for a beauty "
            "industry SaaS company. You have scored thousands of inbound leads "
            "and are known for being consistent and evidence-based: you never "
            "invent facts that aren't implied by the lead's own words, and you "
            "clearly say when a BANT dimension isn't addressed at all rather "
            "than guessing."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )
