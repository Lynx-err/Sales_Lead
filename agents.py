import os

# Disable telemetry and OTel exporter to prevent network timeouts
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from typing import Optional
from crewai import Agent, LLM
from dotenv import load_dotenv

load_dotenv()


def get_llm(model_name: Optional[str] = None, temperature: float = 0.2) -> LLM:
    """Returns the Gemini LLM instance configured for CrewAI agents."""
    model = model_name or os.getenv("GEMINI_MODEL", "gemini/gemini-2.0-flash")
    api_key = os.getenv("GEMINI_API_KEY")
    return LLM(
        model=model,
        api_key=api_key,
        temperature=temperature,
        timeout=20,
    )


def build_bant_scorer_agent(llm: Optional[LLM] = None) -> Agent:
    """Agent 1: Evaluates raw lead inputs using the BANT framework."""
    active_llm = llm or get_llm(temperature=0.1)
    return Agent(
        role="BANT Lead Qualification Specialist",
        goal=(
            "Analyze raw sales lead records, extract Budget, Authority, Need, "
            "and Timeline signals, assign granular dimension scores (0-25 each, 0-100 total), "
            "and determine Qualified or Unqualified status."
        ),
        backstory=(
            "You are a meticulous B2B sales operations analyst specializing in commercial "
            "lead qualification for SaaS, retail, and beauty brands. You evaluate leads "
            "strictly based on documented evidence across Budget, Authority, Need, and Timeline (BANT). "
            "You never invent missing details, rigorously enforce gating requirements when "
            "budget or authority is missing, and provide clear rationales for your scores."
        ),
        llm=active_llm,
        verbose=False,
        allow_delegation=False,
        max_retry_limit=1,
    )


def build_sales_email_agent(llm: Optional[LLM] = None) -> Agent:
    """Agent 2: Generates customized sales outreach or nurture emails."""
    active_llm = llm or get_llm(temperature=0.3)
    return Agent(
        role="Senior B2B Sales Outreach & Communication Specialist",
        goal=(
            "Craft tailored, high-converting sales pitch emails for Qualified leads, "
            "and polite, value-driven nurturing emails for Unqualified leads based on "
            "their BANT profile and specific requirements."
        ),
        backstory=(
            "You are an elite B2B sales communication strategist. For high-potential "
            "Qualified leads, you draft persuasive, professional emails that directly address "
            "their pain points, acknowledge their budget and timeline, and propose low-friction "
            "next steps (e.g. discovery calls, product samples). For Unqualified leads, you craft "
            "warm, courteous nurturing emails that provide educational resources and keep the door "
            "open for future collaboration without being overly aggressive."
        ),
        llm=active_llm,
        verbose=False,
        allow_delegation=False,
        max_retry_limit=1,
    )
