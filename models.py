from pydantic import BaseModel, Field
from typing import Literal


class BANTScore(BaseModel):
    """Structured BANT scoring output for a single lead."""

    lead_name: str = Field(description="Full name of the lead")
    budget_signal: str = Field(
        description="What the notes suggest about budget/spending capacity, "
        "or 'not mentioned' if there is no evidence either way"
    )
    authority_signal: str = Field(
        description="What the notes/job title suggest about the lead's "
        "decision-making authority"
    )
    need_signal: str = Field(
        description="What the notes suggest about the lead's need or pain point"
    )
    timeline_signal: str = Field(
        description="What the notes suggest about urgency/timeline, "
        "or 'not mentioned' if there is no evidence either way"
    )
    bant_score: int = Field(
        ge=0, le=100, description="Overall BANT score from 0-100"
    )
    qualification: Literal["Qualified", "Unqualified"] = Field(
        description="Final qualification label based on bant_score"
    )
    rationale: str = Field(
        description="1-2 sentence explanation citing the specific evidence used"
    )
