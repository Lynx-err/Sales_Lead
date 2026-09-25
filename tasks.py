from typing import Dict, Any, Union
from crewai import Agent, Task
from models import BANTScore, EmailResponse


def build_scoring_task(agent: Agent, lead: Dict[str, Any], threshold: int = 60) -> Task:
    """Builds the BANT scoring task for Agent 1."""
    lead_block = "\n".join(f"{key}: {value}" for key, value in lead.items() if value)

    return Task(
        description=f"""
Score the following inbound sales lead using the standard BANT framework
(Budget, Authority, Need, Timeline).

LEAD RECORD:
{lead_block}

QUALIFICATION THRESHOLD: {threshold}/100

Instructions:
1. Category Scoring Breakdown (0 to 25 points each):
   - Budget (0-25): Evaluate indicated budget, spending capacity, or pricing inquiry.
     * 20-25: Explicit budget figure ($50k+, $100k+, enterprise budget).
     * 10-19: Moderate or flexible budget stated ("tight budget", "budget TBD", "pricing requested").
     * 0-9: Zero mention of budget, or unrealistic expectations.
   - Authority (0-25): Evaluate lead's title and purchasing authority.
     * 20-25: Direct decision maker (Owner, Founder, VP, Director, Head of Merchandising).
     * 10-19: Influencer or manager requiring sign-off ("Store Manager", "Specialist with sign-off needed").
     * 0-9: Low-influence role or unknown authority.
   - Need (0-25): Evaluate clarity of requirements, product fit, and pain points.
     * 20-25: Urgent, well-defined operational or product requirement (e.g. launching eco-friendly line, new assortment).
     * 10-19: General exploratory interest or broad inquiry.
     * 0-9: Vague or misaligned need.
   - Timeline (0-25): Evaluate urgency and expected implementation date.
     * 20-25: Immediate to near-term horizon (within 1-3 months).
     * 10-19: Mid-term horizon (3-9 months or next year).
     * 0-9: No timeline mentioned or distant (> 12 months).

2. Total BANT Score:
   - bant_score MUST equal the exact sum of: budget_score + authority_score + need_score + timeline_score.

3. Critical Gating Rule:
   - Budget and Authority are strict gating factors:
   - If Budget is completely unaddressed (budget_score <= 5) OR Authority is completely unaddressed (authority_score <= 5),
     the total bant_score CANNOT exceed 55, regardless of Need and Timeline.
   - If the sum exceeds 55 in such cases, cap bant_score at 55 and adjust category scores proportionally.

4. Qualification Label:
   - If bant_score >= {threshold} (and not capped below threshold by gating), mark as "Qualified".
   - Otherwise, mark as "Unqualified".

5. Signals & Rationale:
   - Extract clear, evidence-based qualitative signals for budget_signal, authority_signal, need_signal, timeline_signal.
   - List 2-3 key pain points or requirements in key_pain_points.
   - Provide a 1-2 sentence rationale explicitly citing evidence from the lead record.
   - Specify recommended_action (e.g., "Schedule technical discovery call", "Send product catalog & MOQ list").
""",
        expected_output="A structured BANTScore Pydantic object containing all category scores and signals.",
        agent=agent,
        output_pydantic=BANTScore,
    )


def build_email_generation_task(
    agent: Agent, lead: Dict[str, Any], bant_info: Union[Dict[str, Any], BANTScore]
) -> Task:
    """Builds the personalized outreach or nurturing email generation task for Agent 2."""
    if isinstance(bant_info, BANTScore):
        bant_data = bant_info.model_dump()
    else:
        bant_data = bant_info

    lead_block = "\n".join(f"{key}: {value}" for key, value in lead.items() if value)
    bant_summary = "\n".join(f"{key}: {value}" for key, value in bant_data.items())

    qualification = bant_data.get("qualification", "Unqualified")
    recipient_email = lead.get("Email") or lead.get("email") or "prospect@example.com"
    lead_name = lead.get("Name") or lead.get("name") or "Valued Partner"

    return Task(
        description=f"""
Draft an authentic, highly personalized B2B sales email for this prospect based on their BANT qualification.

LEAD DETAILS:
{lead_block}

BANT QUALIFICATION DATA:
{bant_summary}

RECIPIENT EMAIL: {recipient_email}
RECIPIENT NAME: {lead_name}
QUALIFICATION STATUS: {qualification}

Instructions based on Qualification Status:

IF QUALIFIED:
1. email_type: "sales_pitch"
2. Subject Line: High-impact, personalized, mentioning their company/initiative (e.g. "Next steps on [Company]'s new lipstick range").
3. Salutation: Professional salutation addressing {lead_name}.
4. Body:
   - Demonstrate clear understanding of their specific requirements (reference their notes, industry, pain points).
   - Acknowledge their timeline and budget parameters respectfully.
   - Position our capabilities directly solving their challenges (e.g. sample availability, custom formulations, rapid turnaround, MOQ flexibility).
   - Keep tone consultative, confident, and professional (avoid pushy buzzwords).
5. Call to Action: Low-friction next step (e.g. 15-minute alignment call, dispatching sample kits, sharing wholesale catalog).
6. Professional Sign-off: "Best regards, \nThe LeadSense AI Sales Operations Team".

IF UNQUALIFIED:
1. email_type: "nurturing_followup"
2. Subject Line: Warm, polite, non-transactional (e.g. "Resources for [Company]'s future planning").
3. Salutation: Warm salutation addressing {lead_name}.
4. Body:
   - Thank them sincerely for reaching out.
   - Gently acknowledge that current parameters (timing/budget) may align better down the line.
   - Provide genuine value (e.g. offering product lookup guide, educational whitepaper, or product lookbook).
   - Reassure them that we remain available whenever their timeline opens up.
5. Call to Action: Gentle offer to stay in touch or bookmark our resources.
6. Professional Sign-off: "Warm regards, \nThe LeadSense AI Team".
""",
        expected_output="A structured EmailResponse Pydantic object with subject, body, and CTA.",
        agent=agent,
        output_pydantic=EmailResponse,
    )