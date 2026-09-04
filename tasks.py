from crewai import Agent, Task
from models import BANTScore


def build_scoring_task(agent: Agent, lead: dict) -> Task:
    lead_block = "\n".join(f"{key}: {value}" for key, value in lead.items())

    return Task(
        description=f"""
Score the following inbound sales lead using the BANT framework
(Budget, Authority, Need, Timeline).

LEAD RECORD:
{lead_block}

Instructions:
1. Read the Notes field carefully - it usually contains the lead's own
   description of their situation and is your main source for Budget and
   Timeline signals.
2. Use Job Title as a secondary signal for Authority (e.g. "Head of",
   "Manager", "Director" suggest more authority than "Assistant" or
   "Technician").
3. Use Industry and Company Size as secondary signals for Need and Budget.
4. If a BANT dimension is genuinely not addressed anywhere in the record,
   say so explicitly (e.g. "not mentioned") rather than guessing or
   inventing a number.
5. Weigh all four dimensions roughly equally to produce a single
   bant_score from 0-100, EXCEPT for the gating rule in step 6 below.
6. Budget and Authority are gating factors: a lead cannot score above 55
   unless there is at least SOME concrete evidence of both:
   - Budget: a number, range, or an explicit statement like "budget TBD"
     or "tight budget" counts as evidence. Complete silence on budget
     does not.
   - Authority: any indication of who approves the purchase - the lead
     themselves, a named role, or "requires sign-off from X" - counts as
     evidence. Complete silence on who decides does not.
   A strong Need and a clear Timeline alone should never be enough to
   reach "Qualified" if Budget or Authority is entirely unaddressed -
   cap bant_score at 55 in that case, regardless of how compelling the
   other two dimensions are.
7. Mark qualification as "Qualified" if bant_score >= 60, otherwise
   "Unqualified".
8. Keep the rationale to 1-2 sentences and reference the specific
   evidence you used from the record. If the gating rule in step 6
   capped the score, say so explicitly in the rationale.
""",
        expected_output="A structured BANTScore object matching the output schema.",
        agent=agent,
        output_pydantic=BANTScore,
    )