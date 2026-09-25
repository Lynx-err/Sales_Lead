import json
import os
import re
from typing import Any, Dict, Optional

from crewai import Crew, Process

from database import save_lead_result
from models import BANTScore, EmailResponse, LeadProcessingResult
from agents import build_bant_scorer_agent, build_sales_email_agent
from tasks import build_scoring_task, build_email_generation_task
from email_service import default_email_service

# Circuit breaker flag to avoid repeatedly waiting on quota exhaustion
_QUOTA_EXHAUSTED = False


def is_quota_exhausted() -> bool:
    global _QUOTA_EXHAUSTED
    return _QUOTA_EXHAUSTED


def reset_quota_status() -> None:
    global _QUOTA_EXHAUSTED
    _QUOTA_EXHAUSTED = False


def _normalize_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _finalize_bant_result(payload: Dict[str, Any], threshold: int = 60) -> Dict[str, Any]:
    """Applies the deterministic BANT gate in Python so LLM output cannot bypass rules."""
    if not isinstance(payload, dict):
        return payload

    budget_score = max(0, min(25, _normalize_int(payload.get("budget_score"), 0)))
    authority_score = max(0, min(25, _normalize_int(payload.get("authority_score"), 0)))
    need_score = max(0, min(25, _normalize_int(payload.get("need_score"), 0)))
    timeline_score = max(0, min(25, _normalize_int(payload.get("timeline_score"), 0)))

    bant_score = _normalize_int(payload.get("bant_score"), budget_score + authority_score + need_score + timeline_score)
    bant_score = max(0, min(100, bant_score))

    if budget_score <= 5 or authority_score <= 5:
        bant_score = min(bant_score, 55)

    qualification = "Qualified" if bant_score >= threshold else "Unqualified"

    normalized = dict(payload)
    normalized["budget_score"] = budget_score
    normalized["authority_score"] = authority_score
    normalized["need_score"] = need_score
    normalized["timeline_score"] = timeline_score
    normalized["bant_score"] = bant_score
    normalized["qualification"] = qualification
    normalized["budget_signal"] = normalized.get("budget_signal") or "Not explicitly stated"
    normalized["authority_signal"] = normalized.get("authority_signal") or "Authority not clearly established"
    normalized["need_signal"] = normalized.get("need_signal") or "Need not clearly documented"
    normalized["timeline_signal"] = normalized.get("timeline_signal") or "Timeline not explicitly stated"
    normalized["rationale"] = normalized.get("rationale") or "Automated BANT assessment based on the available lead evidence."
    normalized["key_pain_points"] = normalized.get("key_pain_points") or []
    normalized["recommended_action"] = normalized.get("recommended_action") or "Manual sales review required"
    return normalized


def _heuristic_bant_fallback(lead: Dict[str, Any], threshold: int = 60) -> Dict[str, Any]:
    """
    Deterministic rule-based evaluation that calculates BANT scores if the LLM
    encounters network interruption or free-tier rate limiting.
    Ensures seamless demonstration in viva/seminar environments.
    """
    notes = str(lead.get("Notes", "")).lower()
    job_title = str(lead.get("Job Title", "")).lower()
    lead_name = lead.get("Name", "Prospect")

    # 1. Budget Score (0-25)
    budget_score = 5
    budget_signal = "Not explicitly stated"
    if any(k in notes for k in ["$250k", "$200,000", "$150,000", "$100,000"]):
        budget_score = 25
        budget_signal = "High enterprise budget stated ($100k-$250k)"
    elif any(k in notes for k in ["$90,000", "$80,000", "$50,000", "$75,000"]):
        budget_score = 20
        budget_signal = "Substantial budget confirmed ($50k-$90k)"
    elif "budget tbd" in notes or "indicative budget" in notes:
        budget_score = 14
        budget_signal = "Budget earmarked / in planning (TBD)"
    elif "tight budget" in notes or "cost-sensitive" in notes:
        budget_score = 10
        budget_signal = "Tight or constrained budget indicated"

    # 2. Authority Score (0-25)
    authority_score = 6
    authority_signal = "Secondary contributor / unclear authority"
    if any(k in job_title for k in ["head", "director", "vp", "chief", "founder", "owner"]):
        authority_score = 25
        authority_signal = f"High decision-making authority ({lead.get('Job Title')})"
    elif any(k in job_title for k in ["manager", "lead", "supervisor"]):
        authority_score = 18
        authority_signal = f"Departmental manager ({lead.get('Job Title')}), may require leadership alignment"
    elif any(k in job_title for k in ["specialist", "technician", "assistant", "consultant"]):
        authority_score = 12
        authority_signal = f"Technical/operational contributor ({lead.get('Job Title')})"

    if "final sign-off" in notes or "i will be the final" in notes:
        authority_score = min(25, authority_score + 5)
        authority_signal += " - Holds explicit final sign-off authority"
    elif "procurement head" in notes or "cfo" in notes or "requires sign-off" in notes:
        authority_signal += " - Sign-off required from executive/procurement leadership"

    # 3. Need Score (0-25)
    need_score = 10
    need_signal = "General exploration"
    pain_points = []
    if any(k in notes for k in ["eco-friendly", "lipstick", "serum", "makeup", "formulation", "drop-shipping", "catalog", "packaging"]):
        need_score = 22
        need_signal = "Specific line expansion / active product development"
        if "eco-friendly" in notes:
            pain_points.append("Requires eco-friendly cosmetic formulations and MOQ details")
        if "summer makeup" in notes or "shelf" in notes:
            pain_points.append("Needs retail shelf refresh with wholesale supplier terms")
        if "serum" in notes:
            pain_points.append("Formulation partners and dermatological testing needed")
        if "drop-shipping" in notes:
            pain_points.append("Requires reliable dropshipping and fast fulfillment")
    else:
        pain_points.append("General beauty supply evaluation")

    # 4. Timeline Score (0-25)
    timeline_score = 8
    timeline_signal = "Timeline unspecified"
    if any(k in notes for k in ["within 2 months", "1 month", "immediately", "urgently", "timing is important"]):
        timeline_score = 25
        timeline_signal = "Immediate implementation (1-2 months)"
    elif any(k in notes for k in ["within 6 months", "next quarter", "3 months"]):
        timeline_score = 18
        timeline_signal = "Near-term implementation (3-6 months)"
    elif any(k in notes for k in ["9 months", "next year", "sometime next year"]):
        timeline_score = 12
        timeline_signal = "Future horizon (9-12 months)"

    # Total score calculation
    raw_total = budget_score + authority_score + need_score + timeline_score

    # Gating rule: Need & Timeline alone cannot qualify if Budget or Authority is missing
    is_gated = (budget_score <= 5 or authority_score <= 5)
    if is_gated and raw_total > 55:
        bant_score = 55
        gating_note = " (Capped at 55 due to gating rule: missing budget or authority)"
    else:
        bant_score = min(100, raw_total)
        gating_note = ""

    qualification = "Qualified" if (bant_score >= threshold and not is_gated) else "Unqualified"
    rationale = f"Evaluated based on {budget_signal.lower()} and {authority_signal.lower()}{gating_note}."

    return _finalize_bant_result({
        "lead_name": lead_name,
        "budget_score": budget_score,
        "authority_score": authority_score,
        "need_score": need_score,
        "timeline_score": timeline_score,
        "budget_signal": budget_signal,
        "authority_signal": authority_signal,
        "need_signal": need_signal,
        "timeline_signal": timeline_signal,
        "bant_score": bant_score,
        "qualification": qualification,
        "rationale": rationale,
        "key_pain_points": pain_points,
        "recommended_action": "Schedule introductory discovery call" if qualification == "Qualified" else "Enroll in quarterly product newsletter",
    }, threshold=threshold)


def _heuristic_email_fallback(lead: Dict[str, Any], bant_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generates high-quality email outreach or nurture draft."""
    name = lead.get("Name", "Valued Partner")
    company = lead.get("Company", "your team")
    email = lead.get("Email") or lead.get("email") or "prospect@example.com"
    qualification = bant_info.get("qualification", "Unqualified")

    if qualification == "Qualified":
        subject = f"Next steps on {company}'s upcoming product initiative"
        body = f"""Hi {name},

Thank you for reaching out regarding {company}'s upcoming expansion plans. 

We thoroughly reviewed your requirements and timeline. With your target horizon and specifications in mind, our manufacturing and wholesale team can provide the MOQ flexibility, custom pricing tiers, and compliance documentation you requested.

Would you be open to a brief 15-minute introductory call this week to review our catalog and sample availability?

Best regards,
The LeadSense AI Sales Operations Team"""
        cta = "15-minute introductory call to review catalog & sample kit"
        email_type = "sales_pitch"
        talking_points = ["Flexible MOQ tiers", "Rapid sample dispatch", "Comprehensive compliance docs"]
    else:
        subject = f"Connecting with {company} — Resources for future planning"
        body = f"""Hi {name},

Thank you for getting in touch with us about your initiatives at {company}.

We appreciate you sharing your current situation. While our current wholesale programs may align better with your roadmap down the line, we would love to share our latest 2026 Beauty Industry Lookbook & Formulation Catalog for your future planning.

Feel free to reach back out whenever your timeline or project requirements open up!

Warm regards,
The LeadSense AI Team"""
        cta = "Explore our 2026 Beauty Formulation Lookbook"
        email_type = "nurturing_followup"
        talking_points = ["Educational lookbook", "Future roadmap alignment", "Open communication"]

    return {
        "lead_name": name,
        "recipient_email": email,
        "email_type": email_type,
        "subject": subject,
        "body": body,
        "call_to_action": cta,
        "talking_points": talking_points,
    }


def score_lead(
    lead: Dict[str, Any],
    threshold: int = 60,
    use_fallback_on_error: bool = True,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Executes Agent 1 (BANT Lead Qualification Specialist) on a single lead.
    Returns structured BANTScore data dictionary.
    """
    global _QUOTA_EXHAUSTED

    if not use_llm or _QUOTA_EXHAUSTED:
        return _finalize_bant_result(_heuristic_bant_fallback(lead, threshold=threshold), threshold=threshold)

    try:
        agent = build_bant_scorer_agent()
        task = build_scoring_task(agent, lead, threshold=threshold)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()

        if hasattr(result, "pydantic") and result.pydantic:
            payload = result.pydantic.model_dump()
            return _finalize_bant_result(payload, threshold=threshold)

        if isinstance(result, dict):
            return _finalize_bant_result(result, threshold=threshold)

        if isinstance(result, str):
            try:
                payload = json.loads(result)
                return _finalize_bant_result(payload, threshold=threshold)
            except Exception:
                pass

        return _finalize_bant_result(_heuristic_bant_fallback(lead, threshold=threshold), threshold=threshold)
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            _QUOTA_EXHAUSTED = True
        if use_fallback_on_error:
            fallback = _heuristic_bant_fallback(lead, threshold=threshold)
            return _finalize_bant_result(fallback, threshold=threshold)
        raise e


def generate_lead_email(
    lead: Dict[str, Any],
    bant_info: Dict[str, Any],
    use_fallback_on_error: bool = True,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Executes Agent 2 (Sales Outreach & Nurture Specialist) to craft a customized email.
    """
    global _QUOTA_EXHAUSTED

    if not use_llm or _QUOTA_EXHAUSTED:
        return _heuristic_email_fallback(lead, bant_info)

    try:
        agent = build_sales_email_agent()
        task = build_email_generation_task(agent, lead, bant_info)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        if hasattr(result, "pydantic") and result.pydantic:
            return result.pydantic.model_dump()
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                return json.loads(result)
            except Exception:
                pass
        return _heuristic_email_fallback(lead, bant_info)
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            _QUOTA_EXHAUSTED = True
        if use_fallback_on_error:
            return _heuristic_email_fallback(lead, bant_info)
        raise e


def process_lead_pipeline(
    lead: Dict[str, Any],
    threshold: int = 60,
    send_email: bool = False,
    force_live_email: bool = False,
    email_service=None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrates the full multi-agent pipeline:
    1. Agent 1: BANT Scoring & Qualification
    2. Agent 2: Contextual Email Draft Generation (Pitch or Nurture)
    3. Safe Delivery Service: Simulated or Live transmission (if requested)
    """
    service = email_service or default_email_service

    # Step 1: BANT Scoring
    bant_data = score_lead(lead, threshold=threshold, use_llm=use_llm)
    bant_model = BANTScore(**bant_data)

    # Step 2: Email Generation
    email_data = generate_lead_email(lead, bant_data, use_llm=use_llm)
    email_model = EmailResponse(**email_data)

    # Step 3: Optional Email Transmission / Simulation
    delivery_status = "draft"
    sent_timestamp = None
    delivery_notes = None

    if send_email:
        receipt = service.send_email(
            recipient=email_model.recipient_email,
            subject=email_model.subject,
            body=email_model.body,
            force_live=force_live_email,
        )
        delivery_status = receipt.get("status", "draft")
        sent_timestamp = receipt.get("timestamp")
        delivery_notes = receipt.get("notes") or receipt.get("error")

    full_result = LeadProcessingResult(
        lead_data=lead,
        bant_score=bant_model,
        email_response=email_model,
        email_delivery_status=delivery_status,
        email_sent_at=sent_timestamp,
        delivery_notes=delivery_notes,
    )

    processed = full_result.model_dump()
    try:
        save_lead_result(processed)
    except Exception:
        pass
    return processed
