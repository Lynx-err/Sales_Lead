import json
import os
import re
from typing import Any, Dict, Optional

from crewai import Crew, Process

from database import get_bant_history, get_lead_history, save_lead_result
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


def _safe_lower(value: Any) -> str:
    return str(value or "").lower()


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


def classify_email(lead: Dict[str, Any]) -> Dict[str, Any]:
    """Classify an incoming message as SPAM, NON_SALES, or GENUINE_LEAD."""
    text = " ".join(
        [
            str(lead.get("Notes") or lead.get("notes") or ""),
            str(lead.get("Subject") or lead.get("subject") or ""),
            str(lead.get("Name") or lead.get("name") or ""),
            str(lead.get("Company") or lead.get("company") or ""),
            str(lead.get("Job Title") or lead.get("job_title") or ""),
        ]
    ).lower()

    spam_patterns = [
        "free iphone", "claim your reward", "click here", "winner", "lottery", "limited time",
        "you have won", "cash prize", "urgent payout", "viagra", "work from home", "click the link"
    ]
    non_sales_patterns = [
        "password reset", "reset my password", "login issue", "cannot access account", "forgot password",
        "support ticket", "account access", "billing issue", "troubleshooting", "please help", "technical support"
    ]
    sales_patterns = [
        "budget", "pricing", "demo", "purchase", "vendor", "crm", "solution", "procurement",
        "quotation", "software", "sales", "demo request", "need a", "we are evaluating"
    ]

    if any(pattern in text for pattern in spam_patterns):
        return {"classification": "SPAM", "confidence": 0.97, "reason": "The message uses classic spam language and offers a reward or click-through lure."}

    if any(pattern in text for pattern in non_sales_patterns):
        return {"classification": "NON_SALES", "confidence": 0.92, "reason": "The message is operational or account-support oriented rather than a sales inquiry."}

    if any(pattern in text for pattern in sales_patterns):
        return {"classification": "GENUINE_LEAD", "confidence": 0.86, "reason": "The message shows commercial intent and a likely buying or evaluation requirement."}

    return {"classification": "GENUINE_LEAD", "confidence": 0.68, "reason": "No clear spam or non-sales signals were detected; the message is treated as a possible sales lead."}


def detect_intent(text: str) -> Dict[str, Any]:
    """Identify buying intent for genuine sales leads."""
    lower = _safe_lower(text)
    intent_map = {
        "PURCHASE_INTENT": ["upcoming purchase", "purchase", "buying", "buy now", "purchasing", "vendor evaluation"],
        "PRICING_INQUIRY": ["pricing", "price", "quote", "quotation", "budget range", "cost"],
        "DEMO_REQUEST": ["demo", "product demo", "trial", "walkthrough", "see the product"],
        "PRODUCT_INQUIRY": ["need a solution", "product", "features", "requirements", "use case", "need help"],
        "INFORMATION_REQUEST": ["more information", "learn more", "tell me more", "details", "can you share"],
        "FOLLOW_UP": ["follow up", "checking in", "follow-up", "reaching out again", "touching base"],
        "GENERAL_INQUIRY": ["interested", "looking into", "exploring", "want to know"],
    }

    for intent, patterns in intent_map.items():
        if any(pattern in lower for pattern in patterns):
            confidence = 0.91 if intent in {"PURCHASE_INTENT", "PRICING_INQUIRY", "DEMO_REQUEST"} else 0.76
            evidence = f"The lead mentions {', '.join(patterns[:2])} in a way consistent with {intent.lower().replace('_', ' ')}."
            return {"intent": intent, "confidence": confidence, "evidence": evidence}

    return {"intent": "UNKNOWN", "confidence": 0.42, "evidence": "No clear buying intent signals were found in the message."}


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Evaluate emotional tone without using it as a qualification signal."""
    lower = _safe_lower(text)
    positive_words = ["excited", "interested", "pleased", "love", "happy", "great", "eager", "looking forward", "would love"]
    negative_words = ["frustrated", "urgent issue", "concerned", "problem", "delayed", "dissatisfied", "not happy", "need quickly"]

    pos_score = sum(1 for word in positive_words if word in lower)
    neg_score = sum(1 for word in negative_words if word in lower)

    if pos_score > neg_score:
        return {"sentiment": "POSITIVE", "confidence": min(0.95, 0.65 + (pos_score * 0.1)), "reason": "The customer expresses enthusiasm or clear interest in moving forward."}
    if neg_score > pos_score:
        return {"sentiment": "NEGATIVE", "confidence": min(0.95, 0.65 + (neg_score * 0.1)), "reason": "The customer shows frustration or urgency that may require a more human touch."}
    return {"sentiment": "NEUTRAL", "confidence": 0.62, "reason": "The message is informational or balanced with no strong positive or negative tone."}


def get_next_best_action(lead: Dict[str, Any], bant_info: Dict[str, Any], intent: Optional[Dict[str, Any]] = None, sentiment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Recommend an action based on the lead's actual status and signals."""
    qualification = (bant_info or {}).get("qualification") or "Unqualified"
    score = int((bant_info or {}).get("bant_score") or 0)
    notes = _safe_lower(" ".join([str(lead.get("Notes") or ""), str(lead.get("notes") or "")]))
    intent_name = (intent or {}).get("intent") or "UNKNOWN"
    sentiment_name = (sentiment or {}).get("sentiment") or "NEUTRAL"

    has_budget = any(token in notes for token in ["budget", "priced", "pricing", "$", "cost", "quote", "quoted"])
    has_timeline = any(token in notes for token in ["within", "timeline", "next quarter", "6 weeks", "2 months", "3 months", "next month", "immediately", "by "])

    if qualification == "Unqualified":
        return {"action": "SEND_QUALIFICATION_NURTURING_FOLLOW_UP", "reason": "The lead is currently below the qualification threshold and should receive a nurture follow-up.", "priority": "MEDIUM"}

    if "budget" not in notes and score < 75:
        return {"action": "ASK_BUDGET_RANGE", "reason": "The lead appears promising but budget details are still missing or unclear.", "priority": "HIGH"}

    if not has_timeline and score < 85:
        return {"action": "ASK_IMPLEMENTATION_TIMELINE", "reason": "The lead has strong potential but the purchase timeline still needs to be clarified.", "priority": "HIGH"}

    if sentiment_name == "NEGATIVE":
        return {"action": "PRIORITIZE_HUMAN_FOLLOW_UP", "reason": "The customer is expressing concern or frustration; a human touch is likely required.", "priority": "HIGH"}

    if intent_name in {"PURCHASE_INTENT", "PRICING_INQUIRY"} or score >= 80:
        return {"action": "SCHEDULE_SALES_CALL", "reason": "Strong purchase intent, confirmed buying signals, and a healthy BANT profile support a sales call.", "priority": "HIGH"}

    if intent_name == "DEMO_REQUEST":
        return {"action": "REQUEST_DEMO", "reason": "The customer is asking for a demonstration and should be guided to the next product walkthrough.", "priority": "MEDIUM"}

    return {"action": "SCHEDULE_DISCOVERY_CALL", "reason": "The lead is qualified and should receive a concise discovery meeting to confirm requirements.", "priority": "MEDIUM"}


def _has_supporting_evidence(context: Dict[str, Any], claim: str) -> bool:
    lead_data = context.get("lead_data") or {}
    notes = " ".join([str(lead_data.get("Notes") or lead_data.get("notes") or ""), str(lead_data.get("Company") or ""), str(lead_data.get("Job Title") or "")])
    evidence_keywords = {
        "30-day free trial": ["free trial", "trial period"],
        "guarantee": ["guarantee", "warranty", "money-back"],
        "discount": ["discount", "off", "promotional pricing"],
        "price": ["budget", "pricing", "quote", "cost", "budget range"],
        "feature claim": ["demo", "features", "capabilities", "integration", "support"],
    }
    patterns = evidence_keywords.get(claim, [claim])
    return any(pattern in notes.lower() for pattern in patterns)


def validate_generated_email(email: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Check that generated email content does not invent unsupported facts."""
    if not isinstance(email, dict):
        return {"status": "FAIL", "is_valid": False, "issues": ["Email payload was malformed"], "evidence": "No email content was available to validate.", "claim": None, "retry_count": 0, "max_retries": 2}

    subject = str(email.get("subject") or email.get("Subject") or "")
    body = str(email.get("body") or email.get("Body") or "")
    content = f"{subject} {body}".lower()

    unsupported_claim_map = {
        "30-day free trial": r"30[- ]day.*free trial|free trial",
        "guarantee": r"guaranteed|100%.*guarantee|money-back guarantee",
        "discount": r"\$\d+ off|\d+% off|discount",
        "price": r"\$\d+(?:k|,\d+)?(?:\s*(?:per|monthly|annually|year))?|starting at \$",
        "feature claim": r"industry-leading|best-in-class|fully automated|unlimited|24/7 support",
    }

    issues = []
    claim_names = []
    for claim, pattern in unsupported_claim_map.items():
        if re.search(pattern, content):
            if not _has_supporting_evidence(context, claim):
                issues.append(f"Claim: {claim}")
                claim_names.append(claim)

    if issues:
        return {
            "status": "FAIL",
            "is_valid": False,
            "issues": issues,
            "evidence": "No supporting information available in the lead record for the detected claim(s).",
            "claim": claim_names[0] if claim_names else None,
            "retry_count": 0,
            "max_retries": 2,
        }

    return {
        "status": "PASS",
        "is_valid": True,
        "issues": [],
        "evidence": "No unsupported claims detected.",
        "claim": None,
        "retry_count": 0,
        "max_retries": 2,
    }


def _heuristic_bant_fallback(lead: Dict[str, Any], threshold: int = 60) -> Dict[str, Any]:
    """Deterministic rule-based evaluation that calculates BANT scores..."""
    notes = str(lead.get("Notes", "")).lower()
    job_title = str(lead.get("Job Title", "")).lower()
    lead_name = lead.get("Name", "Prospect")

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

    raw_total = budget_score + authority_score + need_score + timeline_score
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
    """Executes Agent 1 (BANT Lead Qualification Specialist) on a single lead."""
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
    """Executes Agent 2 (Sales Outreach & Nurture Specialist) to craft a customized email."""
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
    """Orchestrates the full multi-agent pipeline including pre-classification, intent, sentiment, BANT, action, validation, and history."""
    service = email_service or default_email_service

    def persist_with_history(result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            lead_id = save_lead_result(result)
            result["lead_history"] = get_lead_history(lead_id)
            result["bant_history"] = get_bant_history(lead_id)
        except Exception:
            pass
        return result

    classification = classify_email(lead)

    if classification["classification"] in {"SPAM", "NON_SALES"}:
        skip_result = {
            "lead_data": lead,
            "bant_score": BANTScore(
                lead_name=str(lead.get("Name") or lead.get("name") or "Unknown"),
                budget_score=0,
                authority_score=0,
                need_score=0,
                timeline_score=0,
                budget_signal="Not assessed",
                authority_signal="Not assessed",
                need_signal="Not assessed",
                timeline_signal="Not assessed",
                bant_score=0,
                qualification="Unqualified",
                rationale=f"{classification['classification']} lead was filtered out before BANT analysis.",
                key_pain_points=[],
                recommended_action="Manual review only",
            ).model_dump(),
            "classification": classification,
            "intent": {"intent": "UNKNOWN", "confidence": 0.0, "evidence": "BANT was skipped because the record was classified as non-sales or spam."},
            "sentiment": {"sentiment": "NEUTRAL", "confidence": 0.0, "reason": "No sentiment analysis run because the lead was filtered out."},
            "next_best_action": {"action": "NO_ACTION_REQUIRED", "reason": classification["reason"], "priority": "LOW"},
            "email_response": None,
            "email_validation": {"status": "SKIPPED", "is_valid": True, "issues": [], "evidence": "Email generation skipped for non-sales or spam lead.", "claim": None, "retry_count": 0, "max_retries": 2},
            "email_delivery_status": "skipped",
            "email_sent_at": None,
            "delivery_notes": classification["reason"],
            "lead_history": [],
            "bant_history": [],
        }
        return persist_with_history(skip_result)

    notes_text = " ".join([str(lead.get("Notes") or lead.get("notes") or ""), str(lead.get("Company") or ""), str(lead.get("Job Title") or "")])
    intent = detect_intent(notes_text)
    sentiment = analyze_sentiment(notes_text)
    bant_data = score_lead(lead, threshold=threshold, use_llm=use_llm)
    bant_model = BANTScore(**bant_data)
    action = get_next_best_action(lead, bant_data, intent=intent, sentiment=sentiment)

    generated_email = generate_lead_email(lead, bant_data, use_llm=use_llm)
    email_validation = validate_generated_email(generated_email, {"lead_data": lead, "bant_score": bant_data})

    if not email_validation["is_valid"]:
        retry_count = 0
        while not email_validation["is_valid"] and retry_count < 2:
            generated_email = generate_lead_email(lead, bant_data, use_llm=use_llm)
            email_validation = validate_generated_email(generated_email, {"lead_data": lead, "bant_score": bant_data})
            retry_count += 1
            email_validation["retry_count"] = retry_count
        if not email_validation["is_valid"]:
            email_validation["status"] = "FAIL"
            email_validation["evidence"] = "Email requires human review after validation failed."

    email_model = EmailResponse(**generated_email)
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

    result = LeadProcessingResult(
        lead_data=lead,
        bant_score=bant_model,
        classification=classification,
        intent=intent,
        sentiment=sentiment,
        next_best_action=action,
        email_response=email_model,
        email_validation=email_validation,
        email_delivery_status=delivery_status,
        email_sent_at=sent_timestamp,
        delivery_notes=delivery_notes,
    )
    processed = result.model_dump()
    return persist_with_history(processed)
