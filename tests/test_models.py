import pytest
from pydantic import ValidationError
from models import BANTScore, EmailResponse, LeadProcessingResult


def test_bant_score_valid():
    """Tests creating a valid BANTScore model."""
    score = BANTScore(
        lead_name="Maria Olson",
        budget_score=25,
        authority_score=25,
        need_score=25,
        timeline_score=20,
        budget_signal="$250k budget earmarked",
        authority_signal="Head of Merchandising with final sign-off",
        need_signal="Refreshing summer makeup shelf",
        timeline_signal="Within 2 months",
        bant_score=95,
        qualification="Qualified",
        rationale="Strong signals across all four BANT dimensions.",
        key_pain_points=["Requires catalog and wholesale terms", "Summer shelf refresh"],
        recommended_action="Dispatch wholesale sample kit and schedule call",
    )
    assert score.lead_name == "Maria Olson"
    assert score.bant_score == 95
    assert score.qualification == "Qualified"
    assert score.budget_score == 25
    assert score.authority_score == 25
    assert len(score.key_pain_points) == 2


def test_bant_score_bounds_validation():
    """Tests that BANT score bounds (0-100) and category bounds (0-25) are enforced."""
    with pytest.raises(ValidationError):
        BANTScore(
            lead_name="Test Lead",
            budget_score=30,  # Invalid: > 25
            authority_score=10,
            need_score=10,
            timeline_score=10,
            budget_signal="test",
            authority_signal="test",
            need_signal="test",
            timeline_signal="test",
            bant_score=60,
            qualification="Qualified",
            rationale="test",
        )

    with pytest.raises(ValidationError):
        BANTScore(
            lead_name="Test Lead",
            budget_score=20,
            authority_score=20,
            need_score=20,
            timeline_score=20,
            budget_signal="test",
            authority_signal="test",
            need_signal="test",
            timeline_signal="test",
            bant_score=105,  # Invalid: > 100
            qualification="Qualified",
            rationale="test",
        )


def test_email_response_valid():
    """Tests structured EmailResponse model."""
    email = EmailResponse(
        lead_name="Zachary Ford",
        recipient_email="david52@valenzuela.com",
        email_type="sales_pitch",
        subject="Next steps on Andrade Group Beauty's new lipstick line",
        body="Hi Zachary,\n\nThank you for reaching out...",
        call_to_action="Schedule 15-min discovery call",
        talking_points=["Eco-friendly options", "MOQ flexibility"],
    )
    assert email.email_type == "sales_pitch"
    assert "Andrade" in email.subject
    assert email.recipient_email == "david52@valenzuela.com"


def test_lead_processing_result():
    """Tests unified LeadProcessingResult wrapper."""
    lead = {
        "Name": "Julie Hurst",
        "Company": "Snyder, Allen and Alexander Beauty",
        "Email": "millereric@harmon.com",
    }
    score = BANTScore(
        lead_name="Julie Hurst",
        budget_score=10,
        authority_score=12,
        need_score=18,
        timeline_score=12,
        budget_signal="Tight budget",
        authority_signal="Specialist, compiles for director",
        need_signal="Drop-shipping partner",
        timeline_signal="In 9 months",
        bant_score=52,
        qualification="Unqualified",
        rationale="Budget is tight and purchase timeline is 9 months away.",
    )
    result = LeadProcessingResult(
        lead_data=lead,
        bant_score=score,
        email_delivery_status="simulated",
    )
    assert result.bant_score.qualification == "Unqualified"
    assert result.email_delivery_status == "simulated"
    assert result.lead_data["Company"] == "Snyder, Allen and Alexander Beauty"
