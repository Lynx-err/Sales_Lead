import os
import csv
import json
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

from models import BANTScore, EmailResponse, LeadProcessingResult
from crew import (
    score_lead,
    generate_lead_email,
    process_lead_pipeline,
    is_quota_exhausted,
    reset_quota_status,
)
from email_service import default_email_service, EmailDeliveryService

# Page Configuration
st.set_page_config(
    page_title="LeadSense AI — BANT Sales Lead Qualification",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b 0%, #0f172a 45%, #020617 100%);
        color: #f8fafc;
    }
    
    .stMetric {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .leadsense-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    
    .badge-qualified {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        letter-spacing: 0.05em;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
    
    .badge-unqualified {
        background: linear-gradient(135deg, #e11d48 0%, #f43f5e 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        letter-spacing: 0.05em;
        box-shadow: 0 0 15px rgba(244, 63, 94, 0.4);
    }
    
    .bant-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    
    .email-container {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 20px;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# Session State Initialization
if "scored_leads" not in st.session_state:
    st.session_state.scored_leads = []
    if os.path.exists("leads_scored.json"):
        try:
            with open("leads_scored.json", "r", encoding="utf-8") as f:
                st.session_state.scored_leads = json.load(f)
        except Exception:
            st.session_state.scored_leads = []

if "raw_leads" not in st.session_state:
    st.session_state.raw_leads = []
    if os.path.exists("leads.csv"):
        try:
            with open("leads.csv", newline="", encoding="utf-8-sig") as f:
                st.session_state.raw_leads = list(csv.DictReader(f))
        except Exception:
            st.session_state.raw_leads = []

if "email_service" not in st.session_state:
    st.session_state.email_service = default_email_service


# Sidebar Controls & Branding
with st.sidebar:
    st.markdown("## ⚡ **LeadSense AI**")
    st.caption("Intelligent BANT Sales Lead Qualification")
    st.divider()

    st.markdown("#### ⚙️ **Engine & Mode**")
    engine_choice = st.radio(
        "Qualification Engine:",
        ["⚡ Fast Demo Engine (Instant)", "🤖 Real Gemini LLM (CrewAI)"],
        index=0,
        help="Fast Demo Engine gives instant results without LLM latency. Gemini LLM invokes Google Gemini 3.6 Flash via CrewAI."
    )
    use_llm = (engine_choice == "🤖 Real Gemini LLM (CrewAI)")

    qualification_threshold = st.slider(
        "BANT Qualification Threshold",
        min_value=40,
        max_value=90,
        value=60,
        step=5,
        help="Leads scoring at or above this cutoff are classified as Qualified."
    )

    st.divider()
    st.markdown("#### 🛡️ **Email Delivery Protection**")
    demo_mode_toggle = st.toggle(
        "Safe Demo Mode",
        value=st.session_state.email_service.is_demo_mode(),
        help="When enabled, email transmissions are simulated with audit logs to prevent accidental outbound emails."
    )
    st.session_state.email_service.set_demo_mode(demo_mode_toggle)

    if demo_mode_toggle:
        st.success("Safe Demo Mode: Active (Mock Delivery)", icon="🛡️")
    else:
        st.warning("Live Delivery: Enabled (GCP/SMTP)", icon="⚠️")

    st.divider()
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        if is_quota_exhausted():
            st.warning("Gemini Quota Limited (Free Tier 20 reqs/day)", icon="⚠️")
            if st.button("Reset Quota Status"):
                reset_quota_status()
                st.rerun()
        else:
            st.caption("🟢 **Gemini AI:** Active (`gemini-3.6-flash`)")
    else:
        st.caption("🔴 **Gemini AI:** Key Not Set (Using Rule Fallback)")

    st.divider()
    st.markdown(
        """
        **Academic Project Details**  
        *Major Project Progress Seminar-II*  
        *Session: 2026-2027*  
        *GHRCE, Nagpur*
        """
    )


# Top Banner
st.markdown("""
<div class="leadsense-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                LeadSense AI
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
                Autonomous BANT Qualification & Multi-Agent Sales Communication Pipeline
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.4); padding: 6px 14px; border-radius: 8px; font-size: 0.85rem; color: #cbd5e1;">
                Powered by Gemini LLM & CrewAI
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Navigation Tabs
tab_overview, tab_single, tab_batch, tab_emails, tab_about = st.tabs([
    "📊 Executive Dashboard",
    "🎯 Single Lead Qualification",
    "📁 Batch Processing & Dataset",
    "✉️ Email Communication Hub",
    "ℹ️ Architecture & Seminar Specs",
])


# ==========================================
# TAB 1: EXECUTIVE DASHBOARD
# ==========================================
with tab_overview:
    scored_leads = st.session_state.scored_leads

    if not scored_leads:
        st.info("No leads scored yet. Use Tab 2 to qualify a single prospect or Tab 3 to run the batch dataset.", icon="💡")
    else:
        total_leads = len(scored_leads)
        qualified_leads = sum(1 for l in scored_leads if l.get("bant_score", {}).get("qualification") == "Qualified")
        unqualified_leads = total_leads - qualified_leads
        qual_rate = (qualified_leads / total_leads * 100) if total_leads > 0 else 0
        avg_score = sum(l.get("bant_score", {}).get("bant_score", 0) for l in scored_leads) / total_leads if total_leads > 0 else 0

        # KPI Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Inbound Leads", total_leads)
        with col2:
            st.metric("Qualified Pipeline", f"{qualified_leads} leads", f"{qual_rate:.1f}% conversion")
        with col3:
            st.metric("Nurture / Follow-up", f"{unqualified_leads} leads")
        with col4:
            st.metric("Avg BANT Score", f"{avg_score:.1f} / 100")

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row
        col_chart1, col_chart2 = st.columns([1, 1])

        with col_chart1:
            st.markdown("##### 🎯 **BANT Framework Dimension Averages**")
            b_avg = sum(l.get("bant_score", {}).get("budget_score", 0) for l in scored_leads) / total_leads
            a_avg = sum(l.get("bant_score", {}).get("authority_score", 0) for l in scored_leads) / total_leads
            n_avg = sum(l.get("bant_score", {}).get("need_score", 0) for l in scored_leads) / total_leads
            t_avg = sum(l.get("bant_score", {}).get("timeline_score", 0) for l in scored_leads) / total_leads

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[b_avg, a_avg, n_avg, t_avg, b_avg],
                theta=['Budget (25)', 'Authority (25)', 'Need (25)', 'Timeline (25)', 'Budget (25)'],
                fill='toself',
                fillcolor='rgba(99, 102, 241, 0.35)',
                line=dict(color='#818cf8', width=2),
                name='Average Score'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 25], gridcolor='rgba(255, 255, 255, 0.1)'),
                    angularaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                    bgcolor='rgba(15, 23, 42, 0.4)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=20, b=20),
                height=320,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_chart2:
            st.markdown("##### 📈 **Lead Score Distribution & Cutoff**")
            scores = [l.get("bant_score", {}).get("bant_score", 0) for l in scored_leads]
            names = [l.get("lead_data", {}).get("Name", "Lead") for l in scored_leads]
            qual_labels = [l.get("bant_score", {}).get("qualification", "Unqualified") for l in scored_leads]

            df_dist = pd.DataFrame({"Lead": names, "BANT_Score": scores, "Status": qual_labels})
            fig_bar = px.bar(
                df_dist,
                x="Lead",
                y="BANT_Score",
                color="Status",
                color_discrete_map={"Qualified": "#10b981", "Unqualified": "#f43f5e"},
                text="BANT_Score",
            )
            fig_bar.add_hline(
                y=qualification_threshold,
                line_dash="dot",
                line_color="#fbbf24",
                annotation_text=f"Threshold ({qualification_threshold})",
                annotation_position="bottom right"
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.4)',
                font=dict(color='#e2e8f0'),
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 105]),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ==========================================
# TAB 2: SINGLE LEAD QUALIFICATION
# ==========================================
with tab_single:
    st.markdown("#### 🎯 **Interactive Real-Time Lead Qualification**")
    st.caption("Agent 1 extracts BANT signals and assigns scores; Agent 2 drafts customized sales outreach or nurture emails.")

    # Preset selection
    sample_leads = st.session_state.raw_leads
    preset_options = ["Custom Input"] + [f"{l.get('Name')} — {l.get('Company')} ({l.get('Job Title')})" for l in sample_leads]
    selected_preset = st.selectbox("⚡ Quick-load from Beauty Industry Dataset:", preset_options)

    default_name = "Maria Olson"
    default_title = "Head of Merchandising"
    default_company = "Gray, Olson and Anderson Beauty"
    default_email = "amandacortez@duncan-foster.org"
    default_phone = "7700803464"
    default_industry = "Cosmetics Retail"
    default_size = "Small (1-50)"
    default_notes = (
        "Hello, My name is Maria Olson from Gray, Olson and Anderson Beauty and I work as the Head of Merchandising. "
        "We are refreshing our summer makeup shelf with a new color assortment and need catalog and wholesale terms. "
        "Our indicative budget for this initiative is $250k. We're targeting to move forward within 2 months and timing is important for us. "
        "Could you please provide scope, sample availability, MOQ, pricing tiers, lead times, and compliance documents? "
        "I will be the final sign-off on supplier selection. Regards, Maria Olson"
    )

    if selected_preset != "Custom Input":
        idx = preset_options.index(selected_preset) - 1
        lead_preset = sample_leads[idx]
        default_name = lead_preset.get("Name", "")
        default_title = lead_preset.get("Job Title", "")
        default_company = lead_preset.get("Company", "")
        default_email = lead_preset.get("Email", "")
        default_phone = lead_preset.get("Phone", "")
        default_industry = lead_preset.get("Industry", "")
        default_size = lead_preset.get("Size", "")
        default_notes = lead_preset.get("Notes", "")

    with st.form("single_lead_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            name_input = st.text_input("Lead Name", value=default_name)
            title_input = st.text_input("Job Title / Designation", value=default_title)
        with col_f2:
            company_input = st.text_input("Company Name", value=default_company)
            industry_input = st.text_input("Industry", value=default_industry)
        with col_f3:
            email_input = st.text_input("Email Address", value=default_email)
            size_input = st.selectbox("Company Size", ["Small (1-50)", "Medium (51-200)", "Large (201-1000)", "Enterprise (1000+)"], index=0)

        notes_input = st.text_area("Inbound Sales Inquiry / Notes (Free-text)", value=default_notes, height=130)
        submit_btn = st.form_submit_button("⚡ Run Multi-Agent LeadSense AI Pipeline", use_container_width=True)

    if submit_btn:
        lead_payload = {
            "Name": name_input,
            "Job Title": title_input,
            "Company": company_input,
            "Email": email_input,
            "Phone": default_phone,
            "Industry": industry_input,
            "Size": size_input,
            "Notes": notes_input,
        }

        with st.spinner("🤖 Agent 1: Evaluating BANT Signals... -> Agent 2: Drafting Email..."):
            result = process_lead_pipeline(
                lead=lead_payload,
                threshold=qualification_threshold,
                send_email=False,
                email_service=st.session_state.email_service,
                use_llm=use_llm,
            )

        # Update Session State
        existing_idx = next((i for i, r in enumerate(st.session_state.scored_leads) if r.get("lead_data", {}).get("Name") == name_input), None)
        if existing_idx is not None:
            st.session_state.scored_leads[existing_idx] = result
        else:
            st.session_state.scored_leads.append(result)

        # Display Results
        score_data = result["bant_score"]
        email_data = result["email_response"]
        is_qual = score_data["qualification"] == "Qualified"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 **Lead Qualification Assessment**")

        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            # Score Overview Card
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 24px; text-align: center;">
                <div style="margin-bottom: 12px;">
                    <span class="{'badge-qualified' if is_qual else 'badge-unqualified'}">
                        {score_data['qualification'].upper()}
                    </span>
                </div>
                <h1 style="font-size: 3.5rem; margin: 0; color: {'#10b981' if is_qual else '#f43f5e'};">
                    {score_data['bant_score']}<span style="font-size: 1.5rem; color: #94a3b8;">/100</span>
                </h1>
                <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">
                    Qualification Cutoff: {qualification_threshold}/100
                </p>
                <div style="text-align: left; margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                    <p style="margin: 0; font-size: 0.9rem; color: #e2e8f0;"><strong>Rationale:</strong> {score_data['rationale']}</p>
                    <p style="margin: 8px 0 0 0; font-size: 0.9rem; color: #38bdf8;"><strong>Recommended Action:</strong> {score_data['recommended_action']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Granular Category Scores
            st.markdown("##### 📊 **Granular Category Scores**")
            b_sc, a_sc = score_data.get("budget_score", 0), score_data.get("authority_score", 0)
            n_sc, t_sc = score_data.get("need_score", 0), score_data.get("timeline_score", 0)

            c_b, c_a = st.columns(2)
            with c_b:
                st.markdown(f"**💰 Budget:** `{b_sc}/25`")
                st.progress(b_sc / 25)
                st.caption(score_data.get("budget_signal", ""))
            with c_a:
                st.markdown(f"**👑 Authority:** `{a_sc}/25`")
                st.progress(a_sc / 25)
                st.caption(score_data.get("authority_signal", ""))

            c_n, c_t = st.columns(2)
            with c_n:
                st.markdown(f"**🎯 Need:** `{n_sc}/25`")
                st.progress(n_sc / 25)
                st.caption(score_data.get("need_signal", ""))
            with c_t:
                st.markdown(f"**⏳ Timeline:** `{t_sc}/25`")
                st.progress(t_sc / 25)
                st.caption(score_data.get("timeline_signal", ""))

        with res_col2:
            st.markdown("##### ✉️ **Agent 2: Generated Outreach Email**")
            st.markdown(f"""
            <div class="email-container">
                <div style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 12px; margin-bottom: 12px;">
                    <div style="font-size: 0.85rem; color: #94a3b8;">RECIPIENT: <span style="color: #f8fafc;">{email_data['recipient_email']}</span></div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">TYPE: <span style="color: {'#34d399' if is_qual else '#fb7185'}; font-weight: 600;">{email_data['email_type'].replace('_', ' ').title()}</span></div>
                    <div style="font-size: 1rem; color: #f8fafc; font-weight: 700; margin-top: 8px;">SUBJECT: {email_data['subject']}</div>
                </div>
                <div style="white-space: pre-wrap; line-height: 1.6; color: #e2e8f0; font-size: 0.95rem;">{email_data['body']}</div>
                <div style="margin-top: 16px; padding-top: 12px; border-top: 1px dashed rgba(255, 255, 255, 0.1); font-size: 0.85rem; color: #38bdf8;">
                    <strong>Proposed Next Step (CTA):</strong> {email_data['call_to_action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_send1, col_send2 = st.columns([1, 1])
            with col_send1:
                if st.button("🚀 Send Email via Gmail", key="btn_send_single", use_container_width=True):
                    with st.spinner("Dispatching through Gmail API / Safe Simulator..."):
                        delivery_receipt = st.session_state.email_service.send_email(
                            recipient=email_data["recipient_email"],
                            subject=email_data["subject"],
                            body=email_data["body"]
                        )
                        st.session_state.scored_leads[-1]["email_delivery_status"] = delivery_receipt["status"]
                        st.session_state.scored_leads[-1]["email_sent_at"] = delivery_receipt["timestamp"]
                        st.session_state.scored_leads[-1]["delivery_notes"] = delivery_receipt.get("notes") or delivery_receipt.get("error")

                    if delivery_receipt["status"] in ("sent", "simulated"):
                        st.success(f"Delivered! Status: {delivery_receipt['status'].upper()} (ID: {delivery_receipt.get('message_id', 'N/A')})")
                    else:
                        st.error(f"Delivery failed: {delivery_receipt.get('error')}")


# ==========================================
# TAB 3: BATCH PROCESSING & DATASET
# ==========================================
with tab_batch:
    st.markdown("#### 📁 **Batch Qualification & Dataset Processing**")
    st.caption("Load beauty industry leads dataset (`leads.csv`) or upload custom CSV for autonomous scoring.")

    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        st.markdown("##### 📂 **Current Dataset**")
        st.write(f"Loaded **{len(st.session_state.raw_leads)} leads** from Kaggle Beauty BANT Dataset (`leads.csv`).")
        if st.session_state.raw_leads:
            df_raw = pd.DataFrame(st.session_state.raw_leads)
            st.dataframe(df_raw[["Name", "Company", "Job Title", "Industry", "Size"]], use_container_width=True, height=200)

    with col_b2:
        st.markdown("##### ⚡ **Batch Run Controls**")
        batch_limit = st.slider("Leads to process in batch", min_value=1, max_value=max(1, len(st.session_state.raw_leads)), value=min(5, len(st.session_state.raw_leads)))
        auto_send_emails = st.checkbox("Auto-send / simulate email delivery during batch processing", value=False)
        run_batch_btn = st.button("🚀 Start Batch Qualification", use_container_width=True)

    if run_batch_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        batch_results = []
        leads_to_run = st.session_state.raw_leads[:batch_limit]

        for idx, lead in enumerate(leads_to_run):
            status_text.text(f"Processing ({idx+1}/{len(leads_to_run)}): {lead.get('Name')} at {lead.get('Company')}...")
            res = process_lead_pipeline(
                lead=lead,
                threshold=qualification_threshold,
                send_email=auto_send_emails,
                email_service=st.session_state.email_service,
                use_llm=use_llm,
            )
            batch_results.append(res)
            progress_bar.progress((idx + 1) / len(leads_to_run))

        st.session_state.scored_leads = batch_results
        status_text.success(f"Batch completed! Successfully scored {len(batch_results)} leads.")

        # Save to file
        with open("leads_scored.json", "w", encoding="utf-8") as f:
            json.dump(batch_results, f, indent=2)

    st.divider()

    # Scored Leads Table
    if st.session_state.scored_leads:
        st.markdown("##### 📋 **Scored Leads Results Table**")
        flat_records = []
        for r in st.session_state.scored_leads:
            lead = r.get("lead_data", {})
            score = r.get("bant_score", {})
            email = r.get("email_response") or {}
            flat_records.append({
                "Name": lead.get("Name"),
                "Company": lead.get("Company"),
                "Job Title": lead.get("Job Title"),
                "BANT Score": score.get("bant_score"),
                "Status": score.get("qualification"),
                "Budget": score.get("budget_score"),
                "Authority": score.get("authority_score"),
                "Need": score.get("need_score"),
                "Timeline": score.get("timeline_score"),
                "Email Type": email.get("email_type"),
                "Delivery Status": r.get("email_delivery_status", "draft"),
            })

        df_flat = pd.DataFrame(flat_records)

        # Filters
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            status_filter = st.selectbox("Filter by Status", ["All", "Qualified", "Unqualified"])
        with f_col2:
            min_score_filter = st.slider("Minimum BANT Score", 0, 100, 0)

        filtered_df = df_flat.copy()
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["Status"] == status_filter]
        filtered_df = filtered_df[filtered_df["BANT Score"] >= min_score_filter]

        st.dataframe(filtered_df, use_container_width=True)

        # Export Buttons
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Filtered Leads to CSV",
                data=csv_data,
                file_name="leads_scored_export.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_exp2:
            json_str = json.dumps(st.session_state.scored_leads, indent=2)
            st.download_button(
                "📥 Export Full Results to JSON",
                data=json_str,
                file_name="leads_scored_full.json",
                mime="application/json",
                use_container_width=True
            )


# ==========================================
# TAB 4: EMAIL COMMUNICATION HUB
# ==========================================
with tab_emails:
    st.markdown("#### ✉️ **Email Communication Hub & Outbox**")
    st.caption("Review, edit, and dispatch generated outreach and nurturing emails with safe delivery safeguards.")

    if not st.session_state.scored_leads:
        st.info("No leads available. Please score leads in Tab 2 or Tab 3 first.", icon="💡")
    else:
        lead_names = [f"{l.get('lead_data', {}).get('Name')} ({l.get('lead_data', {}).get('Company')})" for l in st.session_state.scored_leads]
        selected_lead_idx = st.selectbox("Select Lead Outbox Entry:", range(len(lead_names)), format_func=lambda x: lead_names[x])

        target_lead_res = st.session_state.scored_leads[selected_lead_idx]
        email_info = target_lead_res.get("email_response")

        if not email_info:
            st.warning("No email draft generated for this lead.")
        else:
            col_em1, col_em2 = st.columns([1.5, 1])

            with col_em1:
                st.markdown("##### 📝 **Editable Email Draft**")
                edit_recipient = st.text_input("Recipient Email", value=email_info.get("recipient_email", ""), key="em_rec")
                edit_subject = st.text_input("Subject Line", value=email_info.get("subject", ""), key="em_subj")
                edit_body = st.text_area("Email Content", value=email_info.get("body", ""), height=260, key="em_body")

                col_btn_save, col_btn_send = st.columns(2)
                with col_btn_save:
                    if st.button("💾 Save Edits to Draft", use_container_width=True):
                        st.session_state.scored_leads[selected_lead_idx]["email_response"]["subject"] = edit_subject
                        st.session_state.scored_leads[selected_lead_idx]["email_response"]["body"] = edit_body
                        st.session_state.scored_leads[selected_lead_idx]["email_response"]["recipient_email"] = edit_recipient
                        st.success("Draft updated successfully!")

                with col_btn_send:
                    confirm_send = st.checkbox("Confirm dispatch", value=True, key="chk_confirm")
                    if st.button("📨 Dispatch Email", use_container_width=True):
                        if not confirm_send:
                            st.warning("Please check the confirmation box before sending.")
                        else:
                            receipt = st.session_state.email_service.send_email(
                                recipient=edit_recipient,
                                subject=edit_subject,
                                body=edit_body,
                            )
                            st.session_state.scored_leads[selected_lead_idx]["email_delivery_status"] = receipt["status"]
                            st.session_state.scored_leads[selected_lead_idx]["email_sent_at"] = receipt["timestamp"]
                            st.session_state.scored_leads[selected_lead_idx]["delivery_notes"] = receipt.get("notes") or receipt.get("error")

                            if receipt["status"] in ("sent", "simulated"):
                                st.success(f"Delivered! Mode: {receipt['method'].upper()} | Msg ID: {receipt.get('message_id')}")
                            else:
                                st.error(f"Delivery Error: {receipt.get('error')}")

            with col_em2:
                st.markdown("##### 📊 **Lead Summary & Status**")
                sc = target_lead_res.get("bant_score", {})
                st.markdown(f"""
                <div class="bant-card">
                    <p style="margin: 0;"><strong>Prospect:</strong> {target_lead_res.get('lead_data', {}).get('Name')}</p>
                    <p style="margin: 4px 0;"><strong>Company:</strong> {target_lead_res.get('lead_data', {}).get('Company')}</p>
                    <p style="margin: 4px 0;"><strong>Score:</strong> <span style="color: #38bdf8; font-weight: bold;">{sc.get('bant_score')}/100</span> ({sc.get('qualification')})</p>
                    <p style="margin: 4px 0;"><strong>Delivery Status:</strong> <code>{target_lead_res.get('email_delivery_status', 'draft')}</code></p>
                    <p style="margin: 4px 0; font-size: 0.85rem; color: #94a3b8;"><strong>Timestamp:</strong> {target_lead_res.get('email_sent_at') or 'Not yet transmitted'}</p>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.markdown("##### 📜 **Delivery Audit Trail Log**")
        audit_records = st.session_state.email_service.get_audit_log()
        if audit_records:
            st.dataframe(pd.DataFrame(audit_records), use_container_width=True)
        else:
            st.caption("No emails transmitted or simulated in this session yet.")


# ==========================================
# TAB 5: ARCHITECTURE & SEMINAR SPECS
# ==========================================
with tab_about:
    st.markdown("#### 🏛️ **System Architecture & Progress Seminar-II Specifications**")

    col_sp1, col_sp2 = st.columns([1.2, 1])

    with col_sp1:
        st.markdown("""
        ##### 📌 **Major Project Overview**
        - **Project Title:** LeadSense AI — Intelligent BANT-Based Sales Lead Qualification System
        - **Academic Year:** 2026 - 2027
        - **Department:** Department of Artificial Intelligence
        - **Institution:** GHRCE, Nagpur

        ##### 👥 **Project Team**
        1. **Amisha Gillarkar** (A-01)
        2. **Dnyaneshwari Patharkar** (A-13)
        3. **Samiksha Kapate** (A-17)
        4. **Atharva Bajpai** (A-36)
        - **Project Guide:** Prof. Saundarya Raut

        ##### 🔄 **7-Step End-to-End Methodology (per Seminar PPT)**
        1. **Step 1: Lead Acquisition** — Capture customer requirements via CSV, API, or interactive input.
        2. **Step 2: Lead Pre-processing** — Extract name, company, budget, timeline, and job designation.
        3. **Step 3: BANT Analysis** — Agent 1 evaluates: Budget + Authority + Need + Timeline (0-25 each).
        4. **Step 4: Lead Classification** — Calculate Total BANT Score = B + A + N + T; apply gating rule.
        5. **Step 5: Email Generation** — Agent 2 generates sales pitch for Qualified or nurture for Unqualified.
        6. **Step 6: Email Sending** — GCP + Gmail API transmission with built-in safe demo simulation.
        7. **Step 7: User Interface** — Interactive Streamlit dashboard with real-time analytics.
        """)

    with col_sp2:
        st.markdown("""
        ##### 🤖 **Multi-Agent Architecture**
        - **Agent 1:** `BANT Lead Qualification Specialist`
          * LLM: Google Gemini (`gemini-3.6-flash`)
          * Objective: Structured extraction of BANT dimensions with gating rule enforcement.
        - **Agent 2:** `Sales Outreach & Nurture Specialist`
          * Objective: Hyper-personalized B2B emails with distinct pitches and CTAs.
        - **Safety & Delivery Layer:**
          * Dual Protocol: GCP Gmail API (OAuth2) + Gmail SMTP (App Password)
          * Safe Demo Mode: Automated validation & synthetic delivery simulation.
        """)

        st.info("Dataset: Beauty Industry BANT Leads Dataset (Kaggle realistic enterprise leads).", icon="💄")
