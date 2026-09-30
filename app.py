import os
import csv
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

from crew import (
    process_lead_pipeline,
    is_quota_exhausted,
    reset_quota_status,
)
from email_service import default_email_service

# Page Configuration
st.set_page_config(
    page_title="LeadSense | Lead Qualification",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if st.session_state.dark_mode:
    theme = {
        "scheme": "dark", "background": "#18211f", "surface": "#222e2b",
        "surface_alt": "#2c3936", "text": "#edf4f1", "muted": "#a2b3ad",
        "border": "#3a4a45", "accent": "#62c4ae", "accent_hover": "#83d4c2",
        "success": "#58c895", "warning": "#e6ad55", "danger": "#ed7770",
        "grid": "#40504b",
    }
else:
    theme = {
        "scheme": "light", "background": "#f4f7f6", "surface": "#ffffff",
        "surface_alt": "#eaf1ef", "text": "#24312e", "muted": "#63736e",
        "border": "#d4dfdb", "accent": "#147b68", "accent_hover": "#0d6656",
        "success": "#16835d", "warning": "#a86212", "danger": "#bd4942",
        "grid": "#dce5e2",
    }

st.markdown(f"""
<style>
    :root, .stApp {{
        color-scheme: {theme['scheme']};
        --background-color: {theme['background']};
        --secondary-background-color: {theme['surface']};
        --text-color: {theme['text']};
        --primary-color: {theme['accent']};
        --app-bg: {theme['background']};
        --surface: {theme['surface']};
        --surface-alt: {theme['surface_alt']};
        --text: {theme['text']};
        --muted: {theme['muted']};
        --border: {theme['border']};
        --accent: {theme['accent']};
        --accent-hover: {theme['accent_hover']};
        --success: {theme['success']};
        --warning: {theme['warning']};
        --danger: {theme['danger']};
        --grid: {theme['grid']};
    }}
    html, body, [class*="css"] {{ font-family: 'Segoe UI', 'Aptos', sans-serif; }}
    .stApp, [data-testid="stAppViewContainer"] {{ background: var(--app-bg) !important; color: var(--text) !important; }}
    [data-testid="stHeader"] {{ background: var(--app-bg) !important; }}
    [data-testid="stSidebar"] {{ background: var(--surface) !important; border-right: 1px solid var(--border); color: var(--text) !important; }}
    [data-testid="stMetric"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 14px; box-shadow: none; }}
    [data-testid="stMarkdownContainer"] :is(h1, h2, h3, h4, h5, h6, p, li), [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, [data-testid="stRadio"] label, [data-testid="stRadio"] label * {{ color: var(--text-color) !important; }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: var(--muted) !important; }}
    [data-baseweb="input"] > div, [data-baseweb="textarea"] > div, [data-baseweb="select"] > div {{ background: var(--surface) !important; border-color: var(--border) !important; }}
    [data-baseweb="input"] *, [data-baseweb="textarea"] *, [data-baseweb="select"] * {{ color: var(--text-color) !important; }}
    input, textarea {{ color: var(--text-color) !important; }}
    [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {{ background: var(--accent) !important; border-color: var(--accent) !important; }}
    [data-testid="stSlider"] div[style*="left:"] {{ background: var(--accent) !important; }}
    [data-testid="stSliderThumbValue"] {{ border-color: var(--accent) !important; }}
    .stButton > button, .stFormSubmitButton > button {{ border-radius: 4px; }}
    .stButton > button[kind="primary"], .stFormSubmitButton > button {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
    .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {{ background: var(--accent-hover); border-color: var(--accent-hover); }}
    [role="tab"], [role="tab"] * {{ color: var(--muted) !important; }}
    [role="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] * {{ color: var(--accent) !important; }}
    [role="tab"] [data-testid="stMarkdownContainer"] p {{ color: var(--muted) !important; }}
    [role="tab"][aria-selected="true"] [data-testid="stMarkdownContainer"] p {{ color: var(--accent) !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background: var(--accent) !important; }}
    hr {{ border-color: var(--border) !important; }}
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
    st.markdown("## LeadSense")
    st.caption("Lead qualification workspace")
    st.toggle("Dark mode", key="dark_mode")
    st.divider()

    st.markdown("#### Processing")
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
    st.markdown("#### Email delivery")
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
st.title("Lead qualification")
st.caption("Review inbound leads, qualification signals, and recommended next steps.")


# Main Navigation Tabs
tab_overview, tab_single, tab_batch, tab_emails = st.tabs([
    "Overview",
    "Qualify a lead",
    "Batch processing",
    "Email outbox",
])


# ==========================================
# TAB 1: EXECUTIVE DASHBOARD
# ==========================================
with tab_overview:
    scored_leads = st.session_state.scored_leads

    if not scored_leads:
        st.info("No leads scored yet. Qualify a lead or start a batch run to populate the dashboard.")
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
            st.markdown("##### BANT dimension averages")
            b_avg = sum(l.get("bant_score", {}).get("budget_score", 0) for l in scored_leads) / total_leads
            a_avg = sum(l.get("bant_score", {}).get("authority_score", 0) for l in scored_leads) / total_leads
            n_avg = sum(l.get("bant_score", {}).get("need_score", 0) for l in scored_leads) / total_leads
            t_avg = sum(l.get("bant_score", {}).get("timeline_score", 0) for l in scored_leads) / total_leads

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[b_avg, a_avg, n_avg, t_avg, b_avg],
                theta=['Budget (25)', 'Authority (25)', 'Need (25)', 'Timeline (25)', 'Budget (25)'],
                fill='toself',
                fillcolor=theme["accent"],
                opacity=0.22,
                line=dict(color=theme["accent"], width=2),
                name='Average Score'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 25], gridcolor=theme["grid"], color=theme["muted"]),
                    angularaxis=dict(gridcolor=theme["grid"], color=theme["muted"]),
                    bgcolor=theme["surface"]
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=20, b=20),
                height=320,
            )
            st.plotly_chart(fig_radar, width="stretch")

        with col_chart2:
            st.markdown("##### Lead score distribution")
            scores = [l.get("bant_score", {}).get("bant_score", 0) for l in scored_leads]
            names = [l.get("lead_data", {}).get("Name", "Lead") for l in scored_leads]
            qual_labels = [l.get("bant_score", {}).get("qualification", "Unqualified") for l in scored_leads]

            df_dist = pd.DataFrame({"Lead": names, "BANT_Score": scores, "Status": qual_labels})
            fig_bar = px.bar(
                df_dist,
                x="Lead",
                y="BANT_Score",
                color="Status",
                color_discrete_map={"Qualified": theme["success"], "Unqualified": theme["danger"]},
                text="BANT_Score",
            )
            fig_bar.add_hline(
                y=qualification_threshold,
                line_dash="dot",
                line_color=theme["warning"],
                annotation_text=f"Threshold ({qualification_threshold})",
                annotation_position="bottom right"
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor=theme["surface"],
                font=dict(color=theme["text"]),
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
                xaxis=dict(gridcolor=theme["grid"]),
                yaxis=dict(gridcolor=theme["grid"], range=[0, 105]),
            )
            st.plotly_chart(fig_bar, width="stretch")


# ==========================================
# TAB 2: SINGLE LEAD QUALIFICATION
# ==========================================
with tab_single:
    st.markdown("#### Qualify a lead")
    st.caption("Enter prospect details to review BANT scores and a suggested follow-up.")

    # Preset selection
    sample_leads = st.session_state.raw_leads
    preset_options = ["Enter a new lead"] + [f"{l.get('Name')} — {l.get('Company')} ({l.get('Job Title')})" for l in sample_leads]
    selected_preset = st.selectbox("Load an existing lead", preset_options)

    default_name = ""
    default_title = ""
    default_company = ""
    default_email = ""
    default_phone = ""
    default_industry = ""
    default_size = "Not specified"
    default_notes = ""

    if selected_preset != "Enter a new lead":
        idx = preset_options.index(selected_preset) - 1
        lead_preset = sample_leads[idx]
        default_name = lead_preset.get("Name", "")
        default_title = lead_preset.get("Job Title", "")
        default_company = lead_preset.get("Company", "")
        default_email = lead_preset.get("Email", "")
        default_phone = lead_preset.get("Phone", "")
        default_industry = lead_preset.get("Industry", "")
        default_size = lead_preset.get("Size", "Not specified") or "Not specified"
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
            size_options = ["Not specified", "Small (1-50)", "Medium (51-200)", "Large (201-1000)", "Enterprise (1000+)"]
            if default_size not in size_options:
                size_options.append(default_size)
            size_input = st.selectbox("Company Size", size_options, index=size_options.index(default_size))

        notes_input = st.text_area(
            "Lead notes",
            value=default_notes,
            placeholder="Include the prospect's needs, budget, decision role, and expected timeline.",
            height=130,
        )
        submit_btn = st.form_submit_button("Run qualification", type="primary", width="stretch")

    if submit_btn and not name_input.strip():
        st.error("Enter a lead name before running qualification.")

    if submit_btn and name_input.strip():
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

        with open("leads_scored.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.scored_leads, f, indent=2)

        # Display Results
        score_data = result["bant_score"]
        email_data = result.get("email_response")
        classification = result.get("classification") or {"classification": "GENUINE_LEAD", "confidence": 0.0, "reason": ""}
        intent = result.get("intent") or {"intent": "UNKNOWN", "confidence": 0.0, "evidence": ""}
        sentiment = result.get("sentiment") or {"sentiment": "NEUTRAL", "confidence": 0.0, "reason": ""}
        nxt = result.get("next_best_action") or {"action": "MANUAL_REVIEW", "reason": "No action generated.", "priority": "MEDIUM"}
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Qualification results")

        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            st.metric("BANT score", f"{score_data['bant_score']} / 100")
            st.caption(f"Qualification: {score_data['qualification']}")
            st.caption(f"Qualification threshold: {qualification_threshold} / 100")
            st.progress(score_data["bant_score"] / 100)
            st.write("**Rationale**")
            st.write(score_data["rationale"])
            st.write("**Recommended action**")
            st.write(score_data["recommended_action"])

            st.markdown("#### Lead signals")
            signal_cols = st.columns(3)
            signal_cols[0].metric("Classification", classification["classification"], f"{classification['confidence']:.0%} confidence")
            signal_cols[1].metric("Intent", intent["intent"], f"{intent['confidence']:.0%} confidence")
            signal_cols[2].metric("Sentiment", sentiment["sentiment"])
            st.info(f"Next action: {nxt['action']} · {nxt['priority']} priority\n{nxt['reason']}")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("##### BANT breakdown")
            b_sc, a_sc = score_data.get("budget_score", 0), score_data.get("authority_score", 0)
            n_sc, t_sc = score_data.get("need_score", 0), score_data.get("timeline_score", 0)

            c_b, c_a = st.columns(2)
            with c_b:
                st.markdown(f"**Budget:** `{b_sc}/25`")
                st.progress(b_sc / 25)
                st.caption(score_data.get("budget_signal", ""))
            with c_a:
                st.markdown(f"**Authority:** `{a_sc}/25`")
                st.progress(a_sc / 25)
                st.caption(score_data.get("authority_signal", ""))

            c_n, c_t = st.columns(2)
            with c_n:
                st.markdown(f"**Need:** `{n_sc}/25`")
                st.progress(n_sc / 25)
                st.caption(score_data.get("need_signal", ""))
            with c_t:
                st.markdown(f"**Timeline:** `{t_sc}/25`")
                st.progress(t_sc / 25)
                st.caption(score_data.get("timeline_signal", ""))

        with res_col2:
            if email_data:
                st.markdown("##### Draft preview")
                st.caption(f"To: {email_data['recipient_email']} · {email_data['email_type'].replace('_', ' ').title()}")
                st.write("**Subject**")
                st.text(email_data["subject"])
                st.text_area("Email body", value=email_data["body"], height=220, disabled=True, key="single_email_preview")
                st.caption(f"Suggested next step: {email_data['call_to_action']}")

                st.markdown("<br>", unsafe_allow_html=True)
                col_send1, col_send2 = st.columns([1, 1])
                with col_send1:
                    if st.button("Send email", key="btn_send_single", type="primary", width="stretch"):
                        with st.spinner("Sending email..."):
                            delivery_receipt = st.session_state.email_service.send_email(
                                recipient=email_data["recipient_email"],
                                subject=email_data["subject"],
                                body=email_data["body"]
                            )
                            st.session_state.scored_leads[-1]["email_delivery_status"] = delivery_receipt["status"]
                            st.session_state.scored_leads[-1]["email_sent_at"] = delivery_receipt["timestamp"]
                            st.session_state.scored_leads[-1]["delivery_notes"] = delivery_receipt.get("notes") or delivery_receipt.get("error")

                        if delivery_receipt["status"] in ("sent", "simulated"):
                            st.success(f"Email {delivery_receipt['status']}.")
                        else:
                            st.error(f"Delivery failed: {delivery_receipt.get('error')}")
            else:
                st.warning("BANT analysis skipped because this message was classified as spam or non-sales.")


# ==========================================
# TAB 3: BATCH PROCESSING & DATASET
# ==========================================
with tab_batch:
    st.markdown("#### Batch processing")
    uploaded_csv = st.file_uploader("Upload leads CSV", type=["csv"], key="batch_leads_csv")
    batch_leads = st.session_state.raw_leads
    batch_source = "leads.csv"

    if uploaded_csv is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_csv, keep_default_na=False, encoding="utf-8-sig")
            column_aliases = {
                "name": "Name", "lead name": "Name",
                "company": "Company", "company name": "Company", "organization": "Company",
                "email": "Email", "email address": "Email",
                "job title": "Job Title", "title": "Job Title", "designation": "Job Title",
                "phone": "Phone", "industry": "Industry", "size": "Size",
                "notes": "Notes", "inquiry": "Notes", "message": "Notes", "sales inquiry": "Notes",
            }
            uploaded_df = uploaded_df.rename(
                columns={column: column_aliases.get(str(column).strip().lower(), str(column).strip()) for column in uploaded_df.columns}
            )
            missing_columns = [column for column in ("Name", "Notes") if column not in uploaded_df.columns]
            if uploaded_df.empty:
                st.error("The uploaded CSV has no lead rows.")
                batch_leads = []
            elif missing_columns:
                st.error(f"CSV must include these columns: {', '.join(missing_columns)}.")
                batch_leads = []
            else:
                for column in ("Company", "Job Title", "Email", "Phone", "Industry", "Size"):
                    if column not in uploaded_df.columns:
                        uploaded_df[column] = ""
                batch_leads = uploaded_df.to_dict(orient="records")
                batch_source = uploaded_csv.name
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
            st.error(f"Could not read the CSV: {error}")
            batch_leads = []

    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        st.markdown("##### Leads to process")
        st.caption(f"{len(batch_leads)} leads from {batch_source}.")
        if batch_leads:
            df_raw = pd.DataFrame(batch_leads)
            preview_columns = [column for column in ("Name", "Company", "Job Title", "Email", "Industry", "Size") if column in df_raw.columns]
            st.dataframe(df_raw[preview_columns], width="stretch", height=200)

    with col_b2:
        st.markdown("##### Run controls")
        if len(batch_leads) > 1:
            batch_limit = st.slider(
                "Leads to process",
                min_value=1,
                max_value=len(batch_leads),
                value=min(5, len(batch_leads)),
            )
        else:
            batch_limit = len(batch_leads)
            st.caption(f"{batch_limit} lead{'s' if batch_limit != 1 else ''} will be processed.")
        auto_send_emails = st.checkbox("Auto-send / simulate email delivery during batch processing", value=False)
        run_batch_btn = st.button("Run batch", type="primary", width="stretch", disabled=not batch_leads)

    if run_batch_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        batch_results = []
        leads_to_run = batch_leads[:batch_limit]

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
        status_text.success(f"Batch complete. Processed {len(batch_results)} leads.")

        # Save to file
        with open("leads_scored.json", "w", encoding="utf-8") as f:
            json.dump(batch_results, f, indent=2)

    st.divider()

    # Scored Leads Table
    if st.session_state.scored_leads:
        st.markdown("##### Scored leads")
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

        st.dataframe(filtered_df, width="stretch")

        # Export Buttons
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export filtered CSV",
                data=csv_data,
                file_name="leads_scored_export.csv",
                mime="text/csv",
                width="stretch"
            )
        with col_exp2:
            json_str = json.dumps(st.session_state.scored_leads, indent=2)
            st.download_button(
                "Export full results to JSON",
                data=json_str,
                file_name="leads_scored_full.json",
                mime="application/json",
                width="stretch"
            )


# ==========================================
# TAB 4: EMAIL COMMUNICATION HUB
# ==========================================
with tab_emails:
    st.markdown("#### Email outbox")
    st.caption("Review generated drafts and send them when ready.")

    if not st.session_state.scored_leads:
        st.info("No leads available. Qualify a lead or run a batch first.")
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
                st.markdown("##### Email draft")
                edit_recipient = st.text_input("Recipient Email", value=email_info.get("recipient_email", ""), key="em_rec")
                edit_subject = st.text_input("Subject Line", value=email_info.get("subject", ""), key="em_subj")
                edit_body = st.text_area("Email Content", value=email_info.get("body", ""), height=260, key="em_body")

                col_btn_save, col_btn_send = st.columns(2)
                with col_btn_save:
                    if st.button("Save draft", width="stretch"):
                        st.session_state.scored_leads[selected_lead_idx]["email_response"]["subject"] = edit_subject
                        st.session_state.scored_leads[selected_lead_idx]["email_response"]["body"] = edit_body
                        st.session_state.scored_leads[selected_lead_idx]["email_response"]["recipient_email"] = edit_recipient
                        st.success("Draft updated successfully!")

                with col_btn_send:
                    confirm_send = st.checkbox("Confirm dispatch", value=True, key="chk_confirm")
                    if st.button("Send email", type="primary", width="stretch"):
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
                st.markdown("##### Lead summary")
                sc = target_lead_res.get("bant_score", {})
                st.write(f"**Prospect:** {target_lead_res.get('lead_data', {}).get('Name', 'Unknown')}")
                st.write(f"**Company:** {target_lead_res.get('lead_data', {}).get('Company', 'Unknown')}")
                st.metric("BANT score", f"{sc.get('bant_score', 0)}/100")
                st.caption(f"Qualification: {sc.get('qualification', 'Unqualified')}")
                st.write(f"**Delivery:** {target_lead_res.get('email_delivery_status', 'draft')}")
                st.caption(f"Sent: {target_lead_res.get('email_sent_at') or 'Not yet'}")

        st.divider()
        st.markdown("##### Delivery log")
        audit_records = st.session_state.email_service.get_audit_log()
        if audit_records:
            st.dataframe(pd.DataFrame(audit_records), width="stretch")
        else:
            st.caption("No emails transmitted or simulated in this session yet.")
