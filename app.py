"""
SIA — Streamlit Web App (Improved UI)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys

st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not os.path.exists('sia_model/clf_lr.pkl'):
    st.info("Setting up model for first time. This takes 2-3 minutes...")
    import subprocess
    subprocess.run([sys.executable, 'train_pipeline.py'], check=True)
    st.rerun()

from predict import predict

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0f1117; }
[data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #1e2533; }
.hero { background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%); border: 1px solid #2d3a52; border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 2rem; }
.hero h1 { font-size: 2rem; font-weight: 700; color: #f1f5f9; margin: 0 0 0.5rem 0; }
.hero p { color: #94a3b8; font-size: 1rem; margin: 0; }
.stat-row { display: flex; gap: 1rem; margin-top: 1.5rem; flex-wrap: wrap; }
.stat-pill { background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.3); border-radius: 8px; padding: 0.4rem 0.9rem; font-size: 0.85rem; color: #93c5fd; font-weight: 500; }
.result-mismatch { background: linear-gradient(135deg, #1c0a0a, #2d1515); border: 2px solid #ef4444; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
.result-consistent { background: linear-gradient(135deg, #0a1c0a, #152d15); border: 2px solid #22c55e; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
.verdict-mismatch { font-size: 2rem; font-weight: 700; color: #ef4444; }
.verdict-consistent { font-size: 2rem; font-weight: 700; color: #22c55e; }
.badge-crisis { display: inline-block; background: #7f1d1d; color: #fca5a5; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
.badge-alarm { display: inline-block; background: #1e3a5f; color: #93c5fd; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
.evidence-item { background: #1e2533; border-left: 3px solid #3b82f6; border-radius: 0 8px 8px 0; padding: 0.6rem 1rem; margin: 0.4rem 0; font-size: 0.9rem; color: #cbd5e1; }
.evidence-keyword { border-left-color: #f59e0b; }
.evidence-rt { border-left-color: #8b5cf6; }
.constraint-box { background: #1a2236; border: 1px solid #2d3a52; border-radius: 8px; padding: 1rem 1.2rem; color: #94a3b8; font-size: 0.92rem; line-height: 1.6; margin-top: 0.5rem; }
.kpi-card { background: #161b27; border: 1px solid #1e2533; border-radius: 12px; padding: 1.2rem 1rem; text-align: center; }
.kpi-number { font-size: 2rem; font-weight: 700; color: #f1f5f9; }
.kpi-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }
.kpi-crisis .kpi-number { color: #ef4444; }
.kpi-alarm .kpi-number { color: #3b82f6; }
.kpi-flag .kpi-number { color: #f59e0b; }
.section-title { font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; }
.conf-bar-wrap { background: #1e2533; border-radius: 99px; height: 8px; margin-top: 0.5rem; overflow: hidden; }
.conf-bar-fill { height: 8px; border-radius: 99px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); }
.delta-chip { display: inline-block; padding: 3px 12px; border-radius: 6px; font-size: 1.1rem; font-weight: 700; }
.delta-pos { background: rgba(239,68,68,0.15); color: #ef4444; }
.delta-neg { background: rgba(59,130,246,0.15); color: #3b82f6; }
.sidebar-stat { background: #1e2533; border-radius: 8px; padding: 0.6rem 0.8rem; margin: 0.3rem 0; font-size: 0.85rem; color: #94a3b8; }
.sidebar-stat span { color: #f1f5f9; font-weight: 600; }
.section-card { background: #161b27; border: 1px solid #1e2533; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
label { color: #94a3b8 !important; font-size: 0.85rem !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea, .stNumberInput > div > div > input { background-color: #1e2533 !important; border: 1px solid #2d3a52 !important; color: #f1f5f9 !important; border-radius: 8px !important; }
.stSelectbox > div > div { background-color: #1e2533 !important; border: 1px solid #2d3a52 !important; border-radius: 8px !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #2563eb, #1d4ed8) !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 0.6rem 2rem !important; font-weight: 600 !important; font-size: 0.95rem !important; width: 100% !important; }
.stTabs [data-baseweb="tab-list"] { background: #161b27; border-radius: 8px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 6px; color: #64748b !important; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #1e2d4a !important; color: #93c5fd !important; }
hr { border-color: #1e2533 !important; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR
with st.sidebar:
    st.markdown("## SIA")
    st.markdown("Support Integrity Auditor")
    st.markdown("---")
    mode = st.radio("Mode", ["Single Ticket", "Batch CSV"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="section-title">Model Info</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-stat">Model: <span>LR + RF Ensemble</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-stat">Signals: <span>NLP + Resolution Time</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-stat">Accuracy: <span>98.75%</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-stat">Macro F1: <span>0.987</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-title">Mismatch Types</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem;color:#94a3b8;line-height:1.8;">
    🔴 <b style="color:#fca5a5;">Hidden Crisis</b><br>
    <span style="padding-left:1.2rem;">True severity is higher than labeled</span><br><br>
    🔵 <b style="color:#93c5fd;">False Alarm</b><br>
    <span style="padding-left:1.2rem;">True severity is lower than labeled</span>
    </div>
    """, unsafe_allow_html=True)

# ── HERO
st.markdown("""
<div class="hero">
    <h1>🔍 Support Integrity Auditor</h1>
    <p>Detects priority mismatches in CRM support tickets using NLP and resolution-time signals</p>
    <div class="stat-row">
        <div class="stat-pill">98.75% Accuracy</div>
        <div class="stat-pill">F1 Score 0.987</div>
        <div class="stat-pill">Zero Hallucination</div>
        <div class="stat-pill">Self-Supervised</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SINGLE TICKET MODE
# ══════════════════════════════════════════════════════════
if mode == "Single Ticket":
    st.markdown('<div class="section-title">Ticket Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        subject     = st.text_input("Ticket Subject", placeholder="Enter ticket subject...")
        description = st.text_area("Ticket Description", height=140,
            placeholder="Describe the issue in detail...")
        priority    = st.selectbox("Assigned Priority", ["Low","Medium","High","Critical"])
    with col2:
        res_time       = st.number_input("Resolution Time (hours)", min_value=0.0, value=2.0, step=0.5)
        channel        = st.selectbox("Channel", ["Email","Chat","Phone","Social media"])
        ticket_type    = st.selectbox("Ticket Type", ["Technical","Billing","Account","Feature Request","General Inquiry"])
        customer_email = st.text_input("Customer Email", placeholder="customer@company.com")

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("Analyze Ticket", type="primary")

    if analyze:
        with st.spinner("Analyzing..."):
            row = pd.DataFrame([{
                'ticket_id': 'MANUAL-001',
                'Ticket Subject': subject,
                'Ticket Description': description,
                'Ticket Priority': priority,
                'Ticket Channel': channel,
                'Ticket Type': ticket_type,
                'Resolution Time': res_time,
                'Product Purchased': 'SoftwareA',
                'Customer Email': customer_email,
            }])
            df_out, dossiers = predict(row)
            result = df_out.iloc[0]

        is_mismatch = result['is_mismatch']
        conf = result['pred_prob'] * 100
        st.markdown("---")
        st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

        if is_mismatch and dossiers and result['mismatch_type'] != 'Consistent':
            d = dossiers[0]
            mtype = d['mismatch_type']
            delta = d['severity_delta']
            badge_html = '<span class="badge-crisis">Hidden Crisis</span>' if mtype == "Hidden Crisis" else '<span class="badge-alarm">False Alarm</span>'
            delta_class = "delta-pos" if "+" in str(delta) else "delta-neg"

            st.markdown(f"""
            <div class="result-mismatch">
                <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">
                    <span class="verdict-mismatch">⚠ MISMATCH</span>
                    {badge_html}
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">
                    <div>
                        <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">Assigned</div>
                        <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;">{d['assigned_priority']}</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">Inferred</div>
                        <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;">{d['inferred_severity']}</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">Delta</div>
                        <span class="delta-chip {delta_class}">{delta}</span>
                    </div>
                </div>
                <div style="margin-top:1rem;">
                    <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">Confidence</div>
                    <div style="font-size:1rem;color:#f1f5f9;font-weight:600;">{conf:.1f}%</div>
                    <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{min(conf,100)}%;"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Evidence Dossier</div>', unsafe_allow_html=True)
            for ev in d['feature_evidence']:
                if ev['signal'] == 'keyword':
                    st.markdown(f'<div class="evidence-item evidence-keyword"><b style="color:#fbbf24;">Keyword</b> &nbsp;<code style="background:#2d1f0a;color:#fcd34d;padding:2px 6px;border-radius:4px;">{ev["value"]}</code>&nbsp;—&nbsp;{ev["weight"].replace("_"," ")}</div>', unsafe_allow_html=True)
                elif ev['signal'] == 'resolution_time':
                    st.markdown(f'<div class="evidence-item evidence-rt"><b style="color:#a78bfa;">Resolution Time</b> &nbsp;<code style="background:#1a1230;color:#c4b5fd;padding:2px 6px;border-radius:4px;">{ev["value"]}</code>&nbsp;—&nbsp;{ev["interpretation"]}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Constraint Analysis</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="constraint-box">{d["constraint_analysis"]}</div>', unsafe_allow_html=True)
            with st.expander("View raw dossier JSON"):
                st.json(d)
        else:
            st.markdown(f"""
            <div class="result-consistent">
                <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">
                    <span class="verdict-consistent">✓ CONSISTENT</span>
                </div>
                <div style="color:#86efac;font-size:0.95rem;">No priority mismatch detected. This ticket appears correctly labeled.</div>
                <div style="margin-top:1rem;">
                    <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">Mismatch Probability</div>
                    <div style="font-size:1rem;color:#f1f5f9;font-weight:600;">{conf:.1f}%</div>
                    <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{min(conf,100)}%;background:linear-gradient(90deg,#22c55e,#16a34a);"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# BATCH CSV MODE
# ══════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-title">Batch Analysis</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your CRM tickets CSV", type=['csv'])

    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        with st.spinner(f"Analyzing {len(df_in):,} tickets..."):
            df_out, dossiers = predict(df_in)

        n_total    = len(df_out)
        n_mismatch = int(df_out['is_mismatch'].sum())
        n_hc       = int((df_out['mismatch_type'] == 'Hidden Crisis').sum())
        n_fa       = int((df_out['mismatch_type'] == 'False Alarm').sum())

        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-number">{n_total:,}</div><div class="kpi-label">Total Tickets</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card kpi-flag"><div class="kpi-number">{n_mismatch:,}</div><div class="kpi-label">Flagged ({n_mismatch/n_total:.0%})</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-card kpi-crisis"><div class="kpi-number">{n_hc:,}</div><div class="kpi-label">Hidden Crises</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-card kpi-alarm"><div class="kpi-number">{n_fa:,}</div><div class="kpi-label">False Alarms</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Dashboard", "Predictions", "Dossiers"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Mismatch Type Distribution**")
                st.bar_chart(df_out['mismatch_type'].value_counts())
            with col2:
                st.markdown("**Flagged Tickets by Channel**")
                st.bar_chart(df_out[df_out['is_mismatch']].groupby('Ticket Channel').size())
            st.markdown("**Severity Delta Heatmap**")
            st.dataframe(df_out.groupby(['Ticket Priority','inferred_severity']).size().unstack(fill_value=0), use_container_width=True)

        with tab2:
            display_cols = ['ticket_id','Ticket Priority','inferred_severity','mismatch_type','pred_prob','Ticket Channel','Resolution Time']
            cols_exist = [c for c in display_cols if c in df_out.columns]
            st.dataframe(df_out[cols_exist].sort_values('pred_prob', ascending=False), use_container_width=True)
            st.download_button("Download Predictions CSV", df_out.to_csv(index=False).encode(), "sia_predictions.csv", "text/csv")

        with tab3:
            if dossiers:
                st.markdown(f"**{len(dossiers):,} dossiers generated**")
                for d in dossiers[:25]:
                    badge = "🔴" if d['mismatch_type'] == "Hidden Crisis" else "🔵"
                    with st.expander(f"{badge} {d['ticket_id']}  ·  {d['assigned_priority']} → {d['inferred_severity']}  (Δ {d['severity_delta']})  ·  {d['confidence']}"):
                        st.json(d)
                st.download_button("Download All Dossiers JSON", json.dumps(dossiers, indent=2).encode(), "sia_dossiers.json", "application/json")
    else:
        st.markdown("""
        <div class="section-card" style="text-align:center;padding:3rem;">
            <div style="font-size:2.5rem;margin-bottom:1rem;">📂</div>
            <div style="color:#f1f5f9;font-size:1.1rem;font-weight:600;margin-bottom:0.5rem;">Upload your tickets CSV to begin</div>
            <div style="color:#64748b;font-size:0.9rem;">Required columns: Ticket Subject, Ticket Description, Priority_Level, Ticket_Channel, Resolution_Time_Hours</div>
        </div>
        """, unsafe_allow_html=True)
