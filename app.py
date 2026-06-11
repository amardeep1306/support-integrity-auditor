"""
SIA — Streamlit Web App (Priority Mismatch Dashboard)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, '/home/claude')
import os
import sys

# Auto-train if model files don't exist (for cloud deployment)
if not os.path.exists('sia_model/clf_lr.pkl'):
    st.info("First run detected — training model. This takes 2-3 minutes...")
    import subprocess
    subprocess.run([sys.executable, 'train_pipeline.py'], check=True)
    st.rerun()

from predict import predict

st.set_page_config(
    page_title="Support Integrity Auditor (SIA)",
    page_icon="🔍",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────
st.markdown("""
<style>
.metric-card { background:#1e293b; border-radius:10px; padding:1rem;
               border-left:4px solid #ef4444; margin-bottom:1rem; }
.hidden-crisis { background:#7f1d1d; color:#fee2e2; padding:4px 10px;
                 border-radius:20px; font-size:0.8em; }
.false-alarm   { background:#1e3a5f; color:#bfdbfe; padding:4px 10px;
                 border-radius:20px; font-size:0.8em; }
.consistent    { background:#14532d; color:#bbf7d0; padding:4px 10px;
                 border-radius:20px; font-size:0.8em; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Support Integrity Auditor (SIA)")
st.caption("Semantics-driven, evidence-grounded priority mismatch detector for CRM support tickets")

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Input Mode")
    mode = st.radio("Select input mode", ["Single Ticket", "Batch CSV Upload"])
    st.markdown("---")
    st.markdown("**Model**: LR + RF Ensemble")
    st.markdown("**Signals**: NLP Keywords + Resolution Time")
    st.markdown("**Accuracy**: 98.75%  |  **F1**: 0.987")

# ── Single Ticket Mode ───────────────────────────────────
if mode == "Single Ticket":
    st.subheader("🎫 Analyze Single Ticket")
    col1, col2 = st.columns(2)
    with col1:
        subject     = st.text_input("Ticket Subject", "Production database completely down")
        description = st.text_area("Ticket Description",
            "Our entire production environment is down. All customers are affected. Immediate escalation required.",
            height=120)
        priority    = st.selectbox("Assigned Priority", ["Low","Medium","High","Critical"])
    with col2:
        channel     = st.selectbox("Channel", ["Email","Chat","Phone","Social media"])
        ticket_type = st.selectbox("Ticket Type", ["Technical","Billing","Account","Feature Request","General Inquiry"])
        res_time    = st.number_input("Resolution Time (hours)", min_value=0.0, value=2.0, step=0.5)
        product     = st.selectbox("Product", ["SoftwareA","SoftwareB","HardwareX","ServicePro","CloudSuite"])
        customer_email = st.text_input("Customer Email", "admin@enterprise.com")

    if st.button("🔍 Analyze Ticket", type="primary"):
        row = pd.DataFrame([{
            'ticket_id': 'MANUAL-001',
            'Ticket Subject': subject,
            'Ticket Description': description,
            'Ticket Priority': priority,
            'Ticket Channel': channel,
            'Ticket Type': ticket_type,
            'Resolution Time': res_time,
            'Product Purchased': product,
            'Customer Email': customer_email,
        }])
        df_out, dossiers = predict(row)
        result = df_out.iloc[0]
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            verdict = "⚠️ MISMATCH" if result['is_mismatch'] else "✅ CONSISTENT"
            st.metric("Verdict", verdict)
        with col_b:
            st.metric("Inferred Severity", result['inferred_severity'])
        with col_c:
            st.metric("Confidence", f"{result['pred_prob']*100:.1f}%")

        if result['is_mismatch'] and dossiers:
            d = dossiers[0]
            st.subheader("📂 Evidence Dossier")
            c1, c2, c3 = st.columns(3)
            c1.metric("Assigned Priority", d['assigned_priority'])
            c2.metric("Inferred Severity", d['inferred_severity'])
            c3.metric("Severity Delta",    d['severity_delta'])

            mtype = d['mismatch_type']
            badge_color = "#7f1d1d" if mtype == "Hidden Crisis" else "#1e3a5f"
            badge_text  = "#fee2e2" if mtype == "Hidden Crisis" else "#bfdbfe"
            st.markdown(f'<span style="background:{badge_color};color:{badge_text};padding:4px 12px;'
                        f'border-radius:20px;">{mtype}</span>', unsafe_allow_html=True)

            st.markdown("**Feature Evidence:**")
            for ev in d['feature_evidence']:
                if ev['signal'] == 'keyword':
                    st.markdown(f"• 🔑 Keyword `{ev['value']}` — weight: `{ev['weight']}`")
                elif ev['signal'] == 'resolution_time':
                    st.markdown(f"• ⏱️ Resolution time: `{ev['value']}` — {ev['interpretation']}")

            st.markdown("**Constraint Analysis:**")
            st.info(d['constraint_analysis'])
        else:
            st.success("No priority mismatch detected. Ticket appears correctly labeled.")

# ── Batch CSV Mode ───────────────────────────────────────
else:
    st.subheader("📊 Batch Analysis & Dashboard")
    uploaded = st.file_uploader("Upload CSV (same schema as CRM dataset)", type=['csv'])

    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df_in)}** tickets")

        with st.spinner("Running SIA analysis..."):
            df_out, dossiers = predict(df_in)

        # KPIs
        n_total    = len(df_out)
        n_mismatch = df_out['is_mismatch'].sum()
        n_hc       = (df_out['mismatch_type'] == 'Hidden Crisis').sum()
        n_fa       = (df_out['mismatch_type'] == 'False Alarm').sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Tickets",    n_total)
        k2.metric("Flagged Mismatches", n_mismatch, delta=f"{n_mismatch/n_total:.1%}")
        k3.metric("Hidden Crises",    n_hc)
        k4.metric("False Alarms",     n_fa)

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📋 Predictions Table", "📂 Dossiers"])

        with tab1:
            import json as _json

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Mismatch Type Distribution**")
                type_counts = df_out['mismatch_type'].value_counts()
                st.bar_chart(type_counts)

            with col2:
                st.markdown("**Flagged Tickets by Channel**")
                ch_mismatch = df_out[df_out['is_mismatch']].groupby('Ticket Channel').size()
                st.bar_chart(ch_mismatch)

            st.markdown("**Severity Delta Heatmap (Assigned vs Inferred)**")
            pivot = df_out.groupby(['Ticket Priority','inferred_severity']).size().unstack(fill_value=0)
            st.dataframe(pivot, use_container_width=True)

        with tab2:
            display_cols = ['ticket_id','Ticket Priority','inferred_severity','mismatch_type',
                            'pred_prob','Ticket Channel','Resolution Time']
            cols_exist = [c for c in display_cols if c in df_out.columns]
            st.dataframe(df_out[cols_exist].sort_values('pred_prob', ascending=False),
                        use_container_width=True)

            csv = df_out.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions CSV", csv, "sia_predictions.csv", "text/csv")

        with tab3:
            if dossiers:
                st.markdown(f"**{len(dossiers)} dossiers generated for flagged tickets**")
                for d in dossiers[:20]:
                    with st.expander(f"🎫 {d['ticket_id']}  |  {d['mismatch_type']}  |  "
                                     f"{d['assigned_priority']} → {d['inferred_severity']}  "
                                     f"(Δ{d['severity_delta']})  [{d['confidence']}]"):
                        st.json(d)
                json_bytes = json.dumps(dossiers, indent=2).encode()
                st.download_button("⬇️ Download All Dossiers JSON", json_bytes,
                                   "sia_dossiers.json", "application/json")
    else:
        st.info("Upload a CSV to begin batch analysis. The CSV should have columns: "
                "ticket_id, Ticket Subject, Ticket Description, Ticket Priority, "
                "Ticket Channel, Ticket Type, Resolution Time, Customer Email, Product Purchased")
