"""Demo + research dashboard.  Run:  streamlit run app/streamlit_app.py
Set STROKE_MODEL_DIR / STROKE_ARTIFACTS to point at a trained run (default: artifacts/)."""
import base64
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from stroke_ai.data.imaging import load_image
from stroke_ai.inference import DISCLAIMER, StrokePredictor

ART = Path(os.environ.get("STROKE_ARTIFACTS", "artifacts"))

st.set_page_config(
    page_title="Stroke-AI | Clinical Risk & Neuroimaging Decision Support",
    page_icon="🏥",
    layout="wide",
)

# Hospital / Clinical Light Theme CSS Override (Fixes top black bar, dark inputs, and hidden hover states)
st.markdown(
    """
    <style>
    /* Main Clinical Canvas & Top Streamlit Toolbar */
    .stApp,
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"] {
        background-color: #F4F8FA !important;
        color: #0F172A !important;
    }
    header[data-testid="stHeader"] {
        border-bottom: 1px solid #E2E8F0;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    /* Force Light High-Contrast Styling on All Inputs, Labels & Radio Options */
    label[data-testid="stWidgetLabel"] p,
    div[data-testid="stRadio"] label p,
    div[data-testid="stCheckbox"] label p,
    .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #0F172A !important;
    }
    label[data-testid="stWidgetLabel"] p {
        font-weight: 600 !important;
    }
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    /* Make Horizontal Radio Buttons Look Like Always-Visible Clinical Pills */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 0.35rem 0.75rem;
        margin: 0 !important;
    }
    section[data-testid="stFileUploaderDropzone"] small,
    section[data-testid="stFileUploaderDropzone"] span {
        color: #334155 !important;
    }
    section[data-testid="stFileUploaderDropzone"] button {
        background-color: #0D9488 !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    /* Hospital Header Banner */
    .hospital-banner {
        background: linear-gradient(135deg, #0F766E 0%, #0D9488 60%, #14B8A6 100%);
        padding: 1.35rem 1.75rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.18);
    }
    .hospital-banner h1 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .hospital-banner p {
        color: #CCFBF1 !important;
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Clinical Disclaimer Strip */
    .clinical-notice {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        color: #92400E;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 1.2rem;
    }

    /* Primary Clinical Buttons */
    div.stButton > button {
        background-color: #0D9488;
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.25rem;
        transition: background-color 0.2s ease;
    }
    div.stButton > button p {
        color: #FFFFFF !important;
    }
    div.stButton > button:hover {
        background-color: #0F766E;
        color: #FFFFFF !important;
    }

    /* Telemetry Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #0D9488;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stMetricLabel"] p {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* Bordered Clinical Panels */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #CBD5E1;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }

    /* Main Mode Navigation Tabs */
    button[data-baseweb="tab"] p {
        font-weight: 600 !important;
        color: #475569 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #0D9488 !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #0D9488 !important;
    }

    /* Risk Status Pill Badges */
    .risk-badge-high {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .risk-badge-low {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hospital Header & Notice
st.markdown(
    """
    <div class="hospital-banner">
        <h1>🏥 Stroke-AI Clinical Decision Support</h1>
        <p>Explainable Multimodal Neuro-Risk Stratification • Structured EHR Vitals + Brain MRI Slice Fusion</p>
    </div>
    """,
    unsafe_allow_html=True,
)

demo, research = st.tabs(["🩺 Patient Risk Assessment (Demo Mode)", "📊 Validation & Research Benchmarks"])


@st.cache_resource
def _load(path: str):
    return StrokePredictor.load(path)


with demo:
    if not (ART / "model" / "scorer.json").exists():
        st.info("No trained model found. Run `python -m stroke_ai.pipeline` first.")
    else:
        pred = _load(str(ART / "model"))

        # Quick Patient Profile Presets for Live Viva Demonstration
        st.markdown("#### Quick Patient Profile Presets")
        p1, p2, p3 = st.columns([1, 1, 2])
        if p1.button("Load Low-Risk Patient", use_container_width=True):
            st.session_state.update({
                "age": 38, "gender": "Female", "hyp": False, "hd": False,
                "glu": 88.0, "bmi_known": True, "bmi": 22.5, "smoke": "never smoked"
            })
        if p2.button("Load High-Risk Patient", use_container_width=True):
            st.session_state.update({
                "age": 74, "gender": "Male", "hyp": True, "hd": True,
                "glu": 215.0, "bmi_known": True, "bmi": 33.4, "smoke": "smokes"
            })
        p3.caption("Click a preset button to auto-fill clinical vitals during a live demonstration, or adjust fields directly below.")

        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            with st.container(border=True):
                st.markdown("### 📋 Patient Clinical Vitals")

                r1a, r1b = st.columns(2)
                age = r1a.number_input("Age (years)", 0, 120, st.session_state.get("age", 62))
                glu = r1b.number_input("Avg. Glucose (mg/dL)", 30.0, 500.0, st.session_state.get("glu", 105.0))

                # Always-visible horizontal options instead of dropdowns
                gender = st.radio(
                    "Biological Sex",
                    ["Male", "Female"],
                    index=0 if st.session_state.get("gender", "Male") == "Male" else 1,
                    horizontal=True,
                )

                smoke_opts = ["never smoked", "formerly smoked", "smokes", "Unknown"]
                smoke_default = st.session_state.get("smoke", "never smoked")
                smoke = st.radio(
                    "Smoking Status",
                    smoke_opts,
                    index=smoke_opts.index(smoke_default),
                    horizontal=True,
                )

                r2a, r2b = st.columns(2)
                hyp = r2a.checkbox("Hypertension History", value=st.session_state.get("hyp", False))
                hd = r2b.checkbox("Heart Disease History", value=st.session_state.get("hd", False))

                bmi_known = st.checkbox("BMI Recorded", value=st.session_state.get("bmi_known", True))
                bmi = st.number_input("Body Mass Index (BMI)", 10.0, 80.0, st.session_state.get("bmi", 27.0)) if bmi_known else None

        with c2:
            with st.container(border=True):
                st.markdown("### 🧠 Neuroimaging Input (Brain MRI)")
                up = st.file_uploader("Upload Axial MRI Slice (PNG / JPG)", type=["png", "jpg", "jpeg"])
                if up is not None:
                    st.image(up, caption=f"Uploaded Scan: {up.name}", width=260)
                else:
                    st.info("Upload a brain MRI slice (PNG or JPG) to enable multimodal risk scoring.")

        run_btn = st.button("🔬 Run Multimodal Stroke Risk Assessment", use_container_width=True)

        if run_btn:
            if up is None:
                st.warning("Please upload an MRI slice before running the assessment.")
            else:
                img = load_image(up, pred.img_size)
                res = pred.predict(
                    dict(
                        age=age,
                        gender=gender,
                        hypertension=int(hyp),
                        heart_disease=int(hd),
                        avg_glucose_level=glu,
                        bmi=bmi,
                        smoking_status=smoke,
                    ),
                    img,
                )

                st.markdown("---")
                st.markdown("### 🩺 Diagnostic Support Summary")

                is_high = res["risk_probability"] >= res["decision_threshold"]
                badge_html = (
                    '<span class="risk-badge-high">⚠️ ELEVATED STROKE RISK</span>'
                    if is_high
                    else '<span class="risk-badge-low">✅ LOW / BASELINE RISK</span>'
                )
                st.markdown(
                    f"{badge_html} &nbsp; Decision threshold: **{res['decision_threshold']:.2f}** (calibrated on validation cohort)",
                    unsafe_allow_html=True,
                )
                st.write("")

                a, b, c = st.columns(3)
                a.metric("Calibrated Risk Probability", f"{res['risk_probability']:.1%}")
                b.metric("Clinical Triage Tier", res["assessment"])
                c.metric("Model Uncertainty (MC-Dropout)", res["uncertainty"]["level"], help=res["uncertainty"]["note"])

                st.write("")
                exp1, exp2 = st.columns(2, gap="large")
                with exp1:
                    if "top_clinical_factors" in res:
                        with st.container(border=True):
                            st.markdown("#### 📊 Clinical Feature Attribution (SHAP)")
                            st.bar_chart(pd.DataFrame(res["top_clinical_factors"]).set_index("feature")["contribution"])
                            st.caption("Positive bars increase predicted stroke probability; negative bars lower it.")

                with exp2:
                    if "gradcam_png_base64" in res:
                        with st.container(border=True):
                            st.markdown("#### 🎯 Neuroimaging Attention (Grad-CAM)")
                            st.image(base64.b64decode(res["gradcam_png_base64"]), width=320)
                            st.caption("Grad-CAM highlights regions driving the imaging pathway; low-risk scans may display diffuse background normalization.")

with research:
    res_dir = ART / "results"
    if not res_dir.exists():
        st.info("No results yet. Run `python -m stroke_ai.pipeline` to generate benchmark artefacts.")
    else:
        st.markdown("### 📈 Model Ablation, Calibration & Clinical Grid Benchmarks")
        for name in ["ablation_test", "paired_auc_differences", "calibration", "external_validation", "clinical_grid_val"]:
            f = res_dir / f"{name}.csv"
            if f.exists():
                with st.container(border=True):
                    st.markdown(f"#### {name.replace('_', ' ').title()}")
                    st.dataframe(pd.read_csv(f), use_container_width=True)

        st.markdown("### 🛡️ Subgroup Fairness & Stress-Test Robustness")
        sub_f = res_dir / "subgroups_test.csv"
        if sub_f.exists():
            with st.container(border=True):
                st.markdown("#### Subgroup Fairness & Performance Audit")
                st.dataframe(pd.read_csv(sub_f), use_container_width=True)

        rob = res_dir / "robustness_test.csv"
        if rob.exists():
            with st.container(border=True):
                st.markdown("#### Stress-Test Robustness (Test ROC-AUC under Perturbations)")
                r = pd.read_csv(rob)
                for pert, g in r.groupby("perturbation"):
                    st.caption(f"Perturbation: {pert}")
                    st.line_chart(g.pivot(index="level", columns="model", values="roc_auc"))

        st.markdown("### 🖼️ Evaluation Figures (ROC, Calibration, SHAP & Grad-CAM)")
        cols = st.columns(2)
        fig_names = ["roc", "calibration", "shap_clinical", "gradcam"]
        for idx, fig in enumerate(fig_names):
            f = ART / "figures" / f"{fig}.png"
            if f.exists():
                with cols[idx % 2]:
                    with st.container(border=True):
                        st.image(str(f), caption=fig.replace("_", " ").upper(), use_container_width=True)