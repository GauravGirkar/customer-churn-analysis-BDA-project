"""Churn dashboard - reads outputs/scores.parquet + outputs/metrics.json written by `python -m churn.pipeline`.

    streamlit run dashboard/app.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

OUT = Path(os.getenv("CHURN_OUTPUT_DIR", Path(__file__).resolve().parent.parent / "outputs"))

# Palette ---------------------------------------------------------------------
# Categorical slots 1-5 (fixed order, one per model) and a one-hue ramp for risk bands.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BAND_COLORS = {"Low": "#a9c9f2", "Medium": "#4f93e0", "High": "#1c4f94"}
INK, MUTED, GRID = "#0b1f3a", "#5b6472", "#e7eaf0"
FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="📉", layout="wide")


# ------------------------------------------------------------- global styling
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif; }
        .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1280px; }

        /* Hero banner */
        .hero { background: linear-gradient(130deg, #0b2a52 0%, #1c4f94 45%, #2a78d6 100%);
                border-radius: 16px; padding: 24px 30px; color: #fff;
                box-shadow: 0 10px 30px rgba(28,79,148,.25); }
        .hero h1 { color:#fff; font-size: 1.85rem; font-weight: 800; margin: 0 0 6px; letter-spacing:-.02em; }
        .hero p  { color:#d6e4f7; margin: 0; font-size: .97rem; max-width: 760px; line-height:1.5; }
        .badge { display:inline-block; background: rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.28);
                 color:#fff; padding:4px 12px; border-radius:999px; font-size:.78rem; font-weight:500;
                 margin-right:7px; margin-top:12px; backdrop-filter: blur(4px); }

        /* Metric cards */
        [data-testid="stMetric"] { background:#fff; border:1px solid #e7eaf0; border-radius:14px;
                 padding:16px 18px 14px; box-shadow:0 1px 3px rgba(16,24,40,.05); }
        [data-testid="stMetricLabel"] p { color:#5b6472; font-weight:600; font-size:.82rem; }
        [data-testid="stMetricValue"] { color:#0b2a52; font-weight:800; font-size:1.7rem; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid #e7eaf0; }
        .stTabs [data-baseweb="tab"] { padding:9px 18px; border-radius:9px 9px 0 0; font-weight:600;
                 color:#5b6472; font-size:.92rem; }
        .stTabs [aria-selected="true"] { background:#eef4fc; color:#1c4f94; }

        /* Section headings */
        h3 { color:#0b2a52; font-weight:700; letter-spacing:-.01em; }
        .stDataFrame { border-radius:10px; }

        /* Primary buttons */
        .stButton button[kind="primary"] { background:#1c4f94; border:0; font-weight:600; border-radius:10px; }
        .stButton button[kind="primary"]:hover { background:#163f77; }

        /* Tour dialog step dots */
        .dot { height:8px; width:8px; border-radius:50%; display:inline-block; margin-right:6px; background:#d7dee8; }
        .dot.on { background:#1c4f94; width:22px; border-radius:999px; }
        .tour-kicker { color:#2a78d6; font-weight:700; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load(mtime):
    scores = pd.read_parquet(OUT / "scores.parquet")
    metrics = json.loads((OUT / "metrics.json").read_text())
    return scores, metrics


if not (OUT / "scores.parquet").exists():
    st.error("No pipeline output found. Run `python -m churn.generate_data` then `python -m churn.pipeline` first.")
    st.stop()

scores, metrics = load((OUT / "scores.parquet").stat().st_mtime)
models = metrics["models"]
best = metrics["best_model"]
model_order = list(models)
color_of = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(model_order)}


def style(fig, height=340, **kw):
    kw.setdefault("legend", dict(orientation="h", y=-0.2))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=12, family=FONT),
                      hoverlabel=dict(font_size=12, font_family=FONT), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ------------------------------------------------------------------ guided tour
TOUR_STEPS = [
    {"kicker": "Welcome", "title": "Customer Churn Intelligence",
     "body": """This dashboard turns **real telecom customer data** into an early-warning system for churn.

It is powered by a full **Big Data Analytics pipeline**: five relational tables → **Spark SQL** feature
engineering → **five machine-learning models** → a churn probability for every customer.

Use the five tabs along the top to move from the big picture down to a single customer. This quick tour
explains what each one does — it takes about 30 seconds."""},
    {"kicker": "Tab 1 of 5", "title": "📊 Overview — the big picture",
     "body": """Start here. The cards at the top show **how many customers are at risk** and the **monthly revenue**
that risk represents.

Below them you can see how customers split across **Low / Medium / High** risk bands, the full
**distribution of churn probability**, and a side-by-side of **predicted vs. actual** churn broken down by
contract, plan, internet service, payment method or region."""},
    {"kicker": "Tab 2 of 5", "title": "👥 Customers — act on individuals",
     "body": """This is the operational tab. **Filter** the customer base by risk band, contract, region or plan,
then read each customer's churn probability alongside **plain-language risk factors** ("2 late payments",
"usage declining").

**Download** the filtered list as CSV for a retention campaign, or use **Customer lookup** to pull up one
customer and see what every model predicts for them."""},
    {"kicker": "Tab 3 of 5", "title": "🎯 Model performance — how good are the predictions?",
     "body": """Compare all five models on the **held-out test set** they never trained on: ROC-AUC, PR-AUC,
precision, recall and F1.

Explore the **ROC and precision–recall curves**, inspect any model's **confusion matrix**, and read the
**cumulative-gains curve** — e.g. *"contact the riskiest 10% of customers and you reach X% of everyone who
will actually churn."*"""},
    {"kicker": "Tab 4 of 5", "title": "🔍 Churn drivers — the why",
     "body": """See **which signals each model relies on most**, then check real churn rates broken down by behaviour:
late payments, support tickets, complaints, service calls and **tenure**.

This is where the story behind the score lives — useful for deciding *what* to fix, not just *who* to call."""},
    {"kicker": "Tab 5 of 5", "title": "⚙️ Pipeline — under the hood",
     "body": """A transparent view of **how the numbers were produced**: the raw tables and their row counts, the
train / validation / test split, which model was promoted to production, and the total pipeline run time.

Everything is reproducible — the README shows how to rebuild the data and re-run the pipeline. **Enjoy!**"""},
]


@st.dialog("Guided tour", width="large")
def show_tour():
    i = st.session_state.tour_step
    step = TOUR_STEPS[i]
    st.markdown(f"<span class='tour-kicker'>{step['kicker']}</span>", unsafe_allow_html=True)
    st.markdown(f"### {step['title']}")
    st.markdown(step["body"])
    st.markdown(
        "<div style='margin:14px 0 4px'>"
        + "".join(f"<span class='dot {'on' if j == i else ''}'></span>" for j in range(len(TOUR_STEPS)))
        + "</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    if c1.button("Skip", use_container_width=True):
        st.session_state.tour_step = 0
        st.session_state.show_tour = False
        st.rerun()
    if i > 0 and c2.button("← Back", use_container_width=True):
        st.session_state.tour_step -= 1
        st.rerun()
    last = i == len(TOUR_STEPS) - 1
    if c3.button("Finish ✓" if last else "Next →", type="primary", use_container_width=True):
        if last:
            st.session_state.tour_step = 0
            st.session_state.show_tour = False
        else:
            st.session_state.tour_step += 1
        st.rerun()


# auto-open once per browser session, and whenever the user clicks "Take the tour"
if "tour_step" not in st.session_state:
    st.session_state.tour_step = 0
if "show_tour" not in st.session_state:
    st.session_state.show_tour = True  # first load


# ---------------------------------------------------------------- header
hero, action = st.columns([5, 1])
with hero:
    st.markdown(
        f"""
        <div class="hero">
          <h1>Customer Churn Intelligence</h1>
          <p>Predicting which telecom customers are about to leave — from real customer data through a
          Spark big-data pipeline to a churn probability for every account.</p>
          <span class="badge">🗄️ {metrics['n_customers']:,} real customers</span>
          <span class="badge">🏆 Live model: {models[best]['display_name']}</span>
          <span class="badge">📅 Window ending {metrics['snapshot_date']}</span>
          <span class="badge">🔄 Run {metrics['generated_at'][:10]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with action:
    st.write("")
    st.write("")
    if st.button("🧭 Take the tour", type="primary", use_container_width=True):
        st.session_state.tour_step = 0
        st.session_state.show_tour = True
        st.rerun()

if st.session_state.show_tour:
    show_tour()

st.write("")
tab_over, tab_cust, tab_perf, tab_drv, tab_pipe = st.tabs(
    ["📊  Overview", "👥  Customers", "🎯  Model performance", "🔍  Churn drivers", "⚙️  Pipeline"])

# -------------------------------------------------------------- overview
with tab_over:
    high = scores[scores.risk_band == "High"]
    at_risk_rev = (high.monthly_charges.sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", f"{len(scores):,}")
    c2.metric("Actual churn rate", pct(scores.actual_churn.mean()))
    c3.metric("High-risk customers", f"{len(high):,}", help=f"Churn probability ≥ {metrics['band_cutoffs']['high']:.2f} "
                                                            f"({pct(len(high) / len(scores))} of the base)")
    c4.metric("Monthly revenue at risk", f"${at_risk_rev:,.0f}", help="Sum of monthly charges of high-risk customers")

    st.write("")
    left, right = st.columns(2)
    with left:
        st.subheader("Customers by risk band")
        band = scores.groupby("risk_band").agg(n=("customer_id", "size"), actual=("actual_churn", "mean")).reindex(
            ["Low", "Medium", "High"])
        fig = go.Figure(go.Bar(
            x=band.index, y=band.n, marker_color=[BAND_COLORS[b] for b in band.index], marker_cornerradius=4,
            customdata=band.actual, hovertemplate="%{x}: %{y:,} customers<br>actual churn %{customdata:.1%}<extra></extra>"))
        st.plotly_chart(style(fig, yaxis_title="customers"), use_container_width=True)
        st.caption("Actual churn by band: " + " · ".join(f"{b} {pct(r)}" for b, r in band.actual.items()))
    with right:
        st.subheader("Distribution of churn probability")
        fig = go.Figure(go.Histogram(x=scores.churn_probability, xbins=dict(start=0, end=1, size=0.025),
                                     marker_color=MODEL_COLORS[0], marker_line=dict(color="white", width=1),
                                     hovertemplate="p %{x:.2f}: %{y:,}<extra></extra>"))
        for cut in metrics["band_cutoffs"].values():
            fig.add_vline(x=cut, line=dict(color=MUTED, width=1, dash="dot"))
        st.plotly_chart(style(fig, xaxis_title="predicted churn probability", yaxis_title="customers"),
                        use_container_width=True)

    st.subheader("Predicted vs actual churn by segment")
    seg_col = st.radio("Segment by", ["contract", "plan", "internet_service", "payment_method", "region"],
                       horizontal=True, label_visibility="collapsed")
    seg = scores.groupby(seg_col).agg(pred=("churn_probability", "mean"), actual=("actual_churn", "mean"),
                                      n=("customer_id", "size")).sort_values("actual", ascending=False)
    fig = go.Figure()
    fig.add_bar(name="Actual churn", x=seg.index, y=seg.actual, marker_color=MODEL_COLORS[0], marker_cornerradius=4,
                customdata=seg.n, hovertemplate="%{x}<br>actual %{y:.1%} (n=%{customdata:,})<extra></extra>")
    fig.add_bar(name="Mean predicted probability", x=seg.index, y=seg.pred, marker_color=MODEL_COLORS[1],
                marker_cornerradius=4, hovertemplate="%{x}<br>predicted %{y:.1%}<extra></extra>")
    st.plotly_chart(style(fig, barmode="group", bargap=0.35, yaxis_tickformat=".0%"), use_container_width=True)

# ------------------------------------------------------------- customers
with tab_cust:
    st.subheader("Churn probability per customer")
    f1, f2, f3, f4 = st.columns(4)
    bands = f1.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
    contracts = f2.multiselect("Contract", sorted(scores.contract.unique()))
    regions = f3.multiselect("Region", sorted(scores.region.unique()))
    plans = f4.multiselect("Plan", sorted(scores.plan.unique()))
    g1, g2 = st.columns([3, 2])
    prange = g1.slider("Churn probability", 0.0, 1.0, (0.0, 1.0), 0.01)
    heldout = g2.checkbox("Held-out customers only (test split)", value=False,
                          help="The model never saw these customers during training, so their scores are an honest "
                               "preview of production behaviour. Scores for training customers are optimistic.")

    view = scores
    if bands:
        view = view[view.risk_band.isin(bands)]
    if contracts:
        view = view[view.contract.isin(contracts)]
    if regions:
        view = view[view.region.isin(regions)]
    if plans:
        view = view[view.plan.isin(plans)]
    view = view[view.churn_probability.between(*prange)]
    if heldout:
        view = view[view.split == "test"]
    view = view.sort_values("churn_probability", ascending=False)

    st.caption(f"{len(view):,} customers match · avg monthly charge ${view.monthly_charges.mean():,.2f}"
               if len(view) else "No customers match the filters.")
    cols = ["customer_id", "churn_probability", "risk_band", "risk_factors", "contract", "plan", "tenure_months",
            "monthly_charges", "late_payments", "ticket_count", "call_count", "minutes_trend", "region"]
    st.dataframe(
        view[cols].head(1000), use_container_width=True, hide_index=True, height=420,
        column_config={
            "churn_probability": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1,
                                                                 format="%.2f"),
            "monthly_charges": st.column_config.NumberColumn("Monthly $", format="$%.2f"),
            "tenure_months": st.column_config.NumberColumn("Tenure (mo)", format="%d"),
            "minutes_trend": st.column_config.NumberColumn("Usage trend", format="%.2f×",
                                                           help="recent 3 months ÷ prior 3 months"),
        })
    d1, d2 = st.columns([1, 3])
    d1.download_button("⬇️  Download filtered list (CSV)", view[cols].to_csv(index=False).encode(),
                       "churn_scores.csv", "text/csv")
    d2.caption("Table shows the top 1,000 by probability; the download contains all matches.")

    st.divider()
    st.subheader("Customer lookup")
    q = st.text_input("Customer ID", placeholder=f"e.g. {scores.customer_id.iloc[0]}")
    if q:
        row = scores[scores.customer_id.str.upper() == q.strip().upper()]
        if row.empty:
            st.warning("Customer not found.")
        else:
            r = row.iloc[0]
            a, b, c = st.columns(3)
            a.metric("Churn probability", pct(r.churn_probability), help=f"{r.risk_band} risk band")
            b.metric("Tenure", f"{int(r.tenure_months)} months", help=f"{r.contract} contract")
            c.metric("Monthly charge", f"${r.monthly_charges:,.2f}", help=f"{r.plan} plan")
            st.markdown(f"**Why:** {r.risk_factors}")
            comp = pd.DataFrame({"Model": [models[m]["display_name"] for m in model_order],
                                 "Probability": [r[f"prob_{m}"] for m in model_order]})
            fig = go.Figure(go.Bar(y=comp.Model[::-1], x=comp.Probability[::-1], orientation="h",
                                   marker_color=[color_of[m] for m in model_order][::-1], marker_cornerradius=4,
                                   hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            st.plotly_chart(style(fig, height=230, xaxis_tickformat=".0%", xaxis_range=[0, 1]),
                            use_container_width=True)
            st.caption(f"Actual outcome in the data: {'churned' if r.actual_churn else 'stayed'} "
                       f"(split: {r.split}).")

# ------------------------------------------------------- model performance
with tab_perf:
    st.subheader("Model comparison on the held-out test split")
    rows = []
    for m in model_order:
        e, t = models[m], models[m]["at_tuned"]
        rows.append({"Model": e["display_name"], "Engine": e["engine"], "ROC-AUC": e["roc_auc"], "PR-AUC": e["pr_auc"],
                     "Precision": t["precision"], "Recall": t["recall"], "F1": t["f1"],
                     "Lift @ top 10%": e["top_decile_lift"], "Train time (s)": e["train_seconds"]})
    cmp_df = pd.DataFrame(rows)
    st.dataframe(cmp_df, hide_index=True, use_container_width=True, column_config={
        "ROC-AUC": st.column_config.NumberColumn(format="%.4f"), "PR-AUC": st.column_config.NumberColumn(format="%.4f"),
        "Precision": st.column_config.NumberColumn(format="%.3f"), "Recall": st.column_config.NumberColumn(format="%.3f"),
        "F1": st.column_config.NumberColumn(format="%.3f"),
        "Lift @ top 10%": st.column_config.NumberColumn(format="%.2f×")})
    st.caption("Precision / recall / F1 use each model's decision threshold tuned for max F1 on the validation split. "
               f"Production scores come from **{models[best]['display_name']}**, chosen by validation ROC-AUC.")

    st.write("")
    cl, cr = st.columns(2)
    with cl:
        st.subheader("ROC curves")
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRID, dash="dot", width=1), showlegend=False,
                        hoverinfo="skip")
        for m in model_order:
            xy = np.array(models[m]["roc_curve"])
            fig.add_scatter(x=xy[:, 0], y=xy[:, 1], mode="lines", name=f"{models[m]['display_name']} ({models[m]['roc_auc']:.3f})",
                            line=dict(color=color_of[m], width=2),
                            hovertemplate="FPR %{x:.2f}, TPR %{y:.2f}<extra>" + models[m]["display_name"] + "</extra>")
        st.plotly_chart(style(fig, height=380, xaxis_title="false positive rate", yaxis_title="true positive rate",
                              legend=dict(orientation="h", y=-0.3)), use_container_width=True)
    with cr:
        st.subheader("Precision–recall curves")
        fig = go.Figure()
        for m in model_order:
            xy = np.array(models[m]["pr_curve"])
            fig.add_scatter(x=xy[:, 0], y=xy[:, 1], mode="lines", name=f"{models[m]['display_name']} ({models[m]['pr_auc']:.3f})",
                            line=dict(color=color_of[m], width=2),
                            hovertemplate="recall %{x:.2f}, precision %{y:.2f}<extra>" + models[m]["display_name"] + "</extra>")
        fig.add_hline(y=metrics["churn_rate"], line=dict(color=GRID, dash="dot", width=1))
        st.plotly_chart(style(fig, height=380, xaxis_title="recall", yaxis_title="precision",
                              legend=dict(orientation="h", y=-0.3)), use_container_width=True)

    st.divider()
    st.subheader("Inspect one model")
    sel = st.selectbox("Model", model_order, index=model_order.index(best), format_func=lambda m: models[m]["display_name"])
    test = scores[scores.split == "test"]
    e = models[sel]
    k1, k2 = st.columns([1, 1])
    with k1:
        st.markdown("**Confusion matrix** (test split, tuned threshold "
                    f"{e['at_tuned']['threshold']:.2f})")
        cm = e["at_tuned"]["confusion"]
        z = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted stay", "Predicted churn"], y=["Actually stayed", "Actually churned"],
                                   colorscale=[[0, "#eaf2fc"], [1, "#1c4f94"]], showscale=False, xgap=2, ygap=2,
                                   text=z, texttemplate="%{text:,}", textfont=dict(size=16),
                                   hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, height=300), use_container_width=True)
        d = e["at_default_0.5"]
        st.caption(f"At the default 0.5 cut-off: precision {d['precision']:.3f}, recall {d['recall']:.3f}, F1 {d['f1']:.3f} — "
                   "churn is the minority class, so 0.5 misses too many churners.")
    with k2:
        st.markdown("**Cumulative gains** — share of churners found by contacting the riskiest customers first")
        order = np.argsort(-test[f"prob_{sel}"].to_numpy())
        y = test.actual_churn.to_numpy()[order]
        gx = np.arange(1, len(y) + 1) / len(y)
        gy = np.cumsum(y) / y.sum()
        step = max(1, len(y) // 200)
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRID, dash="dot", width=1), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=gx[::step], y=gy[::step], mode="lines", line=dict(color=color_of[sel], width=2),
                        name=e["display_name"], showlegend=False,
                        hovertemplate="contact top %{x:.0%} → reach %{y:.0%} of churners<extra></extra>")
        st.plotly_chart(style(fig, height=300, xaxis_title="share of customers contacted", yaxis_title="share of churners captured",
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%"), use_container_width=True)
        i10 = int(0.1 * len(y))
        st.caption(f"Contacting the top 10% reaches {gy[i10]:.0%} of churners — a {e['top_decile_lift']:.2f}× lift over random.")

# -------------------------------------------------------------- drivers
with tab_drv:
    st.subheader("What drives churn")
    sel2 = st.selectbox("Model", model_order, index=model_order.index(best), key="imp_model",
                        format_func=lambda m: models[m]["display_name"])
    imp = pd.DataFrame(models[sel2]["importance"]).sort_values("importance")
    imp["feature"] = imp["feature"].str.replace("_ohe_", " = ", regex=False).str.replace("_", " ")
    fig = go.Figure(go.Bar(y=imp.feature, x=imp.importance, orientation="h", marker_color=color_of[sel2],
                           marker_cornerradius=3, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    st.plotly_chart(style(fig, height=440, xaxis_title="relative importance (normalised)", xaxis_tickformat=".0%"),
                    use_container_width=True)
    st.caption("Tree models: impurity/gain-based importance. Logistic regression: |standardised coefficient|. "
               "Importance shows what the model uses, not a causal effect.")

    st.divider()
    st.subheader("Churn rate by behaviour")
    b1, b2 = st.columns(2)
    with b1:
        feat = st.selectbox("Feature", ["late_payments", "ticket_count", "complaint_count", "call_count",
                                        "unresolved_calls", "cancellation_inquiries"], key="bf")
        grp = scores.assign(v=scores[feat].clip(upper=5)).groupby("v").agg(rate=("actual_churn", "mean"),
                                                                           n=("customer_id", "size"))
        fig = go.Figure(go.Bar(x=grp.index.astype(int).astype(str).str.replace("5", "5+"), y=grp.rate,
                               marker_color=MODEL_COLORS[0], marker_cornerradius=4, customdata=grp.n,
                               hovertemplate="%{x}: %{y:.1%} churn (n=%{customdata:,})<extra></extra>"))
        st.plotly_chart(style(fig, height=300, xaxis_title=feat.replace("_", " "), yaxis_tickformat=".0%",
                              yaxis_title="actual churn rate"), use_container_width=True)
    with b2:
        tb = pd.cut(scores.tenure_months, [-1, 6, 12, 24, 36, 48, 100], labels=["0–6", "7–12", "13–24", "25–36", "37–48", "49+"])
        grp = scores.groupby(tb, observed=True).agg(rate=("actual_churn", "mean"), n=("customer_id", "size"))
        st.markdown("**Tenure (months)**")
        fig = go.Figure(go.Bar(x=grp.index.astype(str), y=grp.rate, marker_color=MODEL_COLORS[1], marker_cornerradius=4,
                               customdata=grp.n, hovertemplate="%{x} months: %{y:.1%} churn (n=%{customdata:,})<extra></extra>"))
        st.plotly_chart(style(fig, height=300, xaxis_title="tenure (months)", yaxis_tickformat=".0%",
                              yaxis_title="actual churn rate"), use_container_width=True)

# -------------------------------------------------------------- pipeline
with tab_pipe:
    st.subheader("How the numbers were produced")
    st.markdown(f"""
```
Real customers ({metrics['raw_source']})  →  reconstructed into 5 relational tables
   customers · usage · payments · tickets · service_calls
        │  Spark read (HDFS / cloud storage / local)
        ▼
Spark SQL feature engineering  →  {metrics['n_features']} features per customer
        │  deterministic 70 / 15 / 15 train / validation / test split
        ▼
Spark MLlib: Logistic Regression · Random Forest · GBT      Single-node: LightGBM · XGBoost
        │  evaluate on test: ROC-AUC · PR-AUC · precision · recall · F1 · lift
        ▼
scores.parquet + metrics.json  →  this dashboard
```
""")
    rows = pd.DataFrame({"Table": list(metrics["table_rows"]), "Rows": list(metrics["table_rows"].values())})
    p1, p2 = st.columns([1, 2])
    p1.dataframe(rows, hide_index=True, use_container_width=True, column_config={"Rows": st.column_config.NumberColumn(format="%d")})
    with p2:
        st.markdown(f"""
- **Customers:** {metrics['n_customers']:,} (real IBM/Kaggle Telco Customer Churn) · **churn rate:** {pct(metrics['churn_rate'])}
- **Split:** {', '.join(f'{k} {v:,}' for k, v in metrics['split_sizes'].items())}
- **Hyper-parameter search:** {'3-fold CV grid search' if metrics['tuned'] else 'fixed defaults (run with --tune for CV)'}
- **Pipeline wall time:** {metrics['pipeline_seconds']} s
- **Risk bands:** High ≥ {metrics['band_cutoffs']['high']:.2f}, Medium ≥ {metrics['band_cutoffs']['medium']:.2f}
""")
    st.caption("Customer attributes and the churn label are real; the behavioural history (usage, payments, "
               "tickets, calls) is reconstructed per real customer to feed the Spark feature pipeline.")
