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
MODEL_COLORS = ["#1c4f94", "#c05621", "#2f855a", "#b7791f", "#805ad5"]
BAND_COLORS = {"Low": "#b9cce6", "Medium": "#5a86bd", "High": "#1c4f94"}
INK, MUTED, GRID = "#14171c", "#5b6472", "#e3e6ec"
BODY = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace"

st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")


# ------------------------------------------------------------- global styling
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

        html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif; color:#14171c; }
        .block-container { padding-top: 1.3rem; padding-bottom: 2.6rem; max-width: 1200px; }

        /* Masthead (flat, editorial) */
        .masthead { border-bottom: 3px solid #14171c; padding-bottom: 14px; }
        .masthead .eyebrow { font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .24em;
                 text-transform: uppercase; color: #1c4f94; margin-bottom: 4px; }
        .masthead h1 { font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 2.5rem; line-height: 1.04;
                 letter-spacing: -.015em; margin: 2px 0 .4rem; color: #14171c; }
        .masthead .lede { font-size: 1.02rem; color: #434a56; max-width: 72ch; line-height: 1.5; }
        .masthead .meta { font-family: 'JetBrains Mono', monospace; font-size: .74rem; color: #5b6472;
                 margin-top: 12px; }
        .masthead .meta b { color: #14171c; font-weight: 700; }
        .masthead .meta span { color: #c3c8d2; margin: 0 10px; }

        /* Section headings use the serif; captions stay sans */
        h2, h3 { font-family: 'Fraunces', Georgia, serif !important; font-weight: 600 !important; color: #14171c;
                 letter-spacing: -.01em; }

        /* Metric cells: flat, squared, mono numerals, rule on top */
        [data-testid="stMetric"] { background: #fff; border: 1px solid #e3e6ec; border-top: 3px solid #1c4f94;
                 border-radius: 0; padding: 14px 16px 12px; }
        [data-testid="stMetricLabel"] p { font-family: 'JetBrains Mono', monospace; text-transform: uppercase;
                 letter-spacing: .1em; font-size: .66rem; color: #5b6472; font-weight: 500; }
        [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.65rem;
                 color: #14171c; }

        /* Tabs: mono, uppercase, underline the active one */
        .stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid #e3e6ec; }
        .stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; text-transform: uppercase;
                 letter-spacing: .07em; font-size: .73rem; font-weight: 500; color: #5b6472; padding: 11px 18px;
                 border-radius: 0; }
        .stTabs [aria-selected="true"] { color: #14171c; border-bottom: 2px solid #1c4f94; background: transparent; }

        /* Buttons: squared, restrained */
        .stButton button { border-radius: 2px; font-weight: 600; }
        .stButton button[kind="primary"] { background: #1c4f94; border: 0; }
        .stButton button[kind="primary"]:hover { background: #163f77; }

        /* Tour chrome */
        .tour-eyebrow { font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: .2em;
                 font-size: .68rem; color: #1c4f94; font-weight: 700; }
        .dot { height:6px; width:6px; display:inline-block; margin-right:5px; background:#dde1e8; }
        .dot.on { background:#1c4f94; width:18px; }
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
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=12, family=BODY),
                      hoverlabel=dict(font_size=12, font_family=BODY), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(family=MONO, size=11))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(family=MONO, size=11))
    return fig


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ------------------------------------------------------------------ guided tour
# The tour walks through the five SOURCE TABLES - what each stores and the features it feeds -
# then how they join into one scored row per customer.
def tour_steps():
    tr = metrics["table_rows"]
    nf = metrics["n_features"]
    return [
        {"eyebrow": "The pipeline", "title": "From five tables to a churn score",
         "body": f"""This dashboard is the end of a real **Big Data Analytics pipeline**. The raw material is five
relational tables; Spark SQL turns them into one row per customer, five models score that row, and the result is
what you see here.

```
customers · usage · payments · tickets · service_calls      five raw tables
          |   Spark SQL  -  aggregate + join per customer
          v
one feature row per customer   ({nf} features)
          |   Logistic Regression · Random Forest · GBT · LightGBM · XGBoost
          v
churn probability + risk band for every customer
```

The next five steps open up **each source table**: what it records and the signals it produces."""},

        {"eyebrow": "Source table 1 of 5", "title": "customers — who the customer is",
         "body": f"""The **real IBM / Kaggle Telco Customer Churn** data — **{tr['customers']:,} rows**, one per customer.

```
customer_id · contract · internet_service · payment_method
monthly_charges · tenure · senior_citizen · add-ons · churned
```

These real fields become the **profile features**: `tenure_months`, `monthly_charges`, `total_charges`, `age`,
and the categoricals `contract / plan / internet_service / payment_method`. The `churned` column is the **real
outcome** every model is trained to predict."""},

        {"eyebrow": "Source table 2 of 5", "title": "usage — how much they use the service",
         "body": f"""A **six-month behavioural time-series** — six rows per customer, **{tr['usage']:,} rows** in all.

```
customer_id · month · call_minutes · data_gb · sms_count · app_logins
```

Spark SQL averages each metric and compares the **last 3 months against the prior 3**, yielding usage **levels**
(`avg_call_minutes`, `avg_data_gb`, …) and **trends** (`minutes_trend`, `data_trend`, `logins_trend`). A usage
curve that is fading is one of the earliest churn signals."""},

        {"eyebrow": "Source table 3 of 5", "title": "payments — how reliably they pay",
         "body": f"""One row per monthly bill — **{tr['payments']:,} rows**.

```
customer_id · due_date · paid_date · amount · days_late
```

Aggregated into `late_payments`, `late_payment_ratio`, `avg_days_late`, `max_days_late` and
`late_payments_recent` (the last 3 months). Repeated or worsening late payment is a strong dissatisfaction marker."""},

        {"eyebrow": "Source table 4 of 5", "title": "tickets — what they complain about",
         "body": f"""Support tickets raised by some customers — **{tr['tickets']:,} rows**.

```
customer_id · category · is_complaint · escalated · resolution_hours · satisfaction
```

These roll up to `ticket_count`, `complaint_count`, `escalated_count`, `billing_tickets`, `network_tickets`,
`cancellation_inquiries`, `avg_resolution_hours`, `avg_satisfaction` and `tickets_last_30d`. A **cancellation
inquiry** ticket is about as loud as a churn signal gets."""},

        {"eyebrow": "Source table 5 of 5", "title": "service_calls — how those calls went",
         "body": f"""Call-centre interactions — **{tr['service_calls']:,} rows**.

```
customer_id · call_date · duration_min · resolved · sentiment
```

These become `call_count`, `avg_call_duration`, `unresolved_calls`, `avg_sentiment` and `calls_last_30d`.
Unresolved calls and negative sentiment push a customer's risk up sharply."""},

        {"eyebrow": "Putting it together", "title": "Features, models, and the five tabs",
         "body": f"""Spark SQL **joins all five tables into one {nf}-feature row per customer**, split deterministically
**70 / 15 / 15** into train / validation / test. Five models train; the one with the best **validation** ROC-AUC is
promoted to production and scores every customer.

Where each thing lives in this dashboard:

- **Overview** — the portfolio: risk bands, probability spread, predicted vs. actual by segment
- **Customers** — per-customer scores, plain-language risk factors, CSV export, single-customer lookup
- **Model performance** — ROC / PR curves, confusion matrix, cumulative-gains curve
- **Churn drivers** — which features each model leans on, and churn rate by behaviour
- **Pipeline** — the row counts, splits and run provenance behind every number

That is the whole pipeline — open any tab to dig in."""},
    ]


@st.dialog("How this works", width="large")
def show_tour():
    steps = tour_steps()
    i = st.session_state.tour_step
    step = steps[i]
    st.markdown(f"<span class='tour-eyebrow'>{step['eyebrow']}</span>", unsafe_allow_html=True)
    st.markdown(f"### {step['title']}")
    st.markdown(step["body"])
    st.markdown(
        "<div style='margin:16px 0 6px'>"
        + "".join(f"<span class='dot {'on' if j == i else ''}'></span>" for j in range(len(steps)))
        + f"<span style='font-family:{MONO};font-size:.7rem;color:#5b6472;margin-left:8px'>{i + 1} / {len(steps)}</span>"
        + "</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    if c1.button("Skip", use_container_width=True):
        st.session_state.update(tour_step=0, show_tour=False)
        st.rerun()
    if i > 0 and c2.button("Back", use_container_width=True):
        st.session_state.tour_step -= 1
        st.rerun()
    last = i == len(steps) - 1
    if c3.button("Finish" if last else "Next", type="primary", use_container_width=True):
        st.session_state.tour_step = 0 if last else i + 1
        if last:
            st.session_state.show_tour = False
        st.rerun()


if "tour_step" not in st.session_state:
    st.session_state.tour_step = 0
if "show_tour" not in st.session_state:
    st.session_state.show_tour = True  # auto-open once per session


# ---------------------------------------------------------------- masthead
head, action = st.columns([5, 1])
with head:
    st.markdown(
        f"""
        <div class="masthead">
          <div class="eyebrow">Telecom retention analytics</div>
          <h1>Customer Churn Intelligence</h1>
          <div class="lede">Predicting which telecom customers are about to leave &mdash; from real customer records,
          through a Spark big-data pipeline, to a churn probability for every account.</div>
          <div class="meta"><b>{metrics['n_customers']:,}</b> real customers<span>|</span>live model
          <b>{models[best]['display_name']}</b><span>|</span>window ending <b>{metrics['snapshot_date']}</b>
          <span>|</span>run <b>{metrics['generated_at'][:10]}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with action:
    st.write("")
    st.write("")
    if st.button("How this works", type="primary", use_container_width=True):
        st.session_state.update(tour_step=0, show_tour=True)
        st.rerun()

if st.session_state.show_tour:
    show_tour()

st.write("")
tab_over, tab_cust, tab_perf, tab_drv, tab_pipe = st.tabs(
    ["Overview", "Customers", "Model performance", "Churn drivers", "Pipeline"])

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
            x=band.index, y=band.n, marker_color=[BAND_COLORS[b] for b in band.index], marker_cornerradius=2,
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
    fig.add_bar(name="Actual churn", x=seg.index, y=seg.actual, marker_color=MODEL_COLORS[0], marker_cornerradius=2,
                customdata=seg.n, hovertemplate="%{x}<br>actual %{y:.1%} (n=%{customdata:,})<extra></extra>")
    fig.add_bar(name="Mean predicted probability", x=seg.index, y=seg.pred, marker_color=MODEL_COLORS[1],
                marker_cornerradius=2, hovertemplate="%{x}<br>predicted %{y:.1%}<extra></extra>")
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
    d1.download_button("Download filtered list (CSV)", view[cols].to_csv(index=False).encode(),
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
                                   marker_color=[color_of[m] for m in model_order][::-1], marker_cornerradius=2,
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
                           marker_cornerradius=2, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
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
                               marker_color=MODEL_COLORS[0], marker_cornerradius=2, customdata=grp.n,
                               hovertemplate="%{x}: %{y:.1%} churn (n=%{customdata:,})<extra></extra>"))
        st.plotly_chart(style(fig, height=300, xaxis_title=feat.replace("_", " "), yaxis_tickformat=".0%",
                              yaxis_title="actual churn rate"), use_container_width=True)
    with b2:
        tb = pd.cut(scores.tenure_months, [-1, 6, 12, 24, 36, 48, 100], labels=["0–6", "7–12", "13–24", "25–36", "37–48", "49+"])
        grp = scores.groupby(tb, observed=True).agg(rate=("actual_churn", "mean"), n=("customer_id", "size"))
        st.markdown("**Tenure (months)**")
        fig = go.Figure(go.Bar(x=grp.index.astype(str), y=grp.rate, marker_color=MODEL_COLORS[1], marker_cornerradius=2,
                               customdata=grp.n, hovertemplate="%{x} months: %{y:.1%} churn (n=%{customdata:,})<extra></extra>"))
        st.plotly_chart(style(fig, height=300, xaxis_title="tenure (months)", yaxis_tickformat=".0%",
                              yaxis_title="actual churn rate"), use_container_width=True)

# -------------------------------------------------------------- pipeline
with tab_pipe:
    st.subheader("How the numbers were produced")
    st.markdown(f"""
```
Real customers ({metrics['raw_source']})  ->  reconstructed into 5 relational tables
   customers · usage · payments · tickets · service_calls
        |  Spark read (HDFS / cloud storage / local)
        v
Spark SQL feature engineering  ->  {metrics['n_features']} features per customer
        |  deterministic 70 / 15 / 15 train / validation / test split
        v
Spark MLlib: Logistic Regression · Random Forest · GBT      Single-node: LightGBM · XGBoost
        |  evaluate on test: ROC-AUC · PR-AUC · precision · recall · F1 · lift
        v
scores.parquet + metrics.json  ->  this dashboard
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
