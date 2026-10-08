"""Hybrid real + reconstructed telecom churn data -> five relational tables.

Customer attributes and the churn label come from the **real** IBM / Kaggle
*Telco Customer Churn* dataset (7,043 real customers: tenure, contract, internet
service, payment method, monthly charges, senior-citizen flag, add-on services
and the actual Churn outcome).

No public dataset ships the behavioural *history* a churn model needs, so we
reconstruct the five relational tables (customers, 6-month usage, payments,
tickets, service_calls) for each real customer. The behaviour is driven by a
latent "dissatisfaction" that is correlated with that customer's REAL churn
outcome and their real contract / tenure / charges, so declining usage, late
payments, complaints and unresolved calls line up with people who really left.
The model never sees the label or the latent - it has to recover the signal
from the engineered behaviour, exactly as in production.

    python -m churn.generate_data                      # all 7,043 real customers
    python -m churn.generate_data --scale 100000       # bootstrap to cluster-scale volume
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import REAL_CSV, RAW_URI, SEED, SNAPSHOT_DATE

N_MONTHS = 6

# Map the real dataset's values onto this project's table vocabulary.
INTERNET_MAP = {"Fiber optic": "Fiber", "DSL": "DSL", "No": "None"}
PAYMENT_MAP = {
    "Bank transfer (automatic)": "Bank transfer (auto)",
    "Credit card (automatic)": "Credit card (auto)",
    "Electronic check": "Electronic check",
    "Mailed check": "Mailed check",
}
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_real(path=REAL_CSV):
    """Read the real Telco CSV and coerce its columns into the project's customer schema."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    # 11 brand-new (tenure 0) customers have a blank TotalCharges -> fall back to one month.
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    return df


def _scale_real(df, scale, rng):
    """Use every real customer as-is, or bootstrap-sample (with replacement) to `scale` rows
    to simulate cluster-scale volume. Every row is still a real customer record."""
    if scale is None or scale == len(df):
        out = df.reset_index(drop=True).copy()
        out["customer_id"] = out["customerID"].astype(str)
        return out
    idx = rng.integers(0, len(df), size=scale)
    out = df.iloc[idx].reset_index(drop=True).copy()
    # Unique ids for duplicated rows; tiny jitter on the continuous field so bootstrap rows differ.
    out["customer_id"] = [f"{cid}-{i:07d}" for i, cid in enumerate(out["customerID"].astype(str))]
    out["MonthlyCharges"] = (out["MonthlyCharges"] * rng.normal(1.0, 0.03, scale)).round(2).clip(15)
    return out


def generate(scale=None, seed=SEED, real_path=REAL_CSV):
    rng = np.random.default_rng(seed)
    snapshot = pd.Timestamp(SNAPSHOT_DATE)
    real = _scale_real(load_real(real_path), scale, rng)
    n = len(real)

    cust_id = real["customer_id"].to_numpy()
    churned = (real["Churn"] == "Yes").astype(int).to_numpy()
    tenure = real["tenure"].clip(0, 72).astype(int).to_numpy()
    contract = real["Contract"].to_numpy()
    internet = real["InternetService"].map(INTERNET_MAP).to_numpy()
    pay_method = real["PaymentMethod"].map(PAYMENT_MAP).to_numpy()
    monthly_charges = real["MonthlyCharges"].astype(float).round(2).to_numpy()

    # age isn't in the dataset, but the senior-citizen flag is -> draw a plausible age from it.
    senior = real["SeniorCitizen"].astype(int).to_numpy()
    age = np.where(senior == 1, rng.integers(65, 86, n), rng.integers(20, 65, n)).astype(int)

    # plan tier from the number of real add-on subscriptions (0-1 Basic, 2-4 Standard, 5-6 Premium).
    addon_count = sum((real[c] == "Yes").astype(int).to_numpy() for c in ADDON_COLS)
    plan = np.where(addon_count >= 5, "Premium", np.where(addon_count >= 2, "Standard", "Basic"))

    region = rng.choice(["North", "South", "East", "West", "Central"], n)  # not in source; no signal
    signup = snapshot - pd.to_timedelta(np.maximum(tenure, 1) * 30, unit="D")

    customers = pd.DataFrame({
        "customer_id": cust_id, "signup_date": signup.strftime("%Y-%m-%d"), "age": age,
        "region": region, "contract": contract, "plan": plan, "internet_service": internet,
        "payment_method": pay_method, "monthly_charges": monthly_charges, "churned": churned,
    })

    # ---- latent dissatisfaction: correlated with the REAL outcome (+ noise) so the reconstructed
    #      behaviour is predictive but not an oracle. The model never sees this or the label.
    m2m = contract == "Month-to-month"
    echeck = pay_method == "Electronic check"
    d = (rng.normal(0, 1, n)
         + 1.05 * (churned - churned.mean())          # real churners drift unhappy
         + 0.20 * m2m - 0.15 * (contract == "Two year")
         - 0.010 * (tenure - tenure.mean()))
    d = (d - d.mean()) / d.std()                       # standardise

    # -------------------------------------------------------------------- usage
    month_starts = pd.date_range(end=snapshot, periods=N_MONTHS, freq="MS")
    base_min = rng.lognormal(5.6, 0.5, n)
    base_gb = rng.lognormal(2.0, 0.7, n) * (internet != "None")
    base_sms = rng.lognormal(3.5, 0.8, n)
    base_login = rng.lognormal(2.3, 0.6, n)
    slope = -0.11 * d + rng.normal(0, 0.04, n)         # unhappy customers fade out
    rows = []
    for m, ms in enumerate(month_starts):
        f = np.exp(slope * m + rng.normal(0, 0.12, n))
        rows.append(pd.DataFrame({
            "customer_id": cust_id, "month": ms.strftime("%Y-%m-%d"),
            "call_minutes": np.round(base_min * f, 1),
            "data_gb": np.round(base_gb * f * rng.lognormal(0, 0.1, n), 2),
            "sms_count": (base_sms * f).astype(int),
            "app_logins": (base_login * f * rng.lognormal(0, 0.15, n)).astype(int),
        }))
    usage = pd.concat(rows, ignore_index=True)

    # ----------------------------------------------------------------- payments
    late_lambda = np.exp(-1.1 + 0.55 * d + 0.7 * echeck - 0.3 * (pay_method == "Credit card (auto)"))
    rows = []
    for ms in month_starts:
        days_late = np.where(rng.random(n) < 0.15 + 0.1 * _sigmoid(d), rng.poisson(late_lambda * 3), 0)
        due = ms + pd.Timedelta(days=9)
        rows.append(pd.DataFrame({
            "customer_id": cust_id, "due_date": due.strftime("%Y-%m-%d"),
            "paid_date": (due + pd.to_timedelta(days_late, unit="D")).strftime("%Y-%m-%d"),
            "amount": monthly_charges, "days_late": days_late,
        }))
    payments = pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------ tickets
    window_days = (snapshot - month_starts[0]).days
    n_tickets = rng.poisson(np.exp(-1.0 + 0.6 * d + 0.25 * (internet == "Fiber")) * 1.2)
    t_idx = np.repeat(np.arange(n), n_tickets)
    nt = len(t_idx)
    dt = d[t_idx]
    cats = np.array(["Billing", "Network", "Service", "Cancellation inquiry"])
    cat_p = np.stack([np.full(nt, 0.30), np.full(nt, 0.35), np.full(nt, 0.30 - 0.03 * dt),
                      0.05 + 0.04 * _sigmoid(dt * 2)], axis=1)
    cat_p /= cat_p.sum(axis=1, keepdims=True)
    cat_i = (rng.random(nt)[:, None] > np.cumsum(cat_p, axis=1)).sum(axis=1).clip(0, 3)
    sat = np.clip(np.round(3.4 - 0.55 * dt + rng.normal(0, 1.0, nt)), 1, 5)
    tickets = pd.DataFrame({
        "ticket_id": np.arange(1, nt + 1), "customer_id": cust_id[t_idx],
        "created_date": (month_starts[0] + pd.to_timedelta(rng.integers(0, window_days, nt), unit="D")
                         ).strftime("%Y-%m-%d"),
        "category": cats[cat_i],
        "is_complaint": (rng.random(nt) < 0.35 + 0.1 * _sigmoid(dt)).astype(int),
        "resolution_hours": np.round(rng.lognormal(2.2 + 0.25 * dt, 0.7, nt), 1),
        "escalated": (rng.random(nt) < 0.12 + 0.08 * _sigmoid(dt)).astype(int),
        "satisfaction": np.where(rng.random(nt) < 0.3, np.nan, sat),  # survey often skipped
    })

    # ----------------------------------------------------------- service calls
    n_calls = rng.poisson(np.exp(0.1 + 0.45 * d) * 1.5)
    c_idx = np.repeat(np.arange(n), n_calls)
    nc = len(c_idx)
    dc = d[c_idx]
    service_calls = pd.DataFrame({
        "call_id": np.arange(1, nc + 1), "customer_id": cust_id[c_idx],
        "call_date": (month_starts[0] + pd.to_timedelta(rng.integers(0, window_days, nc), unit="D")
                      ).strftime("%Y-%m-%d"),
        "duration_min": np.round(rng.lognormal(1.8 + 0.2 * dc, 0.6, nc), 1),
        "resolved": (rng.random(nc) < _sigmoid(1.2 - 0.7 * dc)).astype(int),
        "sentiment": np.round(np.clip(-0.25 * dc + rng.normal(0, 0.45, nc), -1, 1), 2),
    })

    return {"customers": customers, "usage": usage, "payments": payments,
            "tickets": tickets, "service_calls": service_calls}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=int, default=None,
                    help="bootstrap-sample the real customers (with replacement) to this many rows "
                         "to simulate cluster-scale volume; default uses all real customers")
    ap.add_argument("--real", default=REAL_CSV, help="path to the real Telco Customer Churn CSV")
    ap.add_argument("--out", default=RAW_URI if "://" not in RAW_URI else "data/raw",
                    help="local output folder (upload to HDFS afterwards with `hdfs dfs -put`)")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = generate(args.scale, args.seed, args.real)
    for name, df in tables.items():
        path = out / f"{name}.{args.format}"
        df.to_parquet(path, index=False) if args.format == "parquet" else df.to_csv(path, index=False)
        print(f"{name:14s} {len(df):>10,d} rows  -> {path}")
    rate = tables["customers"]["churned"].mean()
    print(f"\nReal churn rate: {rate:.1%}  ({len(tables['customers']):,} customers)")


if __name__ == "__main__":
    main()
