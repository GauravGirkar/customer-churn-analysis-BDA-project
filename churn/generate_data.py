"""Synthetic telecom churn data generator -> five relational tables.

Real telco data is private, so we simulate it. A hidden per-customer "dissatisfaction" factor drives
BOTH the observable behaviour (falling usage, late payments, tickets, unresolved calls) and the churn
label, plus contract/tenure/price effects and noise. The model never sees the hidden factor, so it has
to recover it from behaviour - which keeps results realistic (AUC ~0.8) instead of trivially perfect.

    python -m churn.generate_data --customers 100000
    python -m churn.generate_data --customers 2000000 --out hdfs-staging/   # then `hdfs dfs -put`
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import RAW_URI, SEED, SNAPSHOT_DATE

N_MONTHS = 6
TARGET_CHURN_RATE = 0.22


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(z, target):
    """Bisection so that mean(sigmoid(z + b)) == target churn rate."""
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate(n, seed=SEED):
    rng = np.random.default_rng(seed)
    snapshot = pd.Timestamp(SNAPSHOT_DATE)
    ids = np.arange(1, n + 1)
    cust_id = np.char.add("C", np.char.zfill(ids.astype(str), 8))

    # ---------------------------------------------------------------- customers
    tenure = np.clip(rng.gamma(2.0, 14.0, n).astype(int) + 1, 1, 72)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    plan = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.35, 0.40, 0.25])
    pay_method = rng.choice(
        ["Credit card (auto)", "Bank transfer (auto)", "Electronic check", "Mailed check"],
        n, p=[0.30, 0.25, 0.30, 0.15])
    internet = rng.choice(["Fiber", "DSL", "None"], n, p=[0.45, 0.35, 0.20])
    region = rng.choice(["North", "South", "East", "West", "Central"], n)
    age = np.clip(rng.normal(44, 15, n).astype(int), 18, 85)
    base_price = pd.Series(plan).map({"Basic": 30, "Standard": 55, "Premium": 85}).to_numpy()
    monthly_charges = np.round(base_price + (internet == "Fiber") * 20 + (internet == "DSL") * 8
                               + rng.normal(0, 6, n), 2).clip(15)
    signup = snapshot - pd.to_timedelta(tenure * 30, unit="D")

    d = rng.normal(0, 1, n)  # hidden dissatisfaction
    m2m = contract == "Month-to-month"
    two = contract == "Two year"
    echeck = pay_method == "Electronic check"

    z = (0.9 * d + 0.85 * m2m - 0.9 * two - 0.022 * tenure
         + 0.012 * (monthly_charges - 60) + 0.45 * echeck + 0.3 * (internet == "Fiber")
         - 0.008 * (age - 44) + rng.normal(0, 0.5, n))
    # non-linear effects real churn has (trees can learn these, a plain linear model cannot):
    z += 0.9 * (tenure <= 6)                                  # "honeymoon cliff" for brand-new accounts
    z += 0.7 * (m2m & echeck)                                 # flight-risk interaction
    z += 0.8 * (m2m & (plan == "Premium") & (monthly_charges > 95))  # price-shocked premium users
    z += 0.6 * np.clip(d, 0, None) ** 2 * (internet == "Fiber")      # unhappy fibre users escalate fast
    z += _calibrate_intercept(z, TARGET_CHURN_RATE)
    churned = (rng.random(n) < _sigmoid(z)).astype(int)

    customers = pd.DataFrame({
        "customer_id": cust_id, "signup_date": signup.strftime("%Y-%m-%d"), "age": age,
        "region": region, "contract": contract, "plan": plan, "internet_service": internet,
        "payment_method": pay_method, "monthly_charges": monthly_charges, "churned": churned,
    })

    # -------------------------------------------------------------------- usage
    month_starts = pd.date_range(end=snapshot, periods=N_MONTHS, freq="MS")
    base_min = rng.lognormal(5.6, 0.5, n)
    base_gb = rng.lognormal(2.0, 0.7, n) * (internet != "None")
    base_sms = rng.lognormal(3.5, 0.8, n)
    base_login = rng.lognormal(2.3, 0.6, n)
    slope = -0.11 * d + rng.normal(0, 0.04, n)  # unhappy customers fade out
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
    ap.add_argument("--customers", type=int, default=100_000)
    ap.add_argument("--out", default=RAW_URI if "://" not in RAW_URI else "data/raw",
                    help="local output folder (upload to HDFS afterwards with `hdfs dfs -put`)")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = generate(args.customers, args.seed)
    for name, df in tables.items():
        path = out / f"{name}.{args.format}"
        df.to_parquet(path, index=False) if args.format == "parquet" else df.to_csv(path, index=False)
        print(f"{name:14s} {len(df):>10,d} rows  -> {path}")
    rate = tables["customers"]["churned"].mean()
    print(f"\nChurn rate: {rate:.1%}  ({args.customers:,} customers)")


if __name__ == "__main__":
    main()
