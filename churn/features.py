"""Step 2 - Preprocessing & feature engineering in Spark SQL.

Five raw tables (customers, usage, payments, tickets, service_calls) are aggregated per customer
over the 6-month observation window and joined into one wide feature table (1 row / customer).
The SQL is plain ANSI/Hive-compatible, so it can be lifted into a Hive view unchanged."""
from churn.config import SNAPSHOT_DATE

CATEGORICAL = ["contract", "plan", "internet_service", "payment_method", "region"]

NUMERIC = [
    # profile
    "age", "tenure_months", "monthly_charges", "total_charges",
    # usage frequency & momentum (recent 3 months vs. the 3 before)
    "avg_call_minutes", "avg_data_gb", "avg_sms", "avg_app_logins",
    "minutes_trend", "data_trend", "logins_trend",
    # payment behaviour
    "late_payments", "late_payment_ratio", "avg_days_late", "max_days_late", "late_payments_recent",
    # complaints / tickets
    "ticket_count", "complaint_count", "escalated_count", "billing_tickets", "network_tickets",
    "cancellation_inquiries", "avg_resolution_hours", "avg_satisfaction", "tickets_last_30d",
    # customer-service interactions
    "call_count", "avg_call_duration", "unresolved_calls", "avg_sentiment", "calls_last_30d",
    # derived
    "contacts_per_month",
]

FEATURE_SQL = f"""
WITH usage_agg AS (
  SELECT customer_id,
    AVG(call_minutes) AS avg_call_minutes,
    AVG(data_gb)      AS avg_data_gb,
    AVG(sms_count)    AS avg_sms,
    AVG(app_logins)   AS avg_app_logins,
    COALESCE(AVG(CASE WHEN CAST(month AS DATE) >  add_months(DATE '{SNAPSHOT_DATE}', -3) THEN call_minutes END)
           / NULLIF(AVG(CASE WHEN CAST(month AS DATE) <= add_months(DATE '{SNAPSHOT_DATE}', -3) THEN call_minutes END), 0), 1.0) AS minutes_trend,
    COALESCE(AVG(CASE WHEN CAST(month AS DATE) >  add_months(DATE '{SNAPSHOT_DATE}', -3) THEN data_gb END)
           / NULLIF(AVG(CASE WHEN CAST(month AS DATE) <= add_months(DATE '{SNAPSHOT_DATE}', -3) THEN data_gb END), 0), 1.0) AS data_trend,
    COALESCE(AVG(CASE WHEN CAST(month AS DATE) >  add_months(DATE '{SNAPSHOT_DATE}', -3) THEN app_logins END)
           / NULLIF(AVG(CASE WHEN CAST(month AS DATE) <= add_months(DATE '{SNAPSHOT_DATE}', -3) THEN app_logins END), 0), 1.0) AS logins_trend
  FROM usage GROUP BY customer_id
),
pay_agg AS (
  SELECT customer_id,
    SUM(CASE WHEN days_late > 0 THEN 1 ELSE 0 END)                  AS late_payments,
    AVG(CASE WHEN days_late > 0 THEN 1.0 ELSE 0.0 END)              AS late_payment_ratio,
    AVG(days_late)                                                  AS avg_days_late,
    MAX(days_late)                                                  AS max_days_late,
    SUM(CASE WHEN days_late > 0 AND CAST(due_date AS DATE) > add_months(DATE '{SNAPSHOT_DATE}', -3)
             THEN 1 ELSE 0 END)                                     AS late_payments_recent
  FROM payments GROUP BY customer_id
),
ticket_agg AS (
  SELECT customer_id,
    COUNT(*)                                                         AS ticket_count,
    SUM(is_complaint)                                                AS complaint_count,
    SUM(escalated)                                                   AS escalated_count,
    SUM(CASE WHEN category = 'Billing' THEN 1 ELSE 0 END)            AS billing_tickets,
    SUM(CASE WHEN category = 'Network' THEN 1 ELSE 0 END)            AS network_tickets,
    SUM(CASE WHEN category = 'Cancellation inquiry' THEN 1 ELSE 0 END) AS cancellation_inquiries,
    AVG(resolution_hours)                                            AS avg_resolution_hours,
    AVG(satisfaction)                                                AS avg_satisfaction,
    SUM(CASE WHEN CAST(created_date AS DATE) >= date_sub(DATE '{SNAPSHOT_DATE}', 30)
             THEN 1 ELSE 0 END)                                      AS tickets_last_30d
  FROM tickets GROUP BY customer_id
),
call_agg AS (
  SELECT customer_id,
    COUNT(*)                                                         AS call_count,
    AVG(duration_min)                                                AS avg_call_duration,
    SUM(1 - resolved)                                                AS unresolved_calls,
    AVG(sentiment)                                                   AS avg_sentiment,
    SUM(CASE WHEN CAST(call_date AS DATE) >= date_sub(DATE '{SNAPSHOT_DATE}', 30)
             THEN 1 ELSE 0 END)                                      AS calls_last_30d
  FROM service_calls GROUP BY customer_id
),
joined AS (
  SELECT c.customer_id, c.churned AS label,
    c.age, c.contract, c.plan, c.internet_service, c.payment_method, c.region, c.monthly_charges,
    CAST(FLOOR(DATEDIFF(DATE '{SNAPSHOT_DATE}', CAST(c.signup_date AS DATE)) / 30) AS DOUBLE) AS tenure_months,
    COALESCE(u.avg_call_minutes, 0) AS avg_call_minutes, COALESCE(u.avg_data_gb, 0) AS avg_data_gb,
    COALESCE(u.avg_sms, 0) AS avg_sms, COALESCE(u.avg_app_logins, 0) AS avg_app_logins,
    COALESCE(u.minutes_trend, 1.0) AS minutes_trend, COALESCE(u.data_trend, 1.0) AS data_trend,
    COALESCE(u.logins_trend, 1.0) AS logins_trend,
    COALESCE(p.late_payments, 0) AS late_payments, COALESCE(p.late_payment_ratio, 0) AS late_payment_ratio,
    COALESCE(p.avg_days_late, 0) AS avg_days_late, COALESCE(p.max_days_late, 0) AS max_days_late,
    COALESCE(p.late_payments_recent, 0) AS late_payments_recent,
    COALESCE(t.ticket_count, 0) AS ticket_count, COALESCE(t.complaint_count, 0) AS complaint_count,
    COALESCE(t.escalated_count, 0) AS escalated_count, COALESCE(t.billing_tickets, 0) AS billing_tickets,
    COALESCE(t.network_tickets, 0) AS network_tickets,
    COALESCE(t.cancellation_inquiries, 0) AS cancellation_inquiries,
    COALESCE(t.avg_resolution_hours, 0) AS avg_resolution_hours,
    COALESCE(t.avg_satisfaction, 3.0) AS avg_satisfaction,     -- neutral when no survey answered
    COALESCE(t.tickets_last_30d, 0) AS tickets_last_30d,
    COALESCE(s.call_count, 0) AS call_count, COALESCE(s.avg_call_duration, 0) AS avg_call_duration,
    COALESCE(s.unresolved_calls, 0) AS unresolved_calls, COALESCE(s.avg_sentiment, 0) AS avg_sentiment,
    COALESCE(s.calls_last_30d, 0) AS calls_last_30d
  FROM customers c
  LEFT JOIN usage_agg  u ON c.customer_id = u.customer_id
  LEFT JOIN pay_agg    p ON c.customer_id = p.customer_id
  LEFT JOIN ticket_agg t ON c.customer_id = t.customer_id
  LEFT JOIN call_agg   s ON c.customer_id = s.customer_id
)
SELECT *,
  monthly_charges * tenure_months                                    AS total_charges,
  -- support contacts per observed month (window is 6 months, shorter for brand-new accounts)
  (ticket_count + call_count) / LEAST(6.0, GREATEST(tenure_months, 1.0)) AS contacts_per_month,
  -- deterministic 70/15/15 split from a hash of the id (stable across runs / cluster sizes)
  CASE WHEN pmod(xxhash64(customer_id), 100) < 70 THEN 'train'
       WHEN pmod(xxhash64(customer_id), 100) < 85 THEN 'val' ELSE 'test' END AS split
FROM joined
"""


def build_features(spark):
    """Run the feature SQL against the registered raw views; returns a cached Spark DataFrame."""
    return spark.sql(FEATURE_SQL).cache()
