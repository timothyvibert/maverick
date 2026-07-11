import numpy as np
import pandas as pd

from pm.config import ALERT_MULTIPLIER, TRADING_DAYS_PER_YEAR


def compute_daily_move_alerts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Today Return (%)"] = pd.to_numeric(out["Today Return (%)"], errors="coerce")
    out["VOL_30D_ANNUAL_PCT"] = pd.to_numeric(out["VOL_30D_ANNUAL_PCT"], errors="coerce")

    out["30D Daily Vol (%)"] = out["VOL_30D_ANNUAL_PCT"] / np.sqrt(TRADING_DAYS_PER_YEAR)
    out["today_return_decimal"] = out["Today Return (%)"] / 100.0

    out["Log Return"] = np.nan
    valid_return_mask = (
        out["today_return_decimal"].notna() &
        (out["today_return_decimal"] > -1.0)
    )

    out.loc[valid_return_mask, "Log Return"] = np.log1p(
        out.loc[valid_return_mask, "today_return_decimal"].astype(float).to_numpy()
    )

    out["daily_sigma_decimal"] = out["30D Daily Vol (%)"] / 100.0
    out["Alert Threshold (%)"] = ALERT_MULTIPLIER * out["30D Daily Vol (%)"]
    out["alert_threshold_decimal"] = ALERT_MULTIPLIER * out["daily_sigma_decimal"]

    out["Alert"] = False
    out["Move / AVg. Daily Volat"] = np.nan

    valid_sigma_mask = (
        out["daily_sigma_decimal"].notna() &
        (out["daily_sigma_decimal"] > 0)
    )

    full_mask = valid_return_mask & valid_sigma_mask

    out.loc[full_mask, "Alert"] = (
        out.loc[full_mask, "Log Return"].abs() >
        out.loc[full_mask, "alert_threshold_decimal"]
    )

    out.loc[full_mask, "Move / AVg. Daily Volat"] = (
        out.loc[full_mask, "Log Return"].abs() /
        out.loc[full_mask, "daily_sigma_decimal"]
    )

    out["Direction"] = np.where(
        out["Today Return (%)"] > 0,
        "UP",
        np.where(out["Today Return (%)"] < 0, "DOWN", "FLAT")
    )

    out["Excess vs Threshold (%)"] = (
        out["Today Return (%)"].abs() - out["Alert Threshold (%)"]
    )

    out["Alert"] = out["Alert"].fillna(False).astype(bool)

    return out
