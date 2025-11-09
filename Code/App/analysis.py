# analysis.py
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

RDW_ENDPOINT = "https://opendata.rdw.nl/resource/m9d7-ebf2.json"

def render_kpi_card(title: str, value_str: str, tooltip: str):
    st.markdown(f"""
    <div class="metric-card">
        <div class="kpi-help" tabindex="0" aria-label="Info">
            ?
            <div class="kpi-tooltip">{tooltip}</div>
        </div>
        <b>{title}</b><br>
        <span style="font-size:1.5rem;color:#6A0DAD;">{value_str}</span>
    </div>
    """, unsafe_allow_html=True)

# RDW fetch and preparation
@st.cache_data(ttl=900)
def fetch_rdw(brand: str, model: str) -> pd.DataFrame:
    params = {"merk": brand.upper(), "handelsbenaming": model.upper(), "$limit": 50000}
    r = requests.get(RDW_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def prepare_rdw(df: pd.DataFrame) -> pd.DataFrame:
    df["reg"] = pd.to_datetime(df.get("datum_tenaamstelling"), errors="coerce", format="%Y%m%d")
    df["reg_first_nl"] = pd.to_datetime(df.get("datum_eerste_tenaamstelling_in_nederland"), errors="coerce", format="%Y%m%d")
    df["Second-Hand"] = np.where(df["reg"] != df["reg_first_nl"], 1, 0)
    df = df.dropna(subset=["reg"])
    df["month"] = df["reg"].dt.to_period("M").dt.to_timestamp()
    df["date"] = df["reg"].dt.floor("D")
    return df

#Forecasting helpers
def hw_monthly(series: pd.Series, steps: int, seasonal="add", seasonal_periods=12) -> pd.DataFrame:
    s = series.sort_index().asfreq("MS").fillna(0)
    if len(s) < 3:
        idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        return pd.DataFrame({"ds": idx, "yhat": np.nan, "lo80": np.nan, "hi80": np.nan})
    use_seasonal = len(s) >= 2 * seasonal_periods
    try:
        model = ExponentialSmoothing(
            s, trend="add", seasonal=seasonal if use_seasonal else None,
            seasonal_periods=seasonal_periods if use_seasonal else None,
            initialization_method="estimated"
        ).fit(optimized=True)
        yhat = model.forecast(steps)
        resid = s - model.fittedvalues.reindex_like(s)
        se = float(resid.std(ddof=1)) if resid.notna().any() else 0.0
        lo = yhat - 1.28 * se
        hi = yhat + 1.28 * se
        return pd.DataFrame({"ds": yhat.index, "yhat": yhat.values, "lo80": lo.values, "hi80": hi.values})
    except Exception:
        idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        last = float(s.iloc[-1])
        return pd.DataFrame({"ds": idx, "yhat": [last]*steps, "lo80": [last]*steps, "hi80": [last]*steps})

def hw_daily(series: pd.Series, steps: int, seasonal="add", season_periods=7) -> pd.DataFrame:
    s = series.sort_index().asfreq("D").fillna(0)
    if len(s) < max(2*season_periods, 21):
        idx = pd.date_range(s.index[-1] + pd.offsets.Day(1), periods=steps, freq="D")
        return pd.DataFrame({"ds": idx, "yhat": np.nan, "lo80": np.nan, "hi80": np.nan})
    model = ExponentialSmoothing(
        s, trend="add", seasonal=seasonal, seasonal_periods=season_periods, initialization_method="estimated"
    ).fit(optimized=True)
    yhat = model.forecast(steps)
    resid = s - model.fittedvalues.reindex_like(s)
    se = float(resid.std(ddof=1))
    lo = yhat - 1.28 * se
    hi = yhat + 1.28 * se
    return pd.DataFrame({"ds": yhat.index, "yhat": yhat.values, "lo80": lo.values, "hi80": hi.values})

def annual_linear_forecast(series: pd.Series, steps: int) -> pd.DataFrame:
    if series.empty or len(series.dropna()) < 2:
        years = _future_years_from_series(series, steps)
        return pd.DataFrame({"ds": years, "yhat": [np.nan]*steps, "lo80": [np.nan]*steps, "hi80": [np.nan]*steps})
    idx = series.index
    if np.issubdtype(idx.dtype, np.integer):
        years_hist = idx.values.astype(float)
    else:
        years_hist = pd.to_datetime(idx).year.values.astype(float)
    y = series.values.astype(float)
    a, b = np.polyfit(years_hist, y, deg=1)
    yhat_hist = a*years_hist + b
    resid = y - yhat_hist
    se = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    last_year = int(np.max(years_hist))
    years_fut = np.arange(last_year+1, last_year+1+steps, dtype=int)
    yhat = a*years_fut + b
    lo = yhat - 1.28*se
    hi = yhat + 1.28*se
    return pd.DataFrame({"ds": years_fut, "yhat": yhat, "lo80": lo, "hi80": hi})

def _future_years_from_series(series: pd.Series, steps: int):
    if series.index.size == 0:
        base = pd.Timestamp.today().year
    else:
        idx = series.index
        base = int(idx.max()) if np.issubdtype(idx.dtype, np.integer) else int(pd.to_datetime(idx).year.max())
    return np.arange(base + 1, base + 1 + steps, dtype=int)

# Metrics & Backtests
def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1.0
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def backtest_smape_monthly(series: pd.Series, seasonal_periods=12) -> float | None:
    s = series.sort_index().asfreq("MS").fillna(0)
    if len(s) < 4:
        return None
    holdout = max(1, min(3, len(s)//6))
    train, test = s.iloc[:-holdout], s.iloc[-holdout:]
    fc = hw_monthly(train, steps=holdout, seasonal="add", seasonal_periods=seasonal_periods)
    fc = fc.set_index("ds").reindex(test.index)
    if fc["yhat"].isna().all():
        return None
    return smape(test.values, fc["yhat"].values)

def backtest_smape_weekly(series_week: pd.Series) -> float | None:
    if len(series_week) < 4:
        return None
    s = series_week.copy()
    s.index = pd.to_datetime(s.index)
    holdout = max(1, min(3, len(s)//6))
    train, test = s.iloc[:-holdout], s.iloc[-holdout:]
    daily_idx = pd.date_range(train.index.min(), train.index.max(), freq="D")
    s_daily = train.reindex(daily_idx).fillna(0)
    fc = hw_daily(s_daily, steps=holdout*7, season_periods=7)
    if fc.empty or fc["yhat"].isna().all():
        return None
    fc_week = pd.Series(fc["yhat"].values, index=pd.to_datetime(fc["ds"])).resample("W-MON").sum()
    test_week = test.resample("W-MON").sum()
    align = test_week.index.intersection(fc_week.index)
    if len(align) == 0:
        return None
    return smape(test_week.loc[align].values, fc_week.loc[align].values)

def backtest_smape_annual(series_year: pd.Series) -> float | None:
    s = series_year.sort_index()
    if len(s) < 3:
        return None
    holdout = 1
    train, test = s.iloc[:-holdout], s.iloc[-holdout:]
    idx = train.index
    x_train = idx.values.astype(float) if np.issubdtype(idx.dtype, np.integer) else pd.to_datetime(idx).year.values.astype(float)
    y_train = train.values.astype(float)
    a, b = np.polyfit(x_train, y_train, deg=1)
    x_test = test.index.values.astype(float) if np.issubdtype(test.index.dtype, np.integer) else pd.to_datetime(test.index).year.values.astype(float)
    y_pred = a * x_test + b
    return smape(test.values.astype(float), y_pred.astype(float))

# KPI helpers (annualization & CAGR 5y)
def annualize_monthly_forecast(fc_df: pd.DataFrame, value_col="yhat") -> pd.Series:
    tmp = fc_df.dropna(subset=["ds", value_col]).copy()
    tmp["year"] = pd.to_datetime(tmp["ds"]).dt.year
    return tmp.groupby("year")[value_col].sum()

def cagr_5y(s0, s5):
    try:
        s0 = float(s0)
        s5 = float(s5)
        if s0 <= 0 or np.isnan(s0) or np.isnan(s5):
            return np.nan
        return (s5 / s0)**(1/5) - 1
    except Exception:
        return np.nan

def _fmt_pct(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "N/A"
        return f"{float(x):.2f} %"
    except Exception:
        return "N/A"

# Plotly helpers
def add_pi_band(fig: go.Figure, x, lo, hi, name="80% PI"):
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill='tonexty', name=name, opacity=0.2, hoverinfo="skip"))

def add_labels_and_smape(fig: go.Figure, x_title: str, y_title: str, smape_val: float | None):
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    if smape_val is not None and np.isfinite(smape_val):
        fig.add_annotation(
            text=f"SMAPE (bt): {smape_val:.3f}",
            xref="paper", yref="paper", x=1, y=1.08,
            showarrow=False, align="right", font=dict(size=12)
        )

#GRAPHIC PART
def render_analysis(selected_brand: str, selected_model: str):
    st.markdown('<div class="section-title">Market Trend Live</div>', unsafe_allow_html=True)

    if not (selected_brand and selected_model):
        st.info("Select a brand and model in the sidebar to load data.")
        return

    #Data loading
    try:
        raw = fetch_rdw(selected_brand, selected_model)
    except Exception as e:
        st.error(f"RDW error: {e}")
        return

    if raw.empty:
        st.info("No RDW rows returned for this selection.")
        return
    df = prepare_rdw(raw)

    #Granularity Selection and Graphs
    granularity = st.selectbox("Select Granularity", ["Weekly", "Monthly", "Annual"])
    
    if granularity == "Annual": #Annual
        df_agg = df.groupby([df['reg_first_nl'].dt.year, "Second-Hand"]).size().reset_index(name="units")
        df_agg.rename(columns={"reg_first_nl": "year"}, inplace=True)
        x_col = "year"
        y_new  = df_agg[df_agg["Second-Hand"]==0].set_index(x_col)["units"]
        y_used = df_agg[df_agg["Second-Hand"]==1].set_index(x_col)["units"]
        forecast_period = st.number_input("Forecast Years", min_value=1, max_value=10, value=2, step=1)

    elif granularity == "Monthly": #Monthly
        df_agg = df.groupby(["month","Second-Hand"]).size().reset_index(name="units")
        x_col = "month"
        y_new  = df_agg[df_agg["Second-Hand"]==0].set_index(x_col)["units"]
        y_used = df_agg[df_agg["Second-Hand"]==1].set_index(x_col)["units"]
        forecast_period = st.number_input("Forecast Months", min_value=1, max_value=36, value=12, step=1)

    else: #Weekly
        df["week"] = df["reg"].dt.to_period("W").apply(lambda r: r.start_time)
        df_agg = df.groupby(["week","Second-Hand"]).size().reset_index(name="units")
        x_col = "week"
        y_new  = df_agg[df_agg["Second-Hand"]==0].set_index(x_col)["units"]
        y_used = df_agg[df_agg["Second-Hand"]==1].set_index(x_col)["units"]
        forecast_period = st.number_input("Forecast Weeks", min_value=1, max_value=52, value=8, step=1)


    col5, col6 = st.columns(2)
    #New Car Registrations
    with col5:
        st.markdown(f"**New Registrations**")
        if granularity == "Monthly": #Monthly
            fc_new = hw_monthly(y_new, forecast_period, seasonal="add", seasonal_periods=12)
            x_hist = y_new.index
            x_fc   = fc_new["ds"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_hist, y=y_new, mode="lines+markers", name="History"))
            if not fc_new.empty and fc_new["yhat"].notna().any():
                fig.add_trace(go.Scatter(x=x_fc, y=fc_new["yhat"], mode="lines+markers", name="Forecast"))
                add_pi_band(fig, x_fc, fc_new["lo80"], fc_new["hi80"])
            sm = backtest_smape_monthly(y_new, seasonal_periods=12)
            add_labels_and_smape(fig, "Month", "Units", sm)
            st.plotly_chart(fig, use_container_width=True)

        elif granularity == "Weekly": #Weekly
            s = y_new.copy()
            s.index = pd.to_datetime(s.index)
            daily_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
            s_daily = s.reindex(daily_idx).fillna(0)
            fc_new = hw_daily(s_daily, steps=forecast_period*7, season_periods=7)
            x_hist = s.index
            x_fc   = fc_new["ds"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_hist, y=s, mode="lines+markers", name="History"))
            if not fc_new.empty and fc_new["yhat"].notna().any():
                fig.add_trace(go.Scatter(x=x_fc, y=fc_new["yhat"], mode="lines+markers", name="Forecast"))
                add_pi_band(fig, x_fc, fc_new["lo80"], fc_new["hi80"])
            sm = backtest_smape_weekly(s)
            add_labels_and_smape(fig, "Week", "Units", sm)
            st.plotly_chart(fig, use_container_width=True)

        else:  #Annual
            fc_new = annual_linear_forecast(y_new, steps=forecast_period)
            x_hist = y_new.index
            x_fc   = fc_new["ds"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_hist, y=y_new, mode="lines+markers", name="History"))
            if not fc_new.empty and fc_new["yhat"].notna().any():
                fig.add_trace(go.Scatter(x=x_fc, y=fc_new["yhat"], mode="lines+markers", name="Forecast"))
                add_pi_band(fig, x_fc, fc_new["lo80"], fc_new["hi80"])
            sm = backtest_smape_annual(y_new)
            add_labels_and_smape(fig, "Year", "Units", sm)
            st.plotly_chart(fig, use_container_width=True)

    #Car Occasions: Second-Hand 
    with col6:
        st.markdown(f"**Second-Hand Cars ({granularity})**")
        if granularity == "Monthly": #Monthly
            fc_used = hw_monthly(y_used, forecast_period, seasonal="add", seasonal_periods=12)
            x_hist = y_used.index
            x_fc   = fc_used["ds"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_hist, y=y_used, mode="lines+markers", name="History"))
            if not fc_used.empty and fc_used["yhat"].notna().any():
                fig.add_trace(go.Scatter(x=x_fc, y=fc_used["yhat"], mode="lines+markers", name="Forecast"))
                add_pi_band(fig, x_fc, fc_used["lo80"], fc_used["hi80"])
            sm = backtest_smape_monthly(y_used, seasonal_periods=12)
            add_labels_and_smape(fig, "Month", "Units", sm)
            st.plotly_chart(fig, use_container_width=True)

        elif granularity == "Weekly": #Weekly
            s = y_used.copy()
            s.index = pd.to_datetime(s.index)
            daily_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
            s_daily = s.reindex(daily_idx).fillna(0)
            fc_used = hw_daily(s_daily, steps=forecast_period*7, season_periods=7)
            x_hist = s.index
            x_fc   = fc_used["ds"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_hist, y=s, mode="lines+markers", name="History"))
            if not fc_used.empty and fc_used["yhat"].notna().any():
                fig.add_trace(go.Scatter(x=x_fc, y=fc_used["yhat"], mode="lines+markers", name="Forecast"))
                add_pi_band(fig, x_fc, fc_used["lo80"], fc_used["hi80"])
            sm = backtest_smape_weekly(s)
            add_labels_and_smape(fig, "Week", "Units", sm)
            st.plotly_chart(fig, use_container_width=True)

        else: #Annual
            fc_used = annual_linear_forecast(y_used, steps=forecast_period)
            x_hist = y_used.index
            x_fc   = fc_used["ds"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_hist, y=y_used, mode="lines+markers", name="History"))
            if not fc_used.empty and fc_used["yhat"].notna().any():
                fig.add_trace(go.Scatter(x=x_fc, y=fc_used["yhat"], mode="lines+markers", name="Forecast"))
                add_pi_band(fig, x_fc, fc_used["lo80"], fc_used["hi80"])
            sm = backtest_smape_annual(y_used)
            add_labels_and_smape(fig, "Year", "Units", sm)
            st.plotly_chart(fig, use_container_width=True)

    #Success Indicators: Projected Market Growth (5-year CAGR)
    st.markdown('<div class="section-title">Success Indicators - Projected Market Growth (5 Years) </div>', unsafe_allow_html=True)
    
    monthly = df.groupby(["month", "Second-Hand"]).size().reset_index(name="units")
    monthly_horizon = 60

    def compute_kpis_from_monthly(monthly_df: pd.DataFrame) -> dict:
        kpis = {}
        for occ, label in [(0, "New"), (1, "Second-Hand")]:
            sub = monthly_df[monthly_df["Second-Hand"] == occ].copy()
            if sub.empty:
                kpis[label] = {"CAGR_5y": np.nan, "CAGR_5y_lo80": np.nan, "CAGR_5y_hi80": np.nan}
                continue

            y = sub.set_index("month")["units"].asfreq("MS").fillna(0)
            fc = hw_monthly(y, steps=monthly_horizon, seasonal="add", seasonal_periods=12)

            last_hist_year = int(y.index.max().year)
            if y.loc[str(last_hist_year)].isna().all() or y.loc[str(last_hist_year)].sum() == 0:
                last_hist_year -= 1

            hist_annual = y.groupby(y.index.year).sum()
            fc_yhat = annualize_monthly_forecast(fc, "yhat")
            fc_lo   = annualize_monthly_forecast(fc, "lo80")
            fc_hi   = annualize_monthly_forecast(fc, "hi80")

            S_t   = hist_annual.get(last_hist_year, np.nan)
            target_year = last_hist_year + 5
            S_t5     = fc_yhat.get(target_year, np.nan)
            S_t5_lo  = fc_lo.get(target_year, np.nan)
            S_t5_hi  = fc_hi.get(target_year, np.nan)

            cagr     = cagr_5y(S_t, S_t5)
            cagr_lo  = cagr_5y(S_t, S_t5_lo) if not np.isnan(S_t5_lo) else np.nan
            cagr_hi  = cagr_5y(S_t, S_t5_hi) if not np.isnan(S_t5_hi) else np.nan

            kpis[label] = {
                "base_year": last_hist_year,
                "S_t": float(S_t) if S_t is not None else np.nan,
                "S_t+5": float(S_t5) if S_t5 is not None else np.nan,
                "CAGR_5y": cagr * 100 if not np.isnan(cagr) else np.nan,
                "CAGR_5y_lo80": cagr_lo * 100 if not np.isnan(cagr_lo) else np.nan,
                "CAGR_5y_hi80": cagr_hi * 100 if not np.isnan(cagr_hi) else np.nan,
            }
        return kpis

    kpis = compute_kpis_from_monthly(monthly)

    KPI4_new_pct       = kpis.get("New", {}).get("CAGR_5y", np.nan)
    KPI4_new_lo_pct    = kpis.get("New", {}).get("CAGR_5y_lo80", np.nan)
    KPI4_new_hi_pct    = kpis.get("New", {}).get("CAGR_5y_hi80", np.nan)

    KPI5_occasion_pct  = kpis.get("Second-Hand", {}).get("CAGR_5y", np.nan)
    KPI5_occ_lo_pct    = kpis.get("Second-Hand", {}).get("CAGR_5y_lo80", np.nan)
    KPI5_occ_hi_pct    = kpis.get("Second-Hand", {}).get("CAGR_5y_hi80", np.nan)

    _, center_col, _ = st.columns([1, 2, 1])

    kpi_left, kpi_right = st.columns(2, gap="large")

    def _render_kpi(title: str, value_str: str, tooltip: str):
        
        if "render_kpi_card" in globals():
            try:
                render_kpi_card(title=title, value_str=value_str, tooltip=tooltip)
                return
            except Exception:
                pass
        st.metric(title, value_str)

    with kpi_left:
        _render_kpi(
            title="Projection New Cars (%)",
            value_str=_fmt_pct(KPI4_new_pct),
            tooltip=(
                "Five-year Compound Annual Growth Rate (CAGR) on new car registrations. "
                "It reflects the average annual rate of change expected over the next 5 years, "
                "aggregating monthly forecasts to calendar years."
            )
        )
        st.caption(f"80 % PI: {_fmt_pct(KPI4_new_lo_pct)} – {_fmt_pct(KPI4_new_hi_pct)}")

    with kpi_right:
        _render_kpi(
            title="Projection Second-Hand Cars (%)",
            value_str=_fmt_pct(KPI5_occasion_pct),
            tooltip=(
                "Five-year CAGR on used car re-registrations (second-hand market). "
                "Derived by annualizing monthly forecasts and comparing the target year to the latest historical year."
            )
        )
        st.caption(f"80 % PI: {_fmt_pct(KPI5_occ_lo_pct)} – {_fmt_pct(KPI5_occ_hi_pct)}")