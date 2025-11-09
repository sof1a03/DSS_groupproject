# ev.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import re as _re_label

def _titleize(s: str) -> str:
    s = s.replace('_', ' ').replace('-', ' ')
    s = _re_label.sub(r'\s+', ' ', s).strip()
    words = []
    for w in s.split(' '):
        words.append(w.upper() if w.lower() in {'ev','pc4','pc3','nl','id'} else w.capitalize())
    return ' '.join(words)

def _rename_columns_nicely(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _titleize(c) for c in df.columns})

def df_full_height(df: pd.DataFrame, row_px: int = 34, header_px: int = 38) -> int:
    n = max(1, len(df))
    return header_px + n * row_px


def ensure_kpi_css():
    if not st.session_state.get("_kpi_css_injected_ev", False):
        st.markdown("""
<style>
.metric-card { 
  position: relative; 
  background: #ffffff; 
  border: 1px solid #e8e2f5; 
  border-radius: 12px; 
  padding: 12px 14px; 
  margin-bottom: 10px;
  box-shadow: 0 2px 10px rgba(0,0,0,.04);
}

.metric-card b { 
  display:block; 
  color:#4b0082; 
  font-weight:700; 
  margin-bottom:2px;
}

.metric-card .metric-value {
  font-size: 1.6rem; 
  color:#6A0DAD; 
  font-weight:800;
}

.kpi-help {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #efe9f7;
  color: #6A0DAD;
  font-size: 12px;
  font-weight: 700;
  line-height: 18px;
  text-align: center;
  cursor: default;
  user-select: none;
  border: 1px solid #e0d4f2;
}

/* Tooltip */
.kpi-help .kpi-tooltip {
  position: absolute;
  top: 24px;
  right: 0;
  background: #FFEAF1;
  color: #333;
  border: 1px solid #e6e6e6;
  border-radius: 8px;
  padding: 8px 10px;
  width: 260px;
  box-shadow: 0 8px 24px rgba(0,0,0,.12);
  font-size: 12.5px;
  line-height: 1.35;
  z-index: 50;
  opacity: 0;
  visibility: hidden;
  transition: opacity .15s ease, visibility .15s ease;
}

.kpi-help .kpi-tooltip:before {
  content: "";
  position: absolute;
  top: -6px;
  right: 8px;
  border-width: 6px;
  border-style: solid;
  border-color: transparent transparent #FFEAF1 transparent;
  filter: drop-shadow(0 -1px 0 #e6e6e6);
}

.kpi-help:hover .kpi-tooltip { opacity: 1; visibility: visible; }

.kpi-help:focus .kpi-tooltip,
.kpi-help:active .kpi-tooltip { opacity: 1; visibility: visible; }

.big-kpi {
  background:#ffffff;
  border:1px solid #ece7f8;
  border-radius:14px;
  padding:16px 18px;
  box-shadow:0 4px 18px rgba(0,0,0,.05);
  text-align:center;
}
.big-kpi .big-kpi-title {
  font-size:0.95rem;
  font-weight:700;
  color:#4b0082;
  margin-bottom:6px;
}
.big-kpi .big-kpi-value {
  font-size:2.2rem;
  font-weight:800;
  color:#6A0DAD;
  letter-spacing:0.2px;
}
.big-kpi .big-kpi-sub {
  font-size:0.85rem;
  color:#666;
  margin-top:4px;
}
</style>
        """, unsafe_allow_html=True)
        st.session_state["_kpi_css_injected_ev"] = True

#Render function for the stile of the Success Indicators
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


def _inject_ev_css_once(key: str = "_ev_css_loaded"):
    if st.session_state.get(key):
        return
    st.session_state[key] = True
    st.markdown("""
        <style>
        .cargo-accordion {
          border: 1px solid #ECE7F8;
          border-radius: 12px;
          overflow: hidden;
          margin: 8px 0 12px 0;
          background: #fff;
          box-shadow: 0 4px 12px rgba(0,0,0,.03);
        }
        .cargo-accordion summary {
          list-style: none;
          cursor: pointer;
          padding: 12px 14px;
          font-weight: 700;
          color: #4b0082;
          user-select: none;
          position: relative;
        }
        .cargo-accordion summary::-webkit-details-marker { display:none; }
        .cargo-accordion summary:after {
          content: "▸";
          position: absolute;
          right: 12px;
          transform: rotate(0deg);
          transition: transform .15s ease;
          color: #6A0DAD;
        }
        .cargo-accordion[open] summary:after { transform: rotate(90deg); }
        .cargo-accordion .accordion-body {
          padding: 10px 14px 14px 14px;
          border-top: 1px solid #F2ECFF;
          font-size: 13px;
          line-height: 1.35;
        }
        </style>
    """, unsafe_allow_html=True)


def render_accordion(title: str, body_html: str, open_: bool=False):
    open_attr = " open" if open_ else ""
    st.markdown(
        f'<details class="cargo-accordion"{open_attr}><summary>{title}</summary>'
        f'<div class="accordion-body">{body_html}</div></details>',
        unsafe_allow_html=True
    )

#Helpers BigQuery
@st.cache_data(ttl=3600)
def bq_read_regions(_client, project_id: str) -> pd.DataFrame:
    q = f"""
    SELECT
      SAFE_CAST(pc4 AS INT64) AS pc4,
      p_electric, p_hybrid, p_gasoline, p_diesel,
      avg_yearly_income_k, urbanization, avg_household_size,
      inhabitants_total,
      SAFE_CAST(std_avg_yearly_income_k AS FLOAT64) AS std_avg_yearly_income_k
    FROM `{project_id}.REGIONAL.REGIONAL_2`
    """
    df = _client.query(q).to_dataframe()
    df = df.dropna(subset=["pc4"]).copy()
    df["pc4"] = df["pc4"].astype(int)
    df = df.sort_values("pc4").drop_duplicates(subset=["pc4"], keep="first").reset_index(drop=True)
    return df

@st.cache_data(ttl=3600)
def bq_read_conversion(_client, project_id: str) -> pd.DataFrame:
    q = f"""
    SELECT DISTINCT
      SAFE_CAST(PC4 AS INT64) AS pc4,
      CAST(Gemeentenaam2023 AS STRING) AS gm_name
    FROM `{project_id}.REGIONAL.conversion`
    WHERE PC4 IS NOT NULL AND Gemeentenaam2023 IS NOT NULL
    """
    df = _client.query(q).to_dataframe()
    df["pc4"] = df["pc4"].astype(int)
    df = df[["pc4", "gm_name"]].drop_duplicates()
    df = df.sort_values(["gm_name","pc4"]).drop_duplicates(subset=["pc4"], keep="first").reset_index(drop=True)
    return df

#URL Map EV columns
OCM_API_URL = "https://api.openchargemap.io/v3/poi/"

@st.cache_data(ttl=60*60*12)
def fetch_ocm_nl(api_key: str, maxresults: int = 50000) -> List[Dict[str, Any]]:
    headers = {
        "User-Agent": "EV-Insights/1.0 (CarGo Dashboard)",
        "X-API-Key": api_key or "",
    }
    params = {
        "output": "json",
        "countrycode": "NL",
        "maxresults": maxresults,
        "compact": "true",
        "verbose": "false",
    }
    r = requests.get(OCM_API_URL, headers=headers, params=params, timeout=180)
    r.raise_for_status()
    return r.json()

def _safe(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k)
    return cur if cur is not None else default

def flatten_ocm(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        addr = _safe(r, "AddressInfo", default={}) or {}
        pc_raw = addr.get("Postcode", None)
        digits = "".join(ch for ch in str(pc_raw) if ch and str(ch).isdigit())
        pc4 = int(digits[:4]) if digits and len(digits) >= 4 else None

        conns = r.get("Connections", []) or []
        n_ports = 0
        total_kw = 0.0
        fast_ports = 0
        ultra_ports = 0
        kw_list = []

        for c in conns:
            n_ports += 1
            kw = c.get("PowerKW", None)
            if kw is None:
                continue
            try:
                kw = float(kw)
                kw_list.append(kw)
                total_kw += kw
                if kw >= 150:
                    ultra_ports += 1
                elif kw >= 50:
                    fast_ports += 1
            except Exception:
                pass

        rows.append({
            "poi_id": r.get("ID"),
            "town": addr.get("Town"),
            "postcode_raw": pc_raw,
            "pc4": pc4,
            "latitude": addr.get("Latitude"),
            "longitude": addr.get("Longitude"),
            "is_operational": bool(_safe(r, "StatusType", "IsOperational", default=np.nan))
                               if _safe(r, "StatusType", "IsOperational", default=None) is not None else np.nan,
            "status_title": _safe(r, "StatusType", "Title"),
            "operator": _safe(r, "OperatorInfo", "Title"),
            "n_ports": n_ports,
            "fast_ports": fast_ports,
            "ultra_fast_ports": ultra_ports,
            "total_power_kw": total_kw,
            "avg_power_kw": (np.mean(kw_list) if kw_list else np.nan),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["pc4"] = pd.to_numeric(df["pc4"], errors="coerce").astype("Int64")
    return df

def ocm_pc4_aggregate(df_flat: pd.DataFrame) -> pd.DataFrame:
    if df_flat is None or df_flat.empty:
        return pd.DataFrame(columns=[
            "pc4","n_stations","n_operational","operational_share",
            "n_ports","fast_ports","ultra_fast_ports",
            "total_power_kw","avg_power_kw"
        ])
    g = (
        df_flat.dropna(subset=["pc4"])
               .groupby("pc4", as_index=False)
               .agg(
                   n_stations=("poi_id", "nunique"),
                   n_operational=("is_operational", lambda s: np.nansum(np.array(s, dtype=float))),
                   n_ports=("n_ports", "sum"),
                   fast_ports=("fast_ports", "sum"),
                   ultra_fast_ports=("ultra_fast_ports", "sum"),
                   total_power_kw=("total_power_kw", "sum"),
                   avg_power_kw=("avg_power_kw", "mean"),
               )
    )
    g["operational_share"] = np.where(g["n_stations"] > 0, g["n_operational"] / g["n_stations"], np.nan)
    g["pc4"] = g["pc4"].astype(int)
    return g[[
        "pc4","n_stations","n_operational","operational_share",
        "n_ports","fast_ports","ultra_fast_ports",
        "total_power_kw","avg_power_kw"
    ]]

#Clutering, Forecasting and Indexes
def quantile_scale(series, q_low=0.01, q_high=0.99):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    s_clipped = s.clip(lo, hi)
    return 100 * (s_clipped - lo) / (hi - lo + 1e-12)

def build_clusters(R: pd.DataFrame):
    clust_cols = [
        "p_electric", "p_hybrid", "p_gasoline", "p_diesel",
        "avg_yearly_income_k", "urbanization", "avg_household_size"
    ]
    Z = R.dropna(subset=clust_cols + ["pc4"]).copy()
    X = Z[clust_cols].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(5, Xs.shape[1]))
    X_p = pca.fit_transform(Xs)

    k_results = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_p)
        sil = silhouette_score(X_p, labels)
        k_results.append((k, sil))
    best_k = max(k_results, key=lambda t: t[1])[0]

    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    Z["ev_cluster"] = km.fit_predict(X_p)
    sil_final = silhouette_score(X_p, Z["ev_cluster"])
    Z["silhouette_i"] = silhouette_samples(X_p, Z["ev_cluster"])

    centroids = km.cluster_centers_
    dists = np.linalg.norm(X_p - centroids[Z["ev_cluster"]], axis=1)
    Z["dist_to_centroid"] = dists
    Z["cluster_confidence"] = 1 - (Z.groupby("ev_cluster")["dist_to_centroid"]
                                     .transform(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)))

    profile_cols = [
        "p_electric","p_hybrid","p_gasoline","p_diesel",
        "avg_yearly_income_k","urbanization","avg_household_size","inhabitants_total"
    ]
    profile = (
        Z.groupby("ev_cluster")[profile_cols]
         .median()
         .reset_index()
         .sort_values("p_electric", ascending=False)
    )
    order = profile["ev_cluster"].tolist()
    label_map = {c: f"Cluster_{i+1}" for i, c in enumerate(order)}
    Z["cluster_label"] = Z["ev_cluster"].map(label_map)

    pca_df = pd.DataFrame({
        "pc4": Z["pc4"].values,
        "x": X_p[:, 0],
        "y": X_p[:, 1],
        "cluster_label": Z["cluster_label"].values
    })

    R2 = R.merge(Z[["pc4","ev_cluster","cluster_label","silhouette_i","cluster_confidence"]],
                 on="pc4", how="left")
    return R2, profile, best_k, sil_final, pca_df

def ev_potential_and_opportunity(R: pd.DataFrame) -> pd.DataFrame:
    potential_cols = ["avg_yearly_income_k", "urbanization", "p_hybrid", "p_diesel", "p_gasoline"]
    Z_ = R[potential_cols].copy()
    Z_ = (Z_ - Z_.mean()) / (Z_.std(ddof=0).replace(0, 1e-9))
    R["EV_Potential_raw"] = (
        Z_["avg_yearly_income_k"] + Z_["urbanization"] + Z_["p_hybrid"]
        - Z_["p_diesel"] - Z_["p_gasoline"]
    ) / 5.0
    R["EV_Potential_0_100"] = quantile_scale(R["EV_Potential_raw"], 0.01, 0.99)
    R["Opportunity"] = R["EV_Potential_0_100"] * np.log1p(R["inhabitants_total"].fillna(0))
    return R

def smape_numpy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    return 100 * np.mean(np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom))

def fit_ridge_structural(R: pd.DataFrame):
    X_cols = ["avg_yearly_income_k", "urbanization", "p_hybrid", "p_gasoline", "p_diesel"]
    y_col  = "p_electric"
    R_fit = R.dropna(subset=X_cols + [y_col]).copy()
    X_all, y_all = R_fit[X_cols], R_fit[y_col]

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 21), cv=5))])
    pipe.fit(X_tr, y_tr)
    y_hat_te = pipe.predict(X_te)
    residuals = (y_te - y_hat_te).values

    evals = {
        "alpha": float(pipe.named_steps["ridge"].alpha_),
        "r2_holdout": float(pipe.score(X_te, y_te)),
        "smape_holdout": float(smape_numpy(y_te, y_hat_te)),
        "mae_holdout": float(np.mean(np.abs(y_te - y_hat_te))),
    }
    model = Pipeline([("scaler", StandardScaler()),
                      ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 21), cv=5))]).fit(X_all, y_all)
    return model, evals, X_cols, residuals

def scenario_df(df, income_mul=1.0, income_add=0.0,
                urb_add=0.0, hybrid_mul=1.0, fossil_rel_drop=0.0):
    S = df.copy()
    S["avg_yearly_income_k"] = S["avg_yearly_income_k"] * income_mul + income_add
    S["urbanization"] = np.clip(S["urbanization"] + urb_add, 1, 5)
    S["p_hybrid"]     = np.clip(S["p_hybrid"] * hybrid_mul, 0, 1)
    S["p_gasoline"]   = np.clip(S["p_gasoline"] * (1 - fossil_rel_drop), 0, 1)
    S["p_diesel"]     = np.clip(S["p_diesel"]   * (1 - fossil_rel_drop), 0, 1)
    return S

def pc4s_for_municipality(gm_query: str, conv_df: pd.DataFrame) -> pd.DataFrame:
    q = gm_query.strip().lower()
    m = conv_df[conv_df["gm_name"].str.lower().str.contains(q, na=False)].copy()
    return m[["pc4","gm_name"]].drop_duplicates()

#Driver Explanation
def build_driver_explanation(mun_df: pd.DataFrame, pc4: int, top_k: int = 4) -> Tuple[str, pd.DataFrame]:
    driver_cols = ["avg_yearly_income_k","urbanization","p_hybrid","p_gasoline","p_diesel",
                   "chargers_per_10k","stations_per_10k"]
    driver_labels = {
        "avg_yearly_income_k": "Income",
        "urbanization": "Urbanization",
        "p_hybrid": "Hybrid share",
        "p_gasoline": "Gasoline share",
        "p_diesel": "Diesel share",
        "chargers_per_10k": "Chargers / 10k",
        "stations_per_10k": "Stations / 10k",
    }
    positive_good = {"avg_yearly_income_k", "urbanization", "p_hybrid", "chargers_per_10k", "stations_per_10k"}

    row = mun_df.loc[mun_df["pc4"] == pc4]
    if row.empty:
        return f"PC4 {pc4}: not found in municipality selection.", pd.DataFrame()
    row = row.iloc[0]

    mun_mean = mun_df[driver_cols].mean(numeric_only=True)
    mun_std  = mun_df[driver_cols].std(ddof=0, numeric_only=True).replace(0, 1e-9)
    z = (row[driver_cols] - mun_mean) / mun_std

    table_rows = []
    for c in driver_cols:
        name = driver_labels[c]
        z_val = float(z[c])
        arrow = "↑" if ((z_val > 0) == (c in positive_good)) else "↓"
        table_rows.append({"Driver": name, "z-score": z_val, "Direction": arrow})

    table = pd.DataFrame(table_rows).sort_values(by="z-score", key=np.abs, ascending=False).reset_index(drop=True)
    parts = [f"{table.loc[i, 'Direction']} {table.loc[i, 'Driver']}" for i in range(min(top_k, len(table)))]
    text = (f"PC4 {pc4}: vs municipality median — " + ", ".join(parts) +
            f". (EV usage: {row['p_electric']:.3f}, chargers/10k: {row['chargers_per_10k']:.2f})")
    return text, table

#Renderer
def render_ev(bq_client, project_id: str, ocm_api_key: str):
    _inject_ev_css_once()

    with st.spinner("Loading regional features…"):
        regions = bq_read_regions(bq_client, project_id)
        conv    = bq_read_conversion(bq_client, project_id)

    with st.spinner("Loading OpenChargeMap (NL)…"):
        try:
            ocm_raw = fetch_ocm_nl(ocm_api_key or "")
            ocm_flat = flatten_ocm(ocm_raw)
            ocm_pc4  = ocm_pc4_aggregate(ocm_flat)
        except Exception as ex:
            st.warning(f"OpenChargeMap fetch failed: {ex}")
            ocm_pc4 = pd.DataFrame(columns=["pc4","n_stations","n_operational","operational_share",
                                            "n_ports","fast_ports","ultra_fast_ports",
                                            "total_power_kw","avg_power_kw"])

    #NATIONAL ANALYSIS
    st.markdown('<div class="section-title" style="font-size: 2rem; text-align: center">National analysis</div>', unsafe_allow_html=True)
    R, profile, best_k, sil, pca_df = build_clusters(regions.copy())
    R = ev_potential_and_opportunity(R)
    model, evals, X_cols, residuals = fit_ridge_structural(R.copy())

    #TOP10 and Map
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="section-title">Top 10 EV Potential</div>', unsafe_allow_html=True)

        top10_nat = (
            R[["pc4","EV_Potential_0_100","Opportunity","p_electric","avg_yearly_income_k","urbanization"]]
            .sort_values(["EV_Potential_0_100","Opportunity"], ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        drop_cols = [c for c in top10_nat.columns
                    if ("ev" in c.lower() and "potential" in c.lower()) or ("urban" in c.lower())]
        top10_nat = top10_nat.drop(columns=drop_cols, errors="ignore")
        top10_nat = _rename_columns_nicely(top10_nat)

        st.dataframe(top10_nat, use_container_width=True, hide_index=True, height=df_full_height(top10_nat))

    with c2:
        st.markdown('<div class="section-title">Charging Map</div>', unsafe_allow_html=True)

        _center_lat = 52.1326
        _center_lon = 5.2913
        _start_zoom = 7 

        try:
            _ocm_key = OCM_API_KEY if "OCM_API_KEY" in globals() else None
        except Exception:
            _ocm_key = None

        # Se vuoi filtrare per operatore imposta qui, altrimenti None
        _operator_id = None 

        _src = "https://map.openchargemap.io/?mode=embedded"

        params = []

        params.append(f"latitude={_center_lat}")
        params.append(f"longitude={_center_lon}")
        params.append(f"zoom={_start_zoom}")
        params.append(f"lat={_center_lat}")
        params.append(f"lon={_center_lon}")

        if _operator_id:
            params.append(f"operatorid={_operator_id}")
        if _ocm_key:
            params.append(f"key={_ocm_key}")
        if params:
            _src += "&" + "&".join(params)

        st.components.v1.html(
            f'<iframe src="{_src}" allow="geolocation" frameborder="0" width="100%" height="500px"></iframe>',
            height=520
        )
        
    #OPTIONALS: Clusters and Rsiduals
    e1, e2 = st.columns([1, 1])

    #Expander Cluster
    with e1:
        with st.expander("CLUSTERS", expanded=False):
            fig = go.Figure()
            for raw_name, grp in pca_df.groupby("cluster_label"):
                m = _re_label.search(r"(\d+)", str(raw_name))
                legend_name = f"Cluster {int(m.group(1)) - 1}" if m else str(raw_name)

                fig.add_trace(go.Scatter(
                    x=grp["x"], y=grp["y"],
                    mode="markers",
                    name=legend_name,
                    text=grp["pc4"].astype(str),
                    hovertemplate="PC4: %{text}<br>PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<extra></extra>"
                ))
            fig.update_layout(
                title=f"Clusters in PCA space (k={best_k}, silhouette={sil:.3f})",
                xaxis_title="PCA1", yaxis_title="PCA2",
                height=420, legend_title="Cluster"
            )
            st.plotly_chart(fig, use_container_width=True)

            prof_tbl = profile.rename(columns={"ev_cluster":"cluster_id"})
            prof_tbl = _rename_columns_nicely(prof_tbl) 

            st.dataframe(prof_tbl, use_container_width=True, hide_index=True, height=df_full_height(prof_tbl))

            render_accordion(
                " How to read clusters",
                ("\n\n - Cluster 0:  Low-income rural areas with limited EV adoption potential\n\n"
                "- Cluster 1: Urban, average-income regions with untapped EV growth opportunity\n\n"
                "- Cluster 2: Wealthy low-density areas already leading in EV adoption")
            )

    #Expander Residuals
    with e2:
        with st.expander("RESIDUALS", expanded=False):
            try:
                fig_res, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(residuals, bins=30, kde=True, ax=ax)
                ax.set_title("Hold-out residuals (y_true - y_pred)")
                ax.set_xlabel("Residual"); ax.set_ylabel("Count")
                fig_res.tight_layout()
                st.pyplot(fig_res, use_container_width=True)
            except Exception:
                st.info("Residuals plot unavailable.")

            k1, k2, k3 = st.columns(3)
            with k1:
                render_kpi_card(
                    title="R²",
                    value_str=f"{evals['r2_holdout']:.3f}",
                    tooltip=(
                        "Goodness of fit on unexpected data. 1.00 = perfect, 0.00 = no improvement vs average."
                    )
                )
            with k2:
                render_kpi_card(
                    title="SMAPE",
                    value_str=f"{evals['smape_holdout']:.2f}%",
                    tooltip=(
                        "Symmetric MAPE: Symmetric mean percentage error. The lower the better."
                    )
                )
            with k3:
                render_kpi_card(
                    title="MAE",
                    value_str=f"{evals['mae_holdout']:.4f}",
                    tooltip=(
                        "Mean Absolute Error: absolute average difference between predicted and true (scale 0–1)."
                    )
                )

            # Spiegazione residuals
            render_accordion(
                "What are residuals?",
                ("They are the differences <code>(y_true − y_pred)</code> on the validation set. "
                 "If the distribution is 0-centered and symmetric → undistorted pattern. "
                 "Code/asymmetries indicate systematic errors. Use them with R²/SMAPE/MAE to evaluate quality and bias.")
            )

    st.divider()

    #MUNICIPALITY ANALYSIS
    st.markdown('<div class="section-title" style="font-size: 2rem; text-align: center">Municipality analysis</div>', unsafe_allow_html=True)

    #Municipality selection and tab
    gm_list = sorted(conv["gm_name"].dropna().unique().tolist())
    default_idx = gm_list.index("Utrecht") if "Utrecht" in gm_list else 0

    cL, cC, cR = st.columns([1, 2, 1])  
    with cC:
        selected_gm = st.selectbox(
            "Select municipality",
            gm_list,
            index=default_idx,
            key="gm_select"          
        )
    pc4_map = pc4s_for_municipality(selected_gm, conv)
    mun_pc4s = set(pc4_map["pc4"].tolist())

    R_valid = R.dropna(subset=["pc4"]).copy()
    R_valid["pc4"] = R_valid["pc4"].astype(int)
    R_mun = R_valid[R_valid["pc4"].isin(mun_pc4s)].copy()

    mun = R_mun.merge(ocm_pc4, on="pc4", how="left")
    mun["chargers_per_10k"] = np.where(
        mun["inhabitants_total"] > 0,
        (mun["n_ports"].fillna(0) / mun["inhabitants_total"]) * 10000.0,
        np.nan
    )
    mun["stations_per_10k"] = np.where(
        mun["inhabitants_total"] > 0,
        (mun["n_stations"].fillna(0) / mun["inhabitants_total"]) * 10000.0,
        np.nan
    )

    st.markdown(f'<div class="section-title">Summary – {selected_gm}</div>', unsafe_allow_html=True)

    kpi = pd.DataFrame({
                "pc4_count": [len(mun)],
                "population_total": [mun["inhabitants_total"].sum(skipna=True)],
                "ev_usage_mean": [mun["p_electric"].mean(skipna=True)],
                "stations_total": [mun["n_stations"].sum(skipna=True)],
                "ports_total": [mun["n_ports"].sum(skipna=True)],
                "fast_ports_total": [mun["fast_ports"].sum(skipna=True)],
                "ultra_ports_total": [mun["ultra_fast_ports"].sum(skipna=True)],
                "chargers_per_10k_mean": [mun["chargers_per_10k"].mean(skipna=True)],
            })
    kpi = _rename_columns_nicely(kpi)
    st.dataframe(kpi, use_container_width=True, hide_index=True, height=df_full_height(kpi))

    #Top-5 by EV usage and Drivers
    b1, b2 = st.columns([1, 1])
    with b1:
        st.markdown('<div class="section-title">Top 5 PC4 by EV usage</div>', unsafe_allow_html=True)
        top5_usage = (
        mun[[
            "pc4","cluster_label","p_electric","avg_yearly_income_k","urbanization",
            "n_stations","n_ports","fast_ports","ultra_fast_ports",
            "chargers_per_10k","stations_per_10k","inhabitants_total",
            "cluster_confidence","silhouette_i"
        ]]
        .sort_values("p_electric", ascending=False)
        .head(5).reset_index(drop=True)
        )
        # apply requested removals and rename labels
        drop_cols = [c for c in top5_usage.columns if c.lower() in {"cluster_label","cluster label"} or ("urban" in c.lower())]
        top5_usage = top5_usage.drop(columns=drop_cols, errors="ignore")
        top5_usage = _rename_columns_nicely(top5_usage)

        st.dataframe(top5_usage, use_container_width=True, hide_index=True, height=df_full_height(top5_usage))

        # Scatterplot of clusters for the selected municipality (below the table)
        try:
            _ev = next((c for c in mun.columns if "ev" in c.lower() and ("share" in c.lower() or "electric" in c.lower() or "rate" in c.lower())), None)
            _inc = next((c for c in mun.columns if "income" in c.lower()), None)
            _cl  = next((c for c in mun.columns if "cluster" in c.lower()), None)
            if _ev and _inc:
                fig, ax = plt.subplots(figsize=(7,5))
                _df = mun.dropna(subset=[_inc, _ev]).copy()
                if _cl and _cl in _df.columns:
                    for k, sub in _df.groupby(_cl):
                        ax.scatter(sub[_inc], sub[_ev], alpha=0.85, label=str(k))
                    ax.legend(title=_titleize(_cl), frameon=False)
                else:
                    ax.scatter(_df[_inc], _df[_ev], alpha=0.85)
                ax.set_xlabel(_titleize(_inc)); ax.set_ylabel(_titleize(_ev))
                ax.set_title(f"Clusters – {selected_gm}")
                ax.grid(True, linewidth=0.5, alpha=0.4)
                st.pyplot(fig, clear_figure=True)
        except Exception as _e:
            st.caption(f"(Scatterplot unavailable: {type(_e).__name__})")

    with b2:
        st.markdown('<div class="section-title">Explain drivers for a PC4</div>', unsafe_allow_html=True)

        pc4_choices = sorted(mun["pc4"].unique().tolist())
        if pc4_choices:
            chosen_pc4 = st.selectbox("Pick a PC4", pc4_choices, index=0)
            txt, drv_tbl = build_driver_explanation(mun, chosen_pc4, top_k=4)
            st.write(txt)
            if not drv_tbl.empty:
                st.dataframe(drv_tbl, use_container_width=True, hide_index=True, height=df_full_height(drv_tbl))
            

    st.divider()

    #WHAT-IF SCENARIOS
    st.markdown('<div class="section-title" style="font-size: 2rem; text-align: center">What-if Scenarios</div>', unsafe_allow_html=True)

    #Sliders
    c_income, c_urb, c_hyb, c_fossil = st.columns([1, 1, 1, 1])
    with c_income:
        income_mul = st.slider("Income multiplier", 0.8, 1.5, 1.10, 0.01)
    with c_urb:
        urb_add = st.slider("Urbanization +", -0.5, 0.5, 0.05, 0.01)
    with c_hyb:
        hybrid_mul = st.slider("Hybrid ×", 0.5, 3.0, 2.0, 0.1)
    with c_fossil:
        fossil_rel_drop = st.slider("Fossil relative drop", 0.0, 0.5, 0.10, 0.01)

    Xc = ["avg_yearly_income_k", "urbanization", "p_hybrid", "p_gasoline", "p_diesel"]
    mun_pred = mun.dropna(subset=Xc).copy()
    if mun_pred.empty:
        st.info("Insufficient data to run the scenario on this municipality.")
        return

    #Baseline and Scenario
    mun_pred["p_electric_hat"] = model.predict(mun_pred[Xc])
    S_mun = scenario_df(
        mun_pred,
        income_mul=income_mul, urb_add=urb_add,
        hybrid_mul=hybrid_mul, fossil_rel_drop=fossil_rel_drop
    )
    S_mun["p_electric_hat_scn"]   = model.predict(S_mun[Xc])
    S_mun["delta_p_electric_hat"] = S_mun["p_electric_hat_scn"] - mun_pred["p_electric_hat"]

    #Table
    tbl = (
        S_mun[["pc4", "p_electric", "p_electric_hat", "p_electric_hat_scn",
               "delta_p_electric_hat", "chargers_per_10k"]]
        .copy()
    )
    tbl = tbl.sort_values("delta_p_electric_hat", ascending=False).head(5).reset_index(drop=True)

    tbl = tbl.rename(columns={
        "pc4": "PC4",
        "p_electric": "Current EV share (%)",
        "p_electric_hat": "Baseline prediction EV share (%)",
        "p_electric_hat_scn": "Scenario prediction EV share (%)",
        "delta_p_electric_hat": "Δ EV Share (%)",
        "chargers_per_10k": "Chargers / 10 000 inhab."
    })

    for col in [
        "Current EV share (%)",
        "Baseline prediction EV share (%)",
        "Scenario prediction EV share (%)",
        "Δ EV Share (%)",
    ]:
        tbl[col] = (tbl[col] * 100).round(1)

    st.markdown('<div class="section-title">Top 5 PC4 by Δ EV Share (scenario − baseline)</div>', unsafe_allow_html=True)
    
    st.dataframe(
        tbl,
        use_container_width=True,
        hide_index=True,
        height=df_full_height(tbl)
    )

    #Success Indicators
    if not tbl.empty:
        best_row = tbl.iloc[0]
        best_pc4 = int(best_row["PC4"])
        baseline_pct = float(best_row["Baseline prediction EV share (%)"])
        scenario_pct = float(best_row["Scenario prediction EV share (%)"])
        delta_pct    = float(best_row["Δ EV Share (%)"])  # differenza in punti percentuali

        st.markdown('<div class="section-title">Predicted EV Conversion: Current vs Scenario</div>', unsafe_allow_html=True)

        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            colA, colB, colC = st.columns(3, gap="large")

            def _kpi(title: str, value_str: str, tooltip: str):
                if "render_kpi_card" in globals():
                    try:
                        render_kpi_card(title=title, value_str=value_str, tooltip=tooltip)
                        return
                    except Exception:
                        pass
                # Fallback: simple metric
                st.metric(title, value_str)

            with colA:
                _kpi(
                    title="Baseline prediction",
                    value_str=f"{baseline_pct:.1f} %",
                    tooltip=(
                        f"EV quota foreseen by the model for the current state of the drivers "
                        f"(PC4 {best_pc4}). It is the forecast ‘baseline’ without scenario changes."
                    )
                )
            with colB:
                _kpi(
                    title="Scenario prediction",
                    value_str=f"{scenario_pct:.1f} %",
                    tooltip=(
                        f"Previous EV quota by applying what-if parameters "
                        f"(PC4 {best_pc4}). Shows the effect of the assumed scenario."
                    )
                )
            with colC:
                _kpi(
                    title="EV Share Change ",
                    value_str=f"{delta_pct:.1f}",
                    tooltip=(
                        "Difference in percentage between Scenario and Baseline "
                        "(Scenario − Baseline). Positive value = expected increase."
                    )
                )
