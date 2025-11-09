# matching.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from google.cloud import bigquery
import json
from pyproj import Transformer
from scipy.stats import norm

W_INTEREST_BODY  = 0.6
W_INTEREST_FUEL  = 0.4
W_FINAL_INTEREST = 0.7
W_FINAL_AFFORD   = 0.3
TOP_QUANTILE     = 0.8  

BODY_CLASSES = ["compact", "medium", "large", "suv", "mpv", "sports"]
FUEL_CLASSES = ["gasoline", "diesel", "electric", "hybrid"]


# Colori (RGBA) for heatmap layers
HEAT_COLORS = {
    "Very High": [153, 0, 0, 180],      
    "High":      [220, 20, 60, 160],    
    "Medium":    [255, 179, 71, 140],   
    "Low":       [255, 239, 153, 120],  
    "Very Low":  [0, 0, 0, 0],          
}

def map_fuel_column(fuel_str: str) -> str:
    f = str(fuel_str).strip().lower()
    if ("benzine" in f) or ("gas" in f) or ("petrol" in f):
        return "p_gasoline"
    if "diesel" in f:
        return "p_diesel"
    if ("electric" in f) or ("ev" in f):
        return "p_electric"
    if "hybrid" in f:
        return "p_hybrid"
    return "p_gasoline"

#Labels dataset 
MATCH_COLS_DICT = {
    "Segment": {
        "Compact": "p_compact",
        "Medium": "p_medium",
        "Large": "p_large",
        "SUV": "p_suv",
        "MPV": "p_mpv",
        "Sports": "p_sports",
    },
    "Fuel": {
        "Benzine": "p_gasoline",
        "Diesel": "p_diesel",
        "Elektrisch": "p_electric",
        "Hybride": "p_hybrid",
    },
    "Weight": {
        "0,850": "p_car_weight_0_to_850",
        "851,1150": "p_car_weight_851_to_1150",
        "1151,1500": "p_car_weight_1151_to_1500",
        "1501,10000": "p_car_weight_1501_to_10000",
    }
}

#BigQuery fetch
def bq_read_regions(client: bigquery.Client, project_id: str) -> pd.DataFrame:
    q = f"SELECT * FROM `{project_id}.REGIONAL.REGIONAL_2`"
    return client.query(q).to_dataframe()

def bq_read_car(client: bigquery.Client, project_id: str, brand: str, model: str) -> pd.DataFrame:
    q = f"""
        SELECT *
        FROM `{project_id}.RDW.rdw_classified_final`
        WHERE brand = @b AND model = @m
        LIMIT 1
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("b", "STRING", brand),
            bigquery.ScalarQueryParameter("m", "STRING", model),
        ]
    )
    return client.query(q, job_config=cfg).to_dataframe()

def bq_read_same_class(client: bigquery.Client, project_id: str, body_class: str) -> pd.DataFrame:
    q = f"""
        SELECT brand, model, body_class, count_2024, price_z_score, seats_median, pw_ratio_median
        FROM `{project_id}.RDW.rdw_classified_final`
        WHERE body_class = @bc
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("bc", "STRING", body_class)]
    )
    return client.query(q, job_config=cfg).to_dataframe()

# Core: Compute ranking PC4
def compute_pc4_ranking(
    regions: pd.DataFrame, car_row: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = ["pc4", "inhabitants_total", "avg_yearly_income_k", "std_avg_yearly_income_k"]
    for c in required_cols:
        if c not in regions.columns:
            raise ValueError(f"Missing column '{c}' in REGIONAL.REGIONAL_2")

    #Body/fuel col
    body_val = str(car_row.get("body_class", "")).strip()
    body_col = f"p_{body_val.lower()}"
    fuel_col = map_fuel_column(car_row.get("fuel_types_primary", ""))

    for c in [body_col, fuel_col]:
        if c not in regions.columns:
            raise ValueError(f"Missing column '{c}' in REGIONAL.REGIONAL_2 (richiesta per body/fuel)")

    R = regions.dropna(subset=[body_col, fuel_col, "avg_yearly_income_k", "inhabitants_total"]).copy()

    #Interest (mix)
    R["interest_score"] = (W_INTEREST_BODY * R[body_col]) + (W_INTEREST_FUEL * R[fuel_col])

    #Affordability
    price_z = pd.to_numeric(car_row.get("price_z_score"), errors="coerce")
    income_z = pd.to_numeric(R["std_avg_yearly_income_k"], errors="coerce")
    d = np.abs(price_z - income_z)
    median_d = float(np.nanmedian(d)) if np.isfinite(d).any() else 0.0
    k = (np.log(2.0) / median_d) if median_d > 0 else 1.0
    R["affordability_fit"] = np.exp(-k * d)

    #Final score
    R["final_score"] = (W_FINAL_INTEREST * R["interest_score"]) + (W_FINAL_AFFORD * R["affordability_fit"])

    #Aggregazione pesata per pc4
    ranked_pc4 = (
        R.groupby("pc4", as_index=False)
         .apply(lambda g: pd.Series({
             "interest_score": np.average(g["interest_score"], weights=g["inhabitants_total"]),
             "affordability_fit": np.average(g["affordability_fit"], weights=g["inhabitants_total"]),
             "final_score": np.average(g["final_score"], weights=g["inhabitants_total"]),
         }))
         .sort_values("final_score", ascending=False)
         .reset_index(drop=True)
    )

    #Binning
    N_BINS = 5
    group_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    ranked_pc4["score_group_label"] = pd.qcut(ranked_pc4["final_score"], q=N_BINS, labels=group_labels)

    return ranked_pc4, R

#Success Indicators
def compute_popularity_niche(same_class: pd.DataFrame, car_row: pd.Series) -> Dict[str, float]:
    #Popularity
    selected_sales = float(pd.to_numeric(car_row.get("count_2024"), errors="coerce") or 0.0)
    best_sales = float(pd.to_numeric(same_class["count_2024"], errors="coerce").max() or 0.0)
    popularity = (selected_sales / best_sales) * 100 if best_sales > 0 else 0.0

    #Niche Score
    df = same_class.copy()
    df["price_pct"] = pd.to_numeric(df["price_z_score"], errors="coerce").rank(pct=True)
    df["sales_pct"] = pd.to_numeric(df["count_2024"], errors="coerce").rank(pct=True)

    car_key = (str(car_row.get("brand")), str(car_row.get("model")))
    row = df[(df["brand"] == car_key[0]) & (df["model"] == car_key[1])]
    if row.empty: #Fallback
        car_price_pct = float(pd.Series([car_row.get("price_z_score")], dtype="float64").rank(pct=True).iloc[0])
        car_sales_pct = 1.0
    else:
        car_price_pct = float(row["price_pct"].iloc[0])
        car_sales_pct = float(row["sales_pct"].iloc[0])

    W_PRICE, W_SALES = 0.6, 0.4
    niche = 100.0 * (W_PRICE * car_price_pct + W_SALES * (1.0 - car_sales_pct))

    return {"popularity_score": float(popularity), "niche_score": float(niche)}


def choose_weight_bucket(mass_empty: float) -> str:
    if mass_empty <= 850:
        return "0,850"
    if mass_empty <= 1150:
        return "851,1150"
    if mass_empty <= 1500:
        return "1151,1500"
    return "1501,10000"

def compute_profile_match_and_radar(
    regions: pd.DataFrame, ranked_pc4: pd.DataFrame, car_row: pd.Series
) -> Dict[str, Any]:
    if ranked_pc4.empty:
        return {"profile_match": np.nan, "radar": None}

    best_pc4 = ranked_pc4.iloc[0]["pc4"]
    R_best = regions[regions["pc4"] == best_pc4].mean(numeric_only=True)

    #Car stats
    C_body = str(car_row.get("body_class", ""))
    C_fuel = str(car_row.get("fuel_types_primary", ""))
    C_weight = float(pd.to_numeric(car_row.get("mass_empty_median"), errors="coerce"))
    C_seats = float(pd.to_numeric(car_row.get("seats_median"), errors="coerce"))
    C_price_z = float(pd.to_numeric(car_row.get("price_z_score"), errors="coerce"))

    seg_col = MATCH_COLS_DICT["Segment"].get(C_body, MATCH_COLS_DICT["Segment"]["Compact"])
    fuel_key = "Benzine" if ("benz" in C_fuel.lower() or "petrol" in C_fuel.lower() or "gas" in C_fuel.lower()) else \
               "Diesel" if "diesel" in C_fuel.lower() else \
               "Elektrisch" if "ele" in C_fuel.lower() or "ev" in C_fuel.lower() else \
               "Hybride" if "hyb" in C_fuel.lower() else "Benzine"
    fuel_col = MATCH_COLS_DICT["Fuel"][fuel_key]
    weight_key = choose_weight_bucket(C_weight)
    weight_col = MATCH_COLS_DICT["Weight"][weight_key]

    if "avg_household_size" not in regions.columns:
        raise ValueError("Missing 'avg_household_size' in REGIONAL.REGIONAL_2")

    #Binning
    labels = [0.2, 0.4, 0.6, 0.8, 1.0]

    house_bins = list(regions["avg_household_size"].quantile([0.0,0.2,0.4,0.6,0.8,1.0]).values)
    seat_bins = [1.0, 2.6, 4.2, 5.8, 7.4, 9.0]

    R_seats_bucket = pd.cut([float(R_best["avg_household_size"])], bins=house_bins, labels=labels, include_lowest=True)[0]
    C_seats_bucket = pd.cut([C_seats], bins=seat_bins, labels=labels, include_lowest=True)[0]

    C_price_01 = float(norm.cdf(C_price_z)) if np.isfinite(C_price_z) else 0.5

    R_min = float(pd.to_numeric(regions["avg_yearly_income_k"], errors="coerce").min())
    R_max = float(pd.to_numeric(regions["avg_yearly_income_k"], errors="coerce").max())
    R_income_val = float(R_best["avg_yearly_income_k"])
    R_price_01 = (R_income_val - R_min) / (R_max - R_min) if (R_max > R_min) else 0.5
    R_price_01 = max(0.0, min(1.0, R_price_01))

    #Vector construction
    R_vec = np.array([
        float(R_best.get(seg_col, 0.0)),
        float(R_best.get(fuel_col, 0.0)),
        float(R_best.get(weight_col, 0.0)),
        float(R_seats_bucket),
        float(R_price_01),
    ])
    C_vec = np.array([
        1.0, 
        1.0,  
        1.0, 
        float(C_seats_bucket),
        float(C_price_01),
    ])

    #Cosine similarity
    denom = (np.linalg.norm(R_vec) * np.linalg.norm(C_vec))
    profile_match = float((np.dot(R_vec, C_vec) / denom) * 100.0) if denom > 0 else np.nan

    radar = {
        "categories": ["Segment", "Fuel", "Weight", "Seats", "Price"],
        "region_values": R_vec.tolist(),
        "car_values": C_vec.tolist(),
        "best_pc4": best_pc4,
        "region_income_norm": R_price_01,
        "car_price_norm": C_price_01,
    }
    return {"profile_match": profile_match, "radar": radar}

#Computation for Match&Map 
def compute_match_map_package(
    client: bigquery.Client, project_id: str, brand: str, model: str
) -> Dict[str, Any]:
    car_df = bq_read_car(client, project_id, brand, model)
    if car_df.empty:
        raise ValueError("Nessuna riga trovata per il veicolo selezionato.")
    car_row = car_df.iloc[0]

    regions = bq_read_regions(client, project_id)
    ranked_pc4, R_top = compute_pc4_ranking(regions, car_row)

    same_class = bq_read_same_class(client, project_id, str(car_row.get("body_class")))
    kpis = compute_popularity_niche(same_class, car_row)

    #Profile Match + radar
    prof = compute_profile_match_and_radar(regions, ranked_pc4, car_row)
    out = {
        "ranked_pc4": ranked_pc4,
        "kpis": kpis,
        "profile_match": prof["profile_match"],
        "radar": prof["radar"],
        "car": car_row,
    }
    return out


def bq_read_pc4_raw_geometry(client: bigquery.Client, project_id: str, pc4_list: list[int]) -> pd.DataFrame:
    if not pc4_list:
        return pd.DataFrame(columns=["pc4", "geometry"])

    q = f"""
    DECLARE pc4s ARRAY<INT64>;
    SET pc4s = @pc4s;

    SELECT
      CAST(postcode AS INT64) AS pc4,
      geometry
    FROM `{project_id}.MAP.GEO_PC4`
    WHERE CAST(postcode AS INT64) IN UNNEST(pc4s)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("pc4s", "INT64", pc4_list)]
    )
    return client.query(q, job_config=job_config).to_dataframe()

#Transformation from RD New (EPSG:28992) -> WGS84 (EPSG:4326)
_TRANSFORMER_RD_TO_WGS = Transformer.from_crs(28992, 4326, always_xy=True)

def _transform_coords_rd_to_wgs(coords):
    x, y = coords
    lon, lat = _TRANSFORMER_RD_TO_WGS.transform(float(x), float(y))
    return [lon, lat]

def _transform_geojson_rd_to_wgs(geom_obj: dict) -> dict:
    gtype = geom_obj.get("type")
    out = {"type": gtype}

    if gtype == "Point":
        out["coordinates"] = _transform_coords_rd_to_wgs(geom_obj["coordinates"])

    elif gtype == "LineString":
        out["coordinates"] = [_transform_coords_rd_to_wgs(c) for c in geom_obj["coordinates"]]

    elif gtype == "Polygon":
        out["coordinates"] = [
            [_transform_coords_rd_to_wgs(c) for c in ring]
            for ring in geom_obj["coordinates"]
        ]

    elif gtype == "MultiPolygon":
        out["coordinates"] = [
            [
                [_transform_coords_rd_to_wgs(c) for c in ring]
                for ring in poly
            ]
            for poly in geom_obj["coordinates"]
        ]
    else: #Fallback
        out = geom_obj

    return out

def build_pc4_heatmap_features(
    client: bigquery.Client,
    project_id: str,
    ranked_pc4: pd.DataFrame
) -> list[dict]:
    
    if ranked_pc4 is None or ranked_pc4.empty:
        return []

    if "score_group_label" not in ranked_pc4.columns:
        labels = ["Very Low", "Low", "Medium", "High", "Very High"]
        ranked_pc4 = ranked_pc4.copy()
        ranked_pc4["score_group_label"] = pd.qcut(
            ranked_pc4["final_score"], q=5, labels=labels
        )

    pc4_list = (
        ranked_pc4["pc4"].dropna().astype(int).unique().tolist()
    )
    if not pc4_list:
        return []

    geo_df = bq_read_pc4_raw_geometry(client, project_id, pc4_list)
    if geo_df.empty:
        return []

    merged = pd.merge(
        geo_df, ranked_pc4[["pc4", "final_score", "score_group_label"]],
        on="pc4", how="left"
    )

    features: list[dict] = []
    for _, row in merged.iterrows():
        geom_raw = row.get("geometry")
        if not isinstance(geom_raw, str) or not geom_raw.strip():
            continue

        geom_str = geom_raw.strip()

        try:
            geom_obj = json.loads(geom_str)
        except Exception:
            geom_str2 = geom_str.strip('"').strip("'").lstrip("{").rstrip("}")
            try:
                geom_obj = json.loads(geom_str2)
            except Exception:
                continue

        try:
            geom_wgs = _transform_geojson_rd_to_wgs(geom_obj)
        except Exception:
            continue

        label = str(row.get("score_group_label", "Very Low"))
        color = HEAT_COLORS.get(label, [0, 0, 0, 0])

        features.append({
            "type": "Feature",
            "geometry": geom_wgs,
            "properties": {
                "pc4": int(row["pc4"]),
                "final_score": round(float(row["final_score"]), 2) if pd.notnull(row["final_score"]) else None,
                "group_label": label,
                "fillColor": color,
            },
        })

    return features
