import os
import pandas as pd
import cbsodata
from sqlalchemy import create_engine

TABLE_ID = "83765NED"                    # Kerncijfers wijken en buurten 2017
PERIOD_DEFAULT = os.getenv("WB_PERIOD", "2017")  # set via env if you like

def _pick_measure_col(df, meta):
    preferred = [
        "AantalInwoners_5",
        "Bevolkingsdichtheid_33",        # note: index may differ by table year; we detect anyway below
        "GemiddeldInkomenPerInwoner_66",
    ]
    for p in preferred:
        if p in df.columns:
            return p
    if meta is not None and not meta.empty and {"Key","Type"}.issubset(meta.columns):
        keys = meta.loc[meta["Type"].str.lower().eq("measure"), "Key"]
        for k in keys:
            if k in df.columns:
                return k
    for c in df.columns:                 # numeric fallback
        if c not in {"WijkenEnBuurten","RegioS","RegioNaam","RegioCode"} and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise KeyError(f"No suitable measure column found. Columns: {list(df.columns)}")

def run():
    # 1) Full typed dataset; subset in pandas
    data = pd.DataFrame(cbsodata.get_data(TABLE_ID))

    # 2) Region code column
    if "WijkenEnBuurten" not in data.columns:
        if "RegioS" in data.columns:
            data = data.rename(columns={"RegioS": "WijkenEnBuurten"})
        else:
            raise KeyError(f"'WijkenEnBuurten' (or RegioS) not found. Columns: {list(data.columns)}")

    # 3) Metadata for human-readable measure title (optional)
    try:
        meta = pd.DataFrame(cbsodata.get_meta(TABLE_ID, "DataProperties"))
        title_map = dict(zip(meta.get("Key", []), meta.get("Title", [])))
    except Exception:
        meta, title_map = pd.DataFrame(), {}

    # 4) Pick one present measure
    measure_col = _pick_measure_col(data, meta)
    measure_title = title_map.get(measure_col, measure_col)

    # 5) Build your long-format rows; **no time column in this table → use constant**
    df = data[["WijkenEnBuurten", measure_col]].copy()
    df["Perioden"] = PERIOD_DEFAULT
    df = df.rename(columns={measure_col: "Value"})
    df["MeasureName"] = measure_title
    df = df[["WijkenEnBuurten", "Perioden", "MeasureName", "Value"]].dropna(subset=["Value"])

    # 6) Load to Postgres
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    engine = create_engine(url, pool_pre_ping=True)
    df.to_sql("wb_observations", engine, schema="cbs", if_exists="append",
              index=False, method="multi", chunksize=5000)
    print(f"[ETL] WB: loaded {len(df)} rows (Perioden='{PERIOD_DEFAULT}', measure='{measure_title}')")
