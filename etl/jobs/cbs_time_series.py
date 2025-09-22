# etl/jobs/cbs_time_series.py
import os, datetime
import pandas as pd
import cbsodata
from sqlalchemy import create_engine

TABLE_ID = "84120NED"
FILTER = "BelastingenEnWettelijkePremies eq 'A045081'"

def _pick_time_col(cols):
    for cand in ("Perioden", "Periods"):
        if cand in cols:
            return cand
    for c in cols:
        if c.lower().startswith("period"):
            return c
    raise KeyError(f"No time column found (Perioden/Periods). Columns: {list(cols)}")

def _parse_period(p):
    if not isinstance(p, str):
        return None, None, None, None
    if p.isdigit() and len(p) == 4:
        y = int(p); return datetime.date(y,1,1), y, "Y", None
    if "KW" in p:
        y, q = int(p[:4]), int(p[-2:])
        return datetime.date(y, 3*(q-1)+1, 1), y, "Q", q
    if "MM" in p:
        y, m = int(p[:4]), int(p[-2:])
        return datetime.date(y, m, 1), y, "M", m
    return None, None, None, None

def run():
    rows = cbsodata.get_data(TABLE_ID, filters=FILTER)
    df = pd.DataFrame(rows)

    time_col = _pick_time_col(df.columns)
    parsed = df[time_col].apply(_parse_period)

    df["date"]      = parsed.apply(lambda x: x[0])
    df["year"]      = parsed.apply(lambda x: x[1])
    df["frequency"] = parsed.apply(lambda x: x[2])
    df["count"]     = parsed.apply(lambda x: x[3])

    # rename to your DB columns; value column is 'Value' in v3
    df = df.rename(columns={time_col: "periodcode", "Value": "value"})[
        ["periodcode", "value", "year", "frequency", "count", "date"]
    ].dropna(subset=["value"])

    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    engine = create_engine(url, pool_pre_ping=True)
    df.to_sql("tourist_tax", engine, schema="cbs",
              if_exists="append", index=False, method="multi", chunksize=5000)

    print(f"[ETL] Time series: loaded {len(df)} rows (time='{time_col}' → 'periodcode')")
