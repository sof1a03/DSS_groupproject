# from the github repository https://github.com/statistiekcbs/CBS-Open-Data-v4/tree/master/Python  

import os, datetime
import pandas as pd
from sqlalchemy import create_engine
from .cbs_utils import get_odata

TABLE = "https://beta-odata4.cbs.nl/CBS/84120NED"
# Example: pick one measure code via $filter (same style as CBS examples)
FILTER = "BelastingenEnWettelijkePremies eq 'A045081'"

def _parse_period(period: str):
    # YYYY, YYYYMMNN, YYYYKWQQ → date + (year, freq, count) similar to CBS tutorials
    if period.isdigit() and len(period) == 4:  # Year
        return datetime.date(int(period), 1, 1), int(period), "Y", None
    # Quarter: 2019KW01..04
    if "KW" in period:
        y = int(period[:4]); q = int(period[-2:])
        return datetime.date(y, 3*(q-1)+1, 1), y, "Q", q
    # Month: 2019MM01..12
    if "MM" in period:
        y = int(period[:4]); m = int(period[-2:])
        return datetime.date(y, m, 1), y, "M", m
    return None, None, None, None

def run():
    obs = get_odata(f"{TABLE}/Observations?$filter={FILTER}&$select=Perioden,Value")

    # Derive date parts
    parsed = obs["Perioden"].apply(_parse_period)
    obs["date"]      = parsed.apply(lambda x: x[0])
    obs["year"]      = parsed.apply(lambda x: x[1])
    obs["frequency"] = parsed.apply(lambda x: x[2])
    obs["count"]     = parsed.apply(lambda x: x[3])

    df = obs.rename(columns={"Perioden": "periodcode", "Value": "value"})[
        ["periodcode", "value", "year", "frequency", "count", "date"]
    ]

    url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@" \
          f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(url, pool_pre_ping=True)
    df.to_sql("tourist_tax", engine, schema="cbs", if_exists="append",
              index=False, method="multi", chunksize=5000)

    print(f"[ETL] Loaded {len(df)} time series rows")
