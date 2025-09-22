# from the github repository https://github.com/statistiekcbs/CBS-Open-Data-v4/tree/master/Python 
import os
import pandas as pd
from sqlalchemy import create_engine
from .cbs_utils import get_odata

SPECIAL_NULLS = {99997: None, 99995: None, 99991: None}
TABLE = "https://beta-odata4.cbs.nl/CBS/83765NED"

def run():
    # Observations (demo limit with $top similar to their “basics” example)
    obs = get_odata(f"{TABLE}/Observations?$top=5000&$select=WijkenEnBuurten,Perioden,Value,Measure,MeasureGroupId")

    # Metadata
    codes  = get_odata(f"{TABLE}/MeasureCodes?$select=Identifier,Title")
    groups = get_odata(f"{TABLE}/MeasureGroups?$select=Id,Title")

    # Joins to get human-friendly titles
    obs = obs.merge(codes, left_on="Measure", right_on="Identifier", how="left")
    obs = obs.merge(groups, left_on="MeasureGroupId", right_on="Id", how="left", suffixes=("", "_Group"))

    # Clean up like CBS examples
    obs["WijkenEnBuurten"] = obs["WijkenEnBuurten"].str.strip()
    obs = obs.replace(SPECIAL_NULLS)

    df = obs[["WijkenEnBuurten", "Perioden", "Value", "Title"]] \
            .rename(columns={"Title": "MeasureName"})

    url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@" \
          f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(url, pool_pre_ping=True)

    # Keep column names exactly as in DB DDL (quoted mixed-case)
    df.to_sql("wb_observations", engine, schema="cbs", if_exists="append",
              index=False, method="multi", chunksize=5000)

    print(f"[ETL] Loaded {len(df)} WB rows")
