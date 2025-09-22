# from the github repository https://github.com/statistiekcbs/CBS-Open-Data-v4/tree/master/Python 

import os, requests
import geopandas as gpd
from shapely.geometry import shape
from sqlalchemy import create_engine

WFS = (
  "https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/"
  "wfs?request=GetFeature&service=WFS&version=2.0.0&"
  "typeName=cbs_gemeente_2017_gegeneraliseerd&outputFormat=json"
)

def run():
    r = requests.get(WFS, timeout=60); r.raise_for_status()
    features = r.json().get("features", [])
    rows = []
    for f in features:
        p = {k.strip(): v for k, v in f.get("properties", {}).items()}
        rows.append({"statcode": p.get("statcode"), "statnaam": p.get("statnaam"),
                     "geometry": shape(f.get("geometry"))})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@" \
          f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(url, pool_pre_ping=True)
    gdf.to_postgis("gemeente_2017", engine, schema="cbs", if_exists="replace", index=False)

    print(f"[ETL] Loaded {len(gdf)} geometries")
