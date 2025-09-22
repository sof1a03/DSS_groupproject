# from the github repository https://github.com/statistiekcbs/CBS-Open-Data-v4/tree/master/Python
import pandas as pd
import requests

HEADERS = {"User-Agent": "dss-dashboard-etl/1.0"}

def get_odata(url: str) -> pd.DataFrame:
    """Fetch all pages from a CBS OData v4 endpoint into a DataFrame (handles @odata.nextLink)."""
    frames = []
    next_url = url
    while next_url:
        resp = requests.get(next_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        frames.append(pd.DataFrame(payload.get("value", [])))
        next_url = payload.get("@odata.nextLink")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

 