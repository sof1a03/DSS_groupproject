import time
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "dss-dashboard-etl/1.0",
    "Accept": "application/json"
}

def get_odata(url: str) -> pd.DataFrame:
    """Fetch all pages from a CBS OData v4 endpoint into a DataFrame."""
    frames = []
    next_url = url
    attempts = 0
    while next_url:
        try:
            resp = requests.get(next_url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            frames.append(pd.DataFrame(payload.get("value", [])))
            next_url = payload.get("@odata.nextLink")
            attempts = 0  # reset after success
        except requests.RequestException as e:
            attempts += 1
            if attempts > 3:
                raise
            time.sleep(1.5 * attempts)  # simple backoff
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
