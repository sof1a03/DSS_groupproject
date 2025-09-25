#!/usr/bin/env python3
"""
CBS OData v4 (Wijken en Buurten 2024) -> BigQuery

Tables created/loaded:
- measure_codes               (from MeasureCodes)
- wijken_en_buurten_codes     (from WijkenEnBuurtenCodes)
- observations_raw            (from Observations)

Env (optional):
  CBS_PERIOD_FILTER   e.g. "2025JJ00" or "2024" to restrict Observations by period
  BQ_WRITE_MODE       WRITE_TRUNCATE | WRITE_APPEND
"""

import os
import sys
import typing as t
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google.cloud import bigquery

# ====== EDIT THESE ======
PROJECT_ID = "compact-garage-473209-u4"
DATASET_ID = "CBS"

# CBS base (OData v4)
BASE = "https://datasets.cbs.nl/odata/v1/CBS/85318NED"
OBS_URL          = f"{BASE}/Observations"
MEASURECODES_URL = f"{BASE}/MeasureCodes?$select=Identifier,Title"
AREACODES_URL    = f"{BASE}/WijkenEnBuurtenCodes?$select=Identifier,Title"

# HTTP headers
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "cbs-ping/1.0 (+BigQuery Loader)"
}

# Tuning
BATCH_UPLOAD_SIZE = 20000

# Optional env
CBS_PERIOD_FILTER = os.getenv("CBS_PERIOD_FILTER", "").strip()
BQ_WRITE_MODE = os.getenv("BQ_WRITE_MODE", "").strip().upper()

# ====== HTTP session with retries ======
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=8, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = make_session()
BQ = bigquery.Client(project=PROJECT_ID)

# ====== Helpers ======
def get_odata_all(url: str) -> list[dict]:
    rows: list[dict] = []
    next_url = url
    while next_url:
        r = SESSION.get(next_url, headers=HEADERS, timeout=180)
        r.raise_for_status()
        payload = r.json()
        rows.extend(payload.get("value", []))
        next_url = payload.get("@odata.nextLink")
    return rows

def iter_odata_pages(url: str) -> t.Iterator[list[dict]]:
    next_url = url
    while next_url:
        r = SESSION.get(next_url, headers=HEADERS, timeout=300)
        r.raise_for_status()
        payload = r.json()
        yield payload.get("value", [])
        next_url = payload.get("@odata.nextLink")

def load_json_chunks_to_bq(
    rows: list[dict],
    table_id: str,
    schema: list[bigquery.SchemaField],
    write_disposition: str = "WRITE_APPEND",
) -> None:
    if not rows:
        print(f"[BQ] No rows to load into {table_id}")
        return
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition=write_disposition)
    job = BQ.load_table_from_json(rows, table_id, job_config=job_config)
    job.result()
    print(f"[BQ] Loaded {len(rows)} rows into {table_id}")

def load_json_stream_to_bq(
    pages: t.Iterator[list[dict]],
    table_id: str,
    schema: list[bigquery.SchemaField],
    first_page_write_disposition: str = "WRITE_TRUNCATE",
) -> int:
    total = 0
    buffer: list[dict] = []
    write_disposition = first_page_write_disposition

    for page in pages:
        if not page:
            continue
        buffer.extend(page)
        while len(buffer) >= BATCH_UPLOAD_SIZE:
            chunk, buffer = buffer[:BATCH_UPLOAD_SIZE], buffer[BATCH_UPLOAD_SIZE:]
            job_config = bigquery.LoadJobConfig(schema=schema, write_disposition=write_disposition)
            job = BQ.load_table_from_json(chunk, table_id, job_config=job_config)
            job.result()
            total += len(chunk)
            write_disposition = "WRITE_APPEND"
            print(f"[BQ] Loaded {total} rows into {table_id}...")

    if buffer:
        job_config = bigquery.LoadJobConfig(schema=schema, write_disposition=write_disposition)
        job = BQ.load_table_from_json(buffer, table_id, job_config=job_config)
        job.result()
        total += len(buffer)
        print(f"[BQ] Loaded {total} rows into {table_id}...")

    return total

# ====== Main ======
def main() -> None:
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"

    # ---------- 1) MeasureCodes ----------
    print("[CBS] Fetching MeasureCodes...")
    measure_codes_raw = get_odata_all(MEASURECODES_URL)

    measures_rows = [
        {"identifier": m.get("Identifier"), "title": m.get("Title")}
        for m in measure_codes_raw if m.get("Identifier")
    ]
    print(f"[CBS] MeasureCodes rows: {len(measures_rows)}")

    measures_schema = [
        bigquery.SchemaField("identifier", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("title", "STRING"),
    ]

    load_json_chunks_to_bq(
        measures_rows,
        f"{dataset_ref}.measure_codes",
        schema=measures_schema,
        write_disposition="WRITE_TRUNCATE" if BQ_WRITE_MODE != "WRITE_APPEND" else "WRITE_APPEND",
    )
    print("[CBS] MeasureCodes -> BigQuery DONE")

    # ---------- 2) WijkenEnBuurtenCodes ----------
    print("[CBS] Fetching WijkenEnBuurtenCodes...")
    area_codes_raw = get_odata_all(AREACODES_URL)

    area_codes_rows = [
        {"identifier": a.get("Identifier"), "title": a.get("Title")}
        for a in area_codes_raw if a.get("Identifier")
    ]
    print(f"[CBS] WijkenEnBuurtenCodes rows: {len(area_codes_rows)}")

    area_codes_schema = [
        bigquery.SchemaField("identifier", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("title", "STRING"),
    ]

    load_json_chunks_to_bq(
        area_codes_rows,
        f"{dataset_ref}.wijken_en_buurten_codes",
        schema=area_codes_schema,
        write_disposition="WRITE_TRUNCATE" if BQ_WRITE_MODE != "WRITE_APPEND" else "WRITE_APPEND",
    )
    print("[CBS] WijkenEnBuurtenCodes -> BigQuery DONE")

    # ---------- 3) Observations (paged) ----------
    obs_url = OBS_URL
    if CBS_PERIOD_FILTER:
        period = CBS_PERIOD_FILTER.replace("'", "''")
        obs_url += f"?$filter=Periods eq '{period}' or Perioden eq '{period}'"

    print(f"[CBS] Fetching Observations (paged) from:\n{obs_url}")
    raw_pages = iter_odata_pages(obs_url)

    def normalized_pages():
        detected_period_key = None
        for page in raw_pages:
            out = []
            for o in page:
                if detected_period_key is None:
                    for k in ("Periods", "Perioden"):
                        if k in o:
                            detected_period_key = k
                            break
                    if detected_period_key is None:
                        detected_period_key = "Periods"
                out.append({
                    "id": o.get("Id"),
                    "measure": o.get("Measure"),
                    "wijkenenbuurten": o.get("WijkenEnBuurten"),
                    "periods": o.get(detected_period_key),
                    "valueattribute": o.get("ValueAttribute"),
                    "value": o.get("Value"),
                    "stringvalue": o.get("StringValue"),
                })
            yield out

    obs_schema = [
        bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("measure", "STRING"),
        bigquery.SchemaField("wijkenenbuurten", "STRING"),
        bigquery.SchemaField("periods", "STRING"),
        bigquery.SchemaField("valueattribute", "STRING"),
        bigquery.SchemaField("value", "FLOAT64"),
        bigquery.SchemaField("stringvalue", "STRING"),
    ]

    first_write = "WRITE_TRUNCATE" if BQ_WRITE_MODE != "WRITE_APPEND" else "WRITE_APPEND"
    total_obs = load_json_stream_to_bq(
        normalized_pages(),
        f"{dataset_ref}.observations_raw",
        schema=obs_schema,
        first_page_write_disposition=first_write,
    )
    print(f"[CBS] Observations -> BigQuery DONE (rows loaded: {total_obs})")

    print("\nAll done ✅")

# Entry point
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except requests.HTTPError as e:
        print(f"\nHTTP error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)
