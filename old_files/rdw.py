#!/usr/bin/env python3
"""
RDW Open Data (Voertuigen + Brandstof) -> BigQuery

Dataset: RDW
Tables:
  - rdw_vehicles_raw
  - rdw_fuel_raw
  - rdw_vehicles (final, joined)

Env (optional):
  BQ_WRITE_MODE       WRITE_TRUNCATE | WRITE_APPEND
"""

import os
import sys
import typing as t
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google.cloud import bigquery

# ====== CONFIG ======
PROJECT_ID = "compact-garage-473209-u4"
DATASET_ID = "RDW"

VEHICLES_URL = (
    "https://opendata.rdw.nl/resource/m9d7-ebf2.json"
    "?$select=kenteken,merk,handelsbenaming,catalogusprijs"
)
FUEL_URL = (
    "https://opendata.rdw.nl/resource/8ys7-d773.json"
    "?$select=kenteken,brandstof_omschrijving"
)

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "rdw-bigquery-streamer/1.0"
}

BATCH_UPLOAD_SIZE = 20000
BQ_WRITE_MODE = os.getenv("BQ_WRITE_MODE", "").strip().upper()

# ====== Setup ======
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=8, connect=8, read=8,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    return s

SESSION = make_session()
BQ = bigquery.Client(project=PROJECT_ID)

# ====== Helpers ======
def iter_socrata_pages(url: str, limit: int = 50000) -> t.Iterator[list[dict]]:
    offset = 0
    while True:
        paged_url = f"{url}&$limit={limit}&$offset={offset}"
        r = SESSION.get(paged_url, headers=HEADERS, timeout=300)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            print(f"[HTTP] Unexpected payload, skipping offset {offset}: {data}")
            break
        if not data:
            break
        yield data
        offset += limit
        print(f"[HTTP] Retrieved {len(data)} rows (offset={offset})")

def load_stream_to_bq(
    pages: t.Iterator[list[dict]],
    table_id: str,
    schema: list[bigquery.SchemaField],
    first_write: str = "WRITE_TRUNCATE",
) -> int:
    total = 0
    disposition = first_write
    for page in pages:
        valid_rows = [r for r in page if isinstance(r, dict) and r is not None]
        if not valid_rows:
            continue
        job_config = bigquery.LoadJobConfig(schema=schema, write_disposition=disposition)
        job = BQ.load_table_from_json(valid_rows, table_id, job_config=job_config)
        job.result()
        total += len(valid_rows)
        print(f"[BQ] {total} rows loaded into {table_id}...")
        disposition = "WRITE_APPEND"
    return total

# ====== Main ======
def main() -> None:
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
    first_write = "WRITE_TRUNCATE" if BQ_WRITE_MODE != "WRITE_APPEND" else "WRITE_APPEND"

    # Ensure dataset exists
    try:
        BQ.get_dataset(dataset_ref)
    except Exception:
        print(f"[BQ] Dataset {DATASET_ID} not found, creating...")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "EU"
        BQ.create_dataset(dataset, exists_ok=True)
        print(f"[BQ] Dataset {DATASET_ID} created.")

    # ---- Step 1: Vehicles ----
    print("[RDW] Streaming vehicles...")
    vehicle_schema = [
        bigquery.SchemaField("kenteken", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("merk", "STRING"),
        bigquery.SchemaField("handelsbenaming", "STRING"),
        bigquery.SchemaField("catalogusprijs", "FLOAT64"),
    ]
    total_vehicles = load_stream_to_bq(
        iter_socrata_pages(VEHICLES_URL),
        f"{dataset_ref}.rdw_vehicles_raw",
        schema=vehicle_schema,
        first_write=first_write,
    )
    print(f"[RDW] Vehicles done ({total_vehicles} rows)")

    # ---- Step 2: Fuel ----
    print("[RDW] Streaming fuel...")
    fuel_schema = [
        bigquery.SchemaField("kenteken", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("brandstof_omschrijving", "STRING"),
    ]
    total_fuel = load_stream_to_bq(
        iter_socrata_pages(FUEL_URL),
        f"{dataset_ref}.rdw_fuel_raw",
        schema=fuel_schema,
        first_write=first_write,
    )
    print(f"[RDW] Fuel done ({total_fuel} rows)")

    # ---- Step 3: Join inside BigQuery ----
    print("[RDW] Executing SQL join in BigQuery...")
    sql = f"""
    CREATE OR REPLACE TABLE `{dataset_ref}.rdw_vehicles` AS
    SELECT
        v.kenteken,
        v.merk AS brand,
        v.handelsbenaming AS model,
        CAST(v.catalogusprijs AS FLOAT64) AS price,
        f.brandstof_omschrijving AS fuel_type
    FROM `{dataset_ref}.rdw_vehicles_raw` v
    LEFT JOIN `{dataset_ref}.rdw_fuel_raw` f
    USING(kenteken)
    """
    query_job = BQ.query(sql)
    query_job.result()
    print("[RDW] Final table created: rdw_vehicles ✅")

# ====== Entry point ======
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
