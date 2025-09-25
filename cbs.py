import os
import sys
import time
import math
import json
import typing as t
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google.cloud import bigquery


PROJECT_ID = "compact-garage-473209-u4"     # your GCP project
DATASET_ID = "CBS"                        # existing BQ dataset
BASE = "https://datasets.cbs.nl/odata/v1/CBS/85318NED"

# Entities
MEASURECODES_URL = f"{BASE}/MeasureCodes?$select=Identifier,Title"
AREAGROUPS_URL   = f"{BASE}/WijkenEnBuurtenGroups?$select=Identifier,Title,ParentIdentifier"
OBS_URL          = f"{BASE}/Observations"

# HTTP headers
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "cbs-ping/1.0 (+BigQuery Loader)"
}

# Tuning
OBS_PAGE_LIMIT = 10000   # observations page size
BATCH_UPLOAD_SIZE = 20000  # rows per BQ load for observations

# Optional filters
CBS_PERIOD_FILTER = os.getenv("CBS_PERIOD_FILTER", "").strip()  # e.g. "2025" or "2025JJ00"
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


def get_odata_all(url: str) -> list[dict]:
    """
    Fetch ALL rows from an OData v4 entity using @odata.nextLink.
    Loads into memory; use only for reasonably sized entities.
    """
    rows: list[dict] = []
    next_url = url
    while next_url:
        r = SESSION.get(next_url, headers=HEADERS, timeout=120)
        r.raise_for_status()
        payload = r.json()
        page = payload.get("value", [])
        rows.extend(page)
        next_url = payload.get("@odata.nextLink")
    return rows


def iter_odata_pages(url: str) -> t.Iterator[list[dict]]:
    """
    Generator over pages for very large entities (Observations).
    Yields list[dict] for each page.
    """
    next_url = url
    while next_url:
        r = SESSION.get(next_url, headers=HEADERS, timeout=300)
        r.raise_for_status()
        payload = r.json()
        page = payload.get("value", [])
        yield page
        next_url = payload.get("@odata.nextLink")


def load_json_chunks_to_bq(rows: list[dict], table_id: str, schema: list[bigquery.SchemaField],
                           write_disposition: str = "WRITE_APPEND") -> None:
    """
    Load a list[dict] to BigQuery (single job). For very large volumes, prefer chunked calls.
    """
    if not rows:
        return
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition=write_disposition)
    job = BQ.load_table_from_json(rows, table_id, job_config=job_config)
    job.result()


def load_json_stream_to_bq(pages: t.Iterator[list[dict]], table_id: str, schema: list[bigquery.SchemaField],
                           first_page_write_disposition: str = "WRITE_TRUNCATE") -> int:
    """
    Stream pages of rows to BigQuery in batches.
    Returns total rows loaded.
    """
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
            # After first write, switch to append
            write_disposition = "WRITE_APPEND"
            print(f"[BQ] Loaded {total} rows into {table_id}...")

    # Flush remainder
    if buffer:
        job_config = bigquery.LoadJobConfig(schema=schema, write_disposition=write_disposition)
        job = BQ.load_table_from_json(buffer, table_id, job_config=job_config)
        job.result()
        total += len(buffer)
        print(f"[BQ] Loaded {total} rows into {table_id}...")

    return total


def main() -> None:
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"

    # ===================== 1) MeasureCodes =====================
    print("[CBS] Fetching MeasureCodes (all)...")
    measure_codes = get_odata_all(MEASURECODES_URL)
    measures_rows = [
        {
            "identifier": m.get("Identifier"),
            "title": m.get("Title")
        }
        for m in measure_codes if m.get("Identifier")
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

    # ===================== 2) WijkenEnBuurtenGroups =====================
    print("[CBS] Fetching WijkenEnBuurtenGroups (all GM/WK/BU members)...")
    area_groups = get_odata_all(AREAGROUPS_URL)
    areas_rows = [
        {
            "identifier": a.get("Identifier"),
            "title": a.get("Title"),
            "parent_identifier": a.get("ParentIdentifier"),
            # helpful derived columns
            "level": ("GM" if str(a.get("Identifier","")).startswith("GM")
                      else "WK" if str(a.get("Identifier","")).startswith("WK")
                      else "BU" if str(a.get("Identifier","")).startswith("BU")
                      else None)
        }
        for a in area_groups if a.get("Identifier")
    ]
    print(f"[CBS] WijkenEnBuurtenGroups rows: {len(areas_rows)}")

    areas_schema = [
        bigquery.SchemaField("identifier", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("parent_identifier", "STRING"),
        bigquery.SchemaField("level", "STRING"),
    ]
    load_json_chunks_to_bq(
        areas_rows,
        f"{dataset_ref}.wijken_en_buurten_groups",
        schema=areas_schema,
        write_disposition="WRITE_TRUNCATE" if BQ_WRITE_MODE != "WRITE_APPEND" else "WRITE_APPEND",
    )
    print("[CBS] WijkenEnBuurtenGroups -> BigQuery DONE")

    # ===================== 3) Observations (ALL) =====================
    # Optional WHERE by period to reduce volume if desired
    obs_url = f"{OBS_URL}?$top={OBS_PAGE_LIMIT}"
    if CBS_PERIOD_FILTER:
        # The column name can be "Periods" or "Perioden" depending on language.
        # We'll filter on either by building an OData 'or' condition.
        # But server-side, only one exists—CBS feeds usually use "Periods".
        period_filter = CBS_PERIOD_FILTER.replace("'", "''")
        obs_url += f"&$filter=Periods eq '{period_filter}' or Perioden eq '{period_filter}'"

    print(f"[CBS] Fetching Observations (paged) from: {obs_url}")
    obs_pages = iter_odata_pages(obs_url)

    obs_schema = [
        bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("measure", "STRING"),
        bigquery.SchemaField("wijkenenbuurten", "STRING"),
        bigquery.SchemaField("periods", "STRING"),
        bigquery.SchemaField("valueattribute", "STRING"),
        bigquery.SchemaField("value", "FLOAT64"),
        bigquery.SchemaField("stringvalue", "STRING"),
    ]

    # Transform pages on the fly to normalized dicts:
    def normalized_pages():
        first_period_key = None
        for page in obs_pages:
            out = []
            for o in page:
                # detect time column (Periods/Perioden) once
                if first_period_key is None:
                    for k in ("Periods", "Perioden"):
                        if k in o:
                            first_period_key = k
                            break
                    if first_period_key is None:
                        first_period_key = "Periods"
                out.append({
                    "id": o.get("Id"),
                    "measure": o.get("Measure"),
                    "wijkenenbuurten": o.get("WijkenEnBuurten"),
                    "periods": o.get(first_period_key),
                    "valueattribute": o.get("ValueAttribute"),
                    "value": o.get("Value"),
                    "stringvalue": o.get("StringValue"),
                })
            yield out

    write_disp_first = "WRITE_TRUNCATE" if BQ_WRITE_MODE != "WRITE_APPEND" else "WRITE_APPEND"
    total_obs = load_json_stream_to_bq(
        normalized_pages(),
        f"{dataset_ref}.observations_raw",
        schema=obs_schema,
        first_page_write_disposition=write_disp_first
    )
    print(f"[CBS] Observations -> BigQuery DONE (rows loaded: {total_obs})")

    print("\nAll done ✅")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
