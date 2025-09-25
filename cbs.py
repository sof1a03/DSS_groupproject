import requests
from google.cloud import bigquery

# ====== CBS OData v4 (Wijken en Buurten 2024) ====== ciao
BASE = "https://datasets.cbs.nl/odata/v1/CBS/85318NED"
OBS_URL = f"{BASE}/Observations"
MEASURECODES_URL = f"{BASE}/MeasureCodes?$select=Identifier,Title"
AREACODES_URL = f"{BASE}/WijkenEnBuurtenCodes?$select=Identifier,Title"

HEADERS = {"Accept": "application/json", "User-Agent": "cbs-ping/1.0"}

# ====== Config (cambia con i tuoi valori) ======
PROJECT_ID = "compact-garage-473209-u4"   # es: "my-gcp-project-123"
DATASET_ID = "prova3"            # crea prima il dataset in BigQuery con "bq mk -d prova3"

client = bigquery.Client(project=PROJECT_ID)


def get_odata_v4(url: str):
    """Scarica tutti i record da un endpoint CBS OData v4."""
    rows = []
    next_url = url
    while next_url:
        r = requests.get(next_url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        payload = r.json()
        rows.extend(payload.get("value", []))
        next_url = payload.get("@odata.nextLink")
    return rows


def upload_to_bigquery(rows, table_id, schema, write_disposition="WRITE_APPEND", batch_size=2000):
    if not rows:
        print(f"[BQ] Nessun dato per {table_id}")
        return

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=write_disposition,
    )

    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        job = client.load_table_from_json(chunk, table_id, job_config=job_config)
        job.result()
        print(f"[BQ] Caricati {len(chunk)} record in {table_id} (batch {i//batch_size+1})")


def main():
    # ---------- 1) Fetch metadata ----------
    print("[CBS] Fetching MeasureCodes...")
    measure_codes = get_odata_v4(MEASURECODES_URL)
    measures = [{"identifier": m["Identifier"], "title": m.get("Title")}
                for m in measure_codes if m.get("Identifier")]

    print("[CBS] Fetching WijkenEnBuurtenCodes...")
    area_codes = get_odata_v4(AREACODES_URL)
    areas = [{"identifier": a["Identifier"], "title": a.get("Title")}
             for a in area_codes if a.get("Identifier")]

    # ---------- 2) Fetch observations ----------
    print("[CBS] Fetching Observations sample...")
    observations = get_odata_v4(f"{OBS_URL}?$top=5000")

    # normalizza colonna dei periodi
    time_key = None
    if observations:
        for k in ("Periods", "Perioden"):
            if k in observations[0]:
                time_key = k
                break
    if not time_key:
        time_key = "Periods"
        for o in observations:
            o[time_key] = "unknown"

    obs_rows = [{
        "id": o.get("Id"),
        "measure": o.get("Measure"),
        "wijkenenbuurten": o.get("WijkenEnBuurten"),
        "periods": o.get(time_key),
        "valueattribute": o.get("ValueAttribute"),
        "value": o.get("Value"),
        "stringvalue": o.get("StringValue"),
    } for o in observations]

    print(f"[CBS] Preparati {len(measures)} measure_codes, {len(areas)} area_codes, {len(obs_rows)} observations.")

    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"

    # ---------- 3) Upload a BigQuery ----------
    upload_to_bigquery(
        measures,
        f"{dataset_ref}.measure_codes",
        schema=[
            bigquery.SchemaField("identifier", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING"),
        ],
        write_disposition="WRITE_TRUNCATE",
    )

    upload_to_bigquery(
        areas,
        f"{dataset_ref}.wijken_en_buurten_codes",
        schema=[
            bigquery.SchemaField("identifier", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING"),
        ],
        write_disposition="WRITE_TRUNCATE",
    )

    upload_to_bigquery(
        obs_rows,
        f"{dataset_ref}.observations_raw",
        schema=[
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("measure", "STRING"),
            bigquery.SchemaField("wijkenenbuurten", "STRING"),
            bigquery.SchemaField("periods", "STRING"),
            bigquery.SchemaField("valueattribute", "STRING"),
            bigquery.SchemaField("value", "FLOAT64"),
            bigquery.SchemaField("stringvalue", "STRING"),
        ],
        write_disposition="WRITE_APPEND",
    )


if __name__ == "__main__":
    main()
