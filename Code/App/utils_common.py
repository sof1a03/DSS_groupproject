# utils_common.py
import os
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

DEFAULT_MAPBOX = "pk.eyJ1IjoiYW5uYWZlcnJpMDIiLCJhIjoiY21nZ2NweXczMGV5YTJscjQ0OGRjMnR1aSJ9.S0SIC7L5oKQpn0Bto1FGFQ"

def apply_base_style():
    st.markdown(
        """
        <style>
        .main-title { font-size: 2rem; font-weight: bold; color: #6A0DAD; text-align: left; margin-bottom: 1rem; }
        .metric-card { background-color: #f7f5fb; border-left: 6px solid #6A0DAD; padding: 1rem; border-radius: 0.5rem;
                       text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1rem; }
        .section-title { font-size: 1.2rem; color: #6A0DAD; font-weight: bold; margin-top: 1.5rem; }
        ul[class^="nav"] li a:hover { background-color: #d6b3ff !important; color: #4b0082 !important; }
        ul[class^="nav"] li:first-child a { border-top-left-radius: 8px !important; border-bottom-left-radius: 8px !important; }
        ul[class^="nav"] li:last-child a { border-top-right-radius: 8px !important; border-bottom-right-radius: 8px !important; }
        ul[class^="nav"] { padding: 0 !important; margin: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

def ensure_mapbox_token():
    tok = os.getenv("MAPBOX_API_KEY") or DEFAULT_MAPBOX
    os.environ["MAPBOX_API_KEY"] = tok
    try:
        import pydeck as pdk
        pdk.settings.mapbox_api_key = tok
    except Exception:
        pass

@st.cache_resource
def get_bq_client():
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/run/secrets/gcp_credentials")
    credentials = service_account.Credentials.from_service_account_file(creds_path)
    project_id = credentials.project_id
    client = bigquery.Client(project=project_id, credentials=credentials)
    return client, project_id

@st.cache_data(ttl=3600)
def get_unique_brands(project_id: str):
    client, _ = get_bq_client()
    q = f"SELECT DISTINCT brand FROM `{project_id}.RDW.rdw_classified_final` ORDER BY brand"
    return client.query(q).to_dataframe()["brand"].dropna().tolist()

@st.cache_data(ttl=3600)
def get_models_by_brand(project_id: str, brand: str):
    client, _ = get_bq_client()
    q = f"SELECT DISTINCT model FROM `{project_id}.RDW.rdw_classified_final` WHERE brand=@b ORDER BY model"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("b", "STRING", brand)]
    )
    return client.query(q, job_config=job_config).to_dataframe()["model"].dropna().tolist()

@st.cache_data(ttl=600)
def get_car_details(project_id: str, brand: str, model: str):
    client, _ = get_bq_client()
    q = f"SELECT * FROM `{project_id}.RDW.rdw_classified_final` WHERE brand=@b AND model=@m LIMIT 1"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("b", "STRING", brand),
            bigquery.ScalarQueryParameter("m", "STRING", model),
        ]
    )
    return client.query(q, job_config=job_config).to_dataframe()
