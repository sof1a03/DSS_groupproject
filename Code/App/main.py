import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import random
import os
import json
from shapely.geometry import shape
from pyproj import Transformer

#Page configuration and style
st.set_page_config(page_title="CarGo", layout="wide")
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: bold;
        color: #6A0DAD;
        text-align: left;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f7f5fb;
        border-left: 6px solid #6A0DAD;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.2rem;
        color: #6A0DAD;
        font-weight: bold;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

#Mapbox set for interactive map
os.environ["MAPBOX_API_KEY"] = "pk.eyJ1IjoiYW5uYWZlcnJpMDIiLCJhIjoiY21nZ2NweXczMGV5YTJscjQ0OGRjMnR1aSJ9.S0SIC7L5oKQpn0Bto1FGFQ"

#Header
st.markdown('<div class="main-title" style="text-align:center; font-size:3.5rem; margin-top:-1rem; margin-bottom:-1.2rem;">CarGo</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:1.3rem; color:#9A5ED5; font-style:italic; margin-top:-0.5rem;">Fuel your business with insights, not assumptions</p>', unsafe_allow_html=True)

#BigQuery connection in order to have access to the db
@st.cache_resource 
def get_bq_client():
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/run/secrets/gcp_credentials")
    if not os.path.isfile(creds_path):
        raise FileNotFoundError(
            f"Credenziali GCP non trovate: {creds_path}. "
            "Controlla il mount in docker-compose e l'ENV GOOGLE_APPLICATION_CREDENTIALS."
        )

    credentials = service_account.Credentials.from_service_account_file(creds_path)
    project_id = os.getenv("GCP_PROJECT", credentials.project_id)  
    bq_location = os.getenv("GCP_BQ_LOCATION")  
    client = bigquery.Client(project=project_id, credentials=credentials, location=bq_location)
    return client, project_id
bq_client, PROJECT_ID = get_bq_client()

#1 -> Sidebar to search for the desired vehicle with a search in the RDW dataset
st.sidebar.header("Search Vehicles:")
@st.cache_data(ttl=3600)
def get_unique_brands():
    query = f"""
        SELECT DISTINCT brand
        FROM `{PROJECT_ID}.RDW.rdw_classified`
        ORDER BY brand
    """
    df = bq_client.query(query).to_dataframe()
    return df['brand'].dropna().tolist()

@st.cache_data(ttl=3600)
def get_models_by_brand(selected_brand):
    query = f"""
        SELECT DISTINCT model
        FROM `{PROJECT_ID}.RDW.rdw_classified`
        WHERE brand = @brand
        ORDER BY model
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("brand", "STRING", selected_brand)]
    )
    df = bq_client.query(query, job_config=job_config).to_dataframe()
    return df['model'].dropna().tolist()

@st.cache_data(ttl=600)
def get_car_details(brand, model):
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.RDW.rdw_classified`
        WHERE brand = @brand AND model = @model
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("brand", "STRING", brand),
            bigquery.ScalarQueryParameter("model", "STRING", model)
        ]
    )
    return bq_client.query(query, job_config=job_config).to_dataframe()

brands = get_unique_brands()
selected_brand = st.sidebar.selectbox("Brand", [""] + brands)

if selected_brand:
    models = get_models_by_brand(selected_brand)
    selected_model = st.sidebar.selectbox("Model", [""] + models)

    if selected_model:
        details_df = get_car_details(selected_brand, selected_model)

        if not details_df.empty:
            car = details_df.iloc[0]

            st.sidebar.markdown(f"### {car['brand']} {car['model']}") #Images and title
            if pd.notnull(car['image_url_2']):
                st.sidebar.image(car['image_url_2'], use_container_width=True)
            elif pd.notnull(car['image_url_3']):
                st.sidebar.image(car['image_url_3'], width=200)

            st.sidebar.markdown("#### Model specifications") #Specifications of the vehicle searched
            st.sidebar.write(f"**Body:** {car.get('body_class', 'N/A')}")
            st.sidebar.write(f"**Seats:** {car.get('seats_median', 'N/A')}")
            st.sidebar.write(f"**Weight (kg):** {car.get('mass_empty_median', 'N/A')}")
            st.sidebar.write(f"**Length (cm):** {car.get('length_median', 'N/A')}")
            st.sidebar.write(f"**Width (cm):** {car.get('width_median', 'N/A')}")
            st.sidebar.write(f"**Wheelbase (cm):** {car.get('wheelbase_median', 'N/A')}")
            st.sidebar.write(f"**Fuel type:** {car.get('fuel_types_primary', 'N/A')}")
            st.sidebar.write(f"**Power/weight ratio:** {car.get('pw_ratio_median', 'N/A')}")

            st.sidebar.markdown("#### Average Prices (€)") #Prices for a first small overview
            st.sidebar.write(f"**2023:** {car.get('avg_2023', 'N/A')}")
            st.sidebar.write(f"**2024:** {car.get('avg_2024', 'N/A')}")
            st.sidebar.write(f"**2025:** {car.get('avg_2025', 'N/A')}")
        else:
            st.sidebar.warning("No details found for this car.")

#2 -> First row with Success Indicators and Interactive Neighborhood Map
@st.cache_data(ttl=3600)
def query_map_geometry(): #Query to get the geometry of the neighborhoods from BigQuery CBS dataset
    if not bq_client:
        return pd.DataFrame()
    query = """
        SELECT string_field_0 AS buurtcode, string_field_1 AS geometry_str
        FROM `compact-garage-473209-u4.MAP.geo-neighbourhoods`
        WHERE string_field_0 != 'buurtcode'
        LIMIT 100
    """
    df = bq_client.query(query).to_dataframe()

    df['geometry'] = df['geometry_str'].apply(lambda x: json.loads(x) if pd.notnull(x) else None)

    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

    def convert_coords(coords): #Convert RD coordinates to lat/lon
        new_coords = []
        for polygon in coords:
            new_poly = []
            for ring in polygon:
                new_ring = [list(transformer.transform(x, y)) for x, y in ring]
                new_poly.append(new_ring)
            new_coords.append(new_poly)
        return new_coords

    df['geometry_converted'] = df['geometry'].apply(
        lambda g: {"type": "MultiPolygon", "coordinates": convert_coords(g['coordinates'])} if g else None
    )

    def get_centroid(geom): #Calculate centroid for map centering
        try:
            shp = shape(geom)
            centroid = shp.centroid
            return centroid.y, centroid.x
        except Exception:
            return None, None

    df['lat_lon'] = df['geometry_converted'].apply(get_centroid)
    df[['latitude', 'longitude']] = pd.DataFrame(df['lat_lon'].tolist(), index=df.index)
    df = df.dropna(subset=['latitude', 'longitude'])

    df['intensity'] = [random.uniform(0.3, 1.0) for _ in range(len(df))]
    return df
geo_df = query_map_geometry()

col1, col2 = st.columns([1, 2]) #Columns for the first row

with col1: #Success Indicators (mock values)
    st.markdown('<div class="section-title">Success Indicators</div>', unsafe_allow_html=True)
    for i in range(1, 4):
        val = round(random.uniform(2.8, 3.5), 3)
        perc = random.randint(75, 95)
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Success Indicator {i}</b><br>
                <span style="font-size:1.5rem;color:#6A0DAD;">{val}</span>  
                <div style="width:20%;background:#eee;border-radius:8px;height:10px;margin-top:5px;">
                    <div style="width:{perc}%;background:#6A0DAD;height:10px;border-radius:8px;"></div>
                </div>
                <small>{perc}%</small>
            </div>
            """,
            unsafe_allow_html=True
        )

with col2: #Interactive Neighborhood Map
    st.markdown('<div class="section-title">Neighborhood Map</div>', unsafe_allow_html=True)

    if not geo_df.empty: 
        color_layer = pdk.Layer( 
            "GeoJsonLayer",
            data={
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": g, "properties": {"buurtcode": b, "intensity": i}}
                    for g, b, i in zip(geo_df['geometry_converted'], geo_df['buurtcode'], geo_df['intensity'])
                ]
            },
            get_fill_color="[255 * (1 - properties.intensity), 100, 255 * properties.intensity, 150]",
            get_line_color=[255, 255, 255],
            opacity=0.7,
            stroked=True,
            filled=True,
            pickable=True,
            auto_highlight=True
        )

        view_state = pdk.ViewState(latitude=52.1326, longitude=5.2913, zoom=7, pitch=0)

        st.pydeck_chart(pdk.Deck(
            layers=[color_layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v9"
        ))
    else:
        st.warning("Map not available.")

#3 -> Second row with Matching Details and Best Matching Regions
col3, col4 = st.columns([1, 2]) #Columns for the second row

with col3: #Matching Details with a radar chart (Mock values)
    st.markdown('<div class="section-title">Matching Details</div>', unsafe_allow_html=True)
    categories = ['Wealth', 'Household', 'Age', 'Size', 'Model']
    values1 = [random.uniform(0.5, 1.0) for _ in categories]
    values2 = [random.uniform(0.3, 0.9) for _ in categories]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values1, theta=categories, fill='toself', name='Ford Focus'))
    fig.add_trace(go.Scatterpolar(r=values2, theta=categories, fill='toself', name='Helderbuurt'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


with col4: #Best Matching Regions with a table (Mock values)
    st.markdown('<div class="section-title">Best Matching Regions</div>', unsafe_allow_html=True)
    data = {
        "Postal code": [9211, 7573, 1071, 2725, 4879, 4145, 3456, 3824],
        "Neighbourhood": ["Statenkwartier", "Harenbroek", "Helderbuurt", "Kadebuurt", "Carnism", "Randwijk", "Waarden", "Stadskwartier"],
        "Municipality": ["Maastricht", "Oldenzaal", "Velsen", "Zoetermeer", "Etten-Leur", "Waalre", "Bodegraven", "Amersfoort"],
        "Inhabitants": [1714, 4827, 3273, 1039, 3181, 2623, 1942, 2463],
        "Income (k€/y)": [35, 42, 38, 39, 41, 35, 32, 40],
        "Match %": [83.1, 92.6, 94.5, 91.2, 93.1, 90.5, 91.6, 92.0]
    }
    table = pd.DataFrame(data)
    st.dataframe(table, use_container_width=True)

#4 -> Third row with Sales Trends (Mock values)
col5, col6 = st.columns(2) #Columns for the third row

with col5: #New Car Sales
    st.markdown('<div class="section-title">Sales - New</div>', unsafe_allow_html=True)
    df_sales_new = pd.DataFrame({
        "Year": list(range(2010, 2026)),
        "Sales": [random.randint(4000, 8000) for _ in range(16)]
    })
    st.line_chart(df_sales_new.set_index("Year"))

with col6: #Sales Occasions
    st.markdown('<div class="section-title">Sales - Occasions</div>', unsafe_allow_html=True)
    df_sales_occ = pd.DataFrame({
        "Year": list(range(2010, 2026)),
        "Sales": [random.randint(10000, 20000) for _ in range(16)]
    })
    st.line_chart(df_sales_occ.set_index("Year"))

#Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray; margin-bottom:-1rem;'>© 2025 DSSGroup04-Dashboard Prototype</p>", unsafe_allow_html=True)
