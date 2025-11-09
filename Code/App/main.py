# main.py
import os
import random
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

#Import functions from the other files
from utils_common import (
    apply_base_style,
    ensure_mapbox_token,
    get_bq_client,
    get_unique_brands,
    get_models_by_brand,
    get_car_details,
)
from analysis import render_analysis 
from matching import compute_match_map_package, build_pc4_heatmap_features
from ev import render_ev

#Setting Page
st.set_page_config(page_title="CarGo", layout="wide")
apply_base_style()

#CSS
st.markdown("""
<style>
.metric-card { position: relative; }

.kpi-help {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #efe9f7;
  color: #6A0DAD;
  font-size: 12px;
  font-weight: 700;
  line-height: 18px;
  text-align: center;
  cursor: default;
  user-select: none;
  border: 1px solid #e0d4f2;
}

/* Tooltip */
.kpi-help .kpi-tooltip {
  position: absolute;
  top: 24px;
  right: 0;
  background: #FFEAF1;
  color: #333;
  border: 1px solid #e6e6e6;
  border-radius: 8px;
  padding: 8px 10px;
  width: 260px;
  box-shadow: 0 8px 24px rgba(0,0,0,.12);
  font-size: 12.5px;
  line-height: 1.35;
  z-index: 50;
  opacity: 0;
  visibility: hidden;
  transition: opacity .15s ease, visibility .15s ease;
}

.kpi-help .kpi-tooltip:before {
  content: "";
  position: absolute;
  top: -6px;
  right: 8px;
  border-width: 6px;
  border-style: solid;
  border-color: transparent transparent #FFEAF1 transparent;
  filter: drop-shadow(0 -1px 0 #e6e6e6);
}

.kpi-help:hover .kpi-tooltip { opacity: 1; visibility: visible; }

.kpi-help:focus .kpi-tooltip,
.kpi-help:active .kpi-tooltip { opacity: 1; visibility: visible; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
        <style>
        .home-wrapper {
            max-width: 900px;
            margin: auto;
            text-align: center;
            line-height: 1.55;
            font-size: 1.03rem;
        }
        .home-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #4B0082;
            text-align: center;
            margin-bottom: 0.8rem;
        }
        .home-section-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #6A0DAD;
            margin-top: 2rem;
            margin-bottom: 0.6rem;
            text-align: center;
        }
        .home-list {
            text-align: left;
            margin: auto;
            max-width: 650px;
            line-height: 1.55;
        }
        .home-list-centre {
            text-align: center;
            margin: auto;
            max-width: 650px;
            line-height: 1.55;
        }
        </style>
        """, unsafe_allow_html=True)

#Inizialize page and url
params = st.query_params
if "page" not in st.session_state:
    st.session_state.page = params.get("page", "home_intro")

def set_page(page: str):
    st.session_state.page = page
    try:
        st.query_params["page"] = page
    except Exception:
        st.experimental_set_query_params(page=page)
    st.rerun() 

ensure_mapbox_token() #Token Mapbox for the map

#Render function for the stile of the Success Indicators
def render_kpi_card(title: str, value_str: str, tooltip: str):
    st.markdown(f"""
    <div class="metric-card">
        <div class="kpi-help" tabindex="0" aria-label="Info">
            ?
            <div class="kpi-tooltip">{tooltip}</div>
        </div>
        <b>{title}</b><br>
        <span style="font-size:1.5rem;color:#6A0DAD;">{value_str}</span>
    </div>
    """, unsafe_allow_html=True)

def render_home_intro():
    # Inject CSS only once
    st.markdown("""
    <div class="home-wrapper">
    This dashboard helps you explore mobility and market potential across the Netherlands — <b>quickly, visually and with real data</b>.
    
    Whether you're a city planner, charging operator, mobility company or simply curious!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="home-wrapper">
    <b> No technical skills required — just explore, click and learn.  
    The dashboard automatically connects data, cleans it, models it and visualises it for you.</b>
    </div>
    """, unsafe_allow_html=True)

    # SECTION 2 & 3
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="home-section-title">What you will find inside</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="home-list-centre"> \n
        Smart rankings and heatmaps  \n
        Cluster analysis to group similar territories  \n
        Charging overview    \n
        Forecasts and scenario simulations  
        </div>
        """, unsafe_allow_html=True)

    with col_right: 
        st.markdown('<div class="home-section-title">What this helps you do</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="home-list-centre">\n
        Identify where demand is growing \n
        Support data-driven decisions  \n
        Validate strategies before investing  \n
        Communicate insights to stakeholders easily 
        </div>
        """, unsafe_allow_html=True)

    # SECTION 4
    st.markdown('<div class="home-section-title">How to get started</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="home-list"> \n
    1. Use the menu on the left to select the car brand and model of your interest \n
    2. Start from the **Match & Map**, or jump into the **Analysis** \n
    3. Hover, filter or click through charts to explore deeper  \n
    4. Interested on **EV**? Get knowledge from this section and do not forget to try the **What-if-Scenarios**
    </div>

    <div class="home-wrapper" style="margin-top: 18px;">
    <b>Ready?</b><br>
    Start exploring — the data will do the talking.
    </div>
    """, unsafe_allow_html=True)



#Title and Subtitle
st.markdown(
    "<h1 style='text-align:center; font-size:5rem; font-weight:800; color:#6A0DAD;'>CarGo</h1>",
    unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align:center;font-size:1.3rem;color:#9A5ED5;font-style:italic;margin-top:-0.5rem;">Fuel your business with insights, not assumptions</p>',
    unsafe_allow_html=True
)

#BigQuery Client
try:
    bq_client, PROJECT_ID = get_bq_client()
except Exception as e:
    bq_client, PROJECT_ID = None, None
    st.error(f"BigQuery init error: {e}")

#Side Bar for searching vehicles 
st.sidebar.header("Search Vehicles:")

brands = get_unique_brands(PROJECT_ID) if bq_client and PROJECT_ID else []
selected_brand = st.sidebar.selectbox("Brand", [""] + brands)

selected_model = ""
car_details = pd.DataFrame()
if selected_brand:
    models = get_models_by_brand(PROJECT_ID, selected_brand) if bq_client else []
    selected_model = st.sidebar.selectbox("Model", [""] + models)

    if selected_model:
        car_details = get_car_details(PROJECT_ID, selected_brand, selected_model) if bq_client else pd.DataFrame()
        if not car_details.empty:
            car = car_details.iloc[0]
            if pd.notnull(car.get("image_url_2")):
                st.sidebar.image(car["image_url_2"], use_container_width=True)
                with st.sidebar.expander("Specifications", expanded=True):
                    spec_labels = {
                    "body_class": "Body Type",
                    "seats_median": "Seats",
                    "fuel_types_primary": "Fuel Type",
                    }

                    for fld, lbl in spec_labels.items():
                        val = car.get(fld, "N/A")

                        # Format seats as an integer if numeric
                        if fld == "seats_median":
                            try:
                                if val is not None and val != "N/A":
                                    val = int(round(float(val)))
                            except Exception:
                                pass
                        st.write(f"**{lbl}:** {val}")

        #Average Prices
        with st.sidebar.expander("Average Prices (€)", expanded=False):
            for yr in ["2023", "2024", "2025"]:
                val = car.get(f"avg_{yr}", None)
                if pd.notnull(val):
                    st.write(f"**{yr}:** € {val:,.0f}".replace(",", "."))
                else:
                    st.write(f"**{yr}:** N/A")
    else:
            st.sidebar.info("No details found for this car.")

#GRAPHIC PART
st.markdown("""
<style>
/* centra la *card* del bottone */
div.stButton { text-align: center; }

/* centra davvero il pulsante e applica lo stile */
div.stButton > button:first-child {
    display: inline-block;
    margin: 0 0;             /* centra */
    width: 280px;               /* opzionale; rimuovi per full-width */
    font-size: 1.1rem;
    padding: 0.6rem 0.65rem;
    background-color: #cdb0f6;
    color: #fff;
    border: 1px solid #cdb0f6;
    border-radius: 10px;
    font-weight: 700;
}
div.stButton > button:first-child:hover {
    background-color: #c39bf1;
    border-color: #c39bf1;
}
</style>
""", unsafe_allow_html=True)

if st.session_state.page == "home_intro":
    render_home_intro()
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([3, 3, 1.3])
    with c2:
        if st.button("Open Dashboard"):
            set_page("tabs_view")

elif st.session_state.page == "tabs_view":
    if st.button("Home"):
        set_page("home_intro")

    #Horizontal tabs via option_menu
    choice = option_menu(
        menu_title=None,
        options=["Match & Map", "Analysis", "EV"],
        icons=["map","bar-chart","lightning"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important","background-color": "#f0f2f6"},
            "nav-link": {"font-size":"16px","text-align":"center","margin":"0!important",
                         "padding":"10px 20px!important","color":"#6a0dad",
                         "background-color":"#f0f2f6","border-radius":"8px !important",
                         "line-height":"30px","box-sizing":"border-box","transition":"all 0.3s ease"},
            "nav-link-selected":{"background-color":"#d6b3ff","color":"#4b0082"},
            "icon":{"color":"inherit","font-size":"18px","vertical-align":"middle"}
        }
    )

    #Match & Map Tab 
    if choice == "Match & Map":
        #Pkg calculation
        pkg = None
        if selected_brand and selected_model and bq_client and PROJECT_ID:
            try:
                from matching import compute_match_map_package
                pkg = compute_match_map_package(bq_client, PROJECT_ID, selected_brand, selected_model)
            except Exception as e:
                st.error(f"Error in the Match&Map Calculation: {e}")
                

        #Success Indicators and Map
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="section-title">Success Indicators</div>', unsafe_allow_html=True)

            if pkg:
                render_kpi_card(
                    title="Profile Match",
                    value_str=f"{pkg['profile_match']:.1f}%",
                    tooltip=(
                        "The Profile Match quantifies the overall suitability of the selected car for"
                        "the target region, providing a single metric from  0%  to  100%. "
                        "A score near 100% indicates a strong fit, meaning the car's segment, "
                        "fuel, weight, price, and seat capacity highly align with the region's demographics and existing market preferences."
                    )
                )

                render_kpi_card(
                    title="Popularity",
                    value_str=f"{pkg['kpis']['popularity_score']:.1f}%",
                    tooltip=(
                        "The Popularity score measures how similar the selected "
                        "car is to the most popular model in its class. "
                        "A score near 100% means the car closely resembles the market leader."
                    )
                )

                render_kpi_card(
                    title="Niche",
                    value_str=f"{pkg['kpis']['niche_score']:.1f}%",
                    tooltip=(
                        "The Niche Score quantifies how concentrated or broad a car model’s market position is. "
                        "Low scores (0–30%) indicate mass-market models — affordable cars with wide appeal and high sales. "
                        "Mid-range scores (30–60%) represent moderately specialized models. "
                        "High scores (60–100%) identify niche or premium vehicles with limited market reach. "
                    )
                )
            else:
                st.info("Select Brand and Model to visualize the Success Indicators")

        with col2:
            st.markdown('<div class="section-title">Neighborhood Map</div>', unsafe_allow_html=True)
            view_state = pdk.ViewState(latitude=52.1326, longitude=5.2913, zoom=7)
            layers = []

            if pkg and "ranked_pc4" in pkg and not pkg["ranked_pc4"].empty and bq_client and PROJECT_ID:
                try:
                    features = build_pc4_heatmap_features(bq_client, PROJECT_ID, pkg["ranked_pc4"])

                    if features:
                        geojson_layer = pdk.Layer(
                            "GeoJsonLayer",
                            data={"type": "FeatureCollection", "features": features},
                            pickable=True,
                            stroked=True,
                            filled=True,
                            get_fill_color="properties.fillColor",
                            get_line_color=[80, 80, 80, 80],
                            lineWidthMinPixels=0.5,
                            auto_highlight=True,
                        )
                        layers.append(geojson_layer)

                        tooltip = {
                            "html": "<b>PC4:</b> {pc4}<br/><b>Score:</b> {final_score}",
                            "style": {"backgroundColor": "white", "color": "black"}
                        }

                        deck = pdk.Deck(
                            initial_view_state=view_state,
                            layers=layers,
                            tooltip=tooltip,
                            map_style="mapbox://styles/annaferri02/cmhp36mwo004i01s45s2zaaeo",
                        )
                        st.pydeck_chart(deck)
                    else:
                        st.info("No PC4 geometry available for current scores")

                except Exception as ex:  
                    st.exception(ex)

            else:
                #Fallback: map without heatlayer
                st.pydeck_chart(
                    pdk.Deck(
                        initial_view_state=view_state,
                        map_style="mapbox://styles/annaferri02/cmhp36mwo004i01s45s2zaaeo",
                    )
                )

        #Matching Details graph & Best Matching Regions table
        col3, col4 = st.columns([1, 2])
        with col3:
            st.markdown('<div class="section-title">Matching Details</div>', unsafe_allow_html=True)
            if pkg and pkg.get("radar"):
                radar = pkg["radar"]
                cats = radar["categories"]
                r_region = radar["region_values"] + [radar["region_values"][0]]
                r_car    = radar["car_values"]    + [radar["car_values"][0]]
                theta    = cats + [cats[0]]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=r_car, theta=theta, fill='toself', name=f"{selected_brand} {selected_model}"))
                fig.add_trace(go.Scatterpolar(r=r_region, theta=theta, fill='toself', name=f'PC4 {radar["best_pc4"]}'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Radar not available: Select a vehicle or verify data")

        with col4:
            st.markdown('<div class="section-title">Best Matching Regions</div>', unsafe_allow_html=True)
            if pkg:
                ranked_pc4 = pkg["ranked_pc4"]
                show_cols = ["pc4", "final_score", "interest_score", "affordability_fit"]
                df_display = ranked_pc4[show_cols].rename(columns={
                    "pc4": "PostalCode 4",
                    "final_score": "Score",
                    "interest_score": "Interest Score",
                    "affordability_fit": "Affordability Fit"
                })
                st.dataframe(df_display.head(10), use_container_width=True)
            else:
                st.info("Select a brand and model to see the PC4 leaderboard")

    #Analysis directly imported from the file "analysis.py"
    elif choice == "Analysis":
        render_analysis(selected_brand, selected_model)

    #EV analysis directly imported from the file "ev.py"
    elif choice == "EV":
        OCM_API_KEY = os.getenv("OCM_API_KEY", "d8d6328c-6367-4a60-a97c-9db909e524c2")
        render_ev(bq_client, PROJECT_ID, OCM_API_KEY)

#footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>©️ 2025 DSSGroup04 Dashboard Prototype</p>",
    unsafe_allow_html=True,
)
