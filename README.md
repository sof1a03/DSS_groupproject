# DSS-Group04

## Running the App

To run the frontend:
1. Make sure Docker is installed and running. Refer to [Docker](https://www.docker.com/)
2. Download and unzip Group04Midterm.zip, then open the `App` folder
3. From the project root, run:

   ```
   docker compose up --build
   ```
4. Once the container is ready, go to `localhost:8501` to view the dashboard.

### Links:
- Codebase: [GitHub](https://github.com/sof1a03/DSS_groupproject/tree/main)
- Datasets: 
	- [RDW](https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Final/RDW.csv)
   	- [REGIONAL](https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Final/REGIONAL.csv)
   	- [GEO](https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Geo/CBS_NBH_HEAD_10.csv)

## Data Collection and Preparation

A unified **ETL pipeline** integrates national open data from the RDW, CBS and KiM. \
The **Extract** phase retrieves raw records and the **Transform** phase cleans, harmonises and standardises the data:
- RDW notebooks classify car models by fuel type, body type and weight ([01_RDW_extraction.ipynb](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/01_RDW_extraction.ipynb), [02_RDW_cleaning.ipynb](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_RDW_cleaning.ipynb), [03_RDW_classification.ipynb](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/03_RDW_classification.ipynb)).
- Regional scripts merge PC6-PC4 areas and calculate proportions and Z-scores([01_REGIONAL_extr_transf.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/01_REGIONAL_extr_transf.py), [02_NBH_REGIONAL_cleaning.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_NBH_REGIONAL_cleaning.py), [02_PC4_REGIONAL_cleaning.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_PC4_REGIONAL_cleaning.py), [02_REGIONAL_cleaning.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_REGIONAL_cleaning.py)) .
 
The **Load** phase exports the final tables to BigQuery and populates the dashboard via Dockerised Streamlit components: 
- Regional script: [03_REGIONAL_load.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/03_REGIONAL_load.py) .

## Data Sources
- RDW   Rijksdienst Wegverkeer
        Gekentekende Voertuigen (2025) [link](https://opendata.rdw.nl/Voertuigen/Open-Data-RDW-Gekentekende_voertuigen/m9d7-ebf2/about_data)
      
- CBS   Centraal Bureau voor de Statistiek
  - Kerncijfers Wijken en Buurten (2024) [link](https://www.cbs.nl/nl-nl/maatwerk/2025/38/kerncijfers-wijken-en-buurten-2024)
  - Kerncijfers Postcode 4 (2024) [link](https://www.cbs.nl/nl-nl/longread/diversen/2023/statistische-gegevens-per-vierkant-en-postcode-2022-2021-2020-2019)
  - Buurt, Wijk en Gemeente (2023) voor Postcode + Huisnummer [link](https://www.cbs.nl/nl-nl/maatwerk/2023/35/buurt-wijk-en-gemeente-2023-voor-postcode-huisnummer)
  - KiM   Kennisinstituut Mobilitiet, Ministerie van Infrastructuur en Waterstaat. Atlas van de Auto (2024) [link](https://www.kimnet.nl/atlas-van-de-auto#auto-op-de-kaart)
  - Open Charge Map (OCM) [Link](https://openchargemap.org/site)
  - Google Custom Search API  [Link](https://programmablesearchengine.google.com/about/)

### Existing Indicators and Visualizations

1. **Vehicle Search and Model Details**  
   - **CSF/KPI:** Data completeness, quality, and market dynamics.  
   - **Data Source:** [RDW](https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Final/RDW.csv) vehicle dataset (`RDW.rdw_classified`, BigQuery).  
   - **Computation:** Queries car specs (fuel type, mass, average price) directly; no derived KPI yet.  
   - **Code File:** [main.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/App/main.py) (functions `get_unique_brands()`, `get_car_details()`).  

2. **Neighbourhood Map (Geo Layer)**  
   - **CSF/KPI:** Data consistency, look-alike matching (future).  
   - **Data Source:** [CBS GeoPackage 2023](https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Geo/CBS_NBH_HEAD_10.csv) (`MAP.geo_neighbourhoods`).  
   - **Computation:** Loads and converts geometries to WGS84; color intensity currently random.  
   - **Code File:** [main.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/App/main.py) (`query_map_geometry()` and PyDeck `GeoJsonLayer`).  

3. **Regional Compatibility – MatchScore**  
   - **CSF/KPI:** *Car–Market Fit* indicator measuring how well a car model aligns with regional socio-economic profiles.  
   - **Data Source:** Combined [RDW](https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Final/RDW.csv) + [CBS Regional](https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Final/REGIONAL.csv) data.  
   - **Computation:** Weighted Manhattan distance between car attributes (body/fuel type, price) and regional indicators (income, fleet composition).  
   - **Visualization:** Heatmap of PC4 regions and *Top-N table* with region name, score, affordability, and interest.  
   - **Code File:** [analysis_matchscore.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/Analytics/analysis_matchscore.py).  

4. **Market Positioning Indicators (Popularity, Niche)**  
   - **CSF/KPI:** Market dynamics and positioning within vehicle segments.  
   - **Data Source:** [RDW](https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Final/RDW.csv).  
   - **Visualization:** Radar or bar charts showing model distribution across body classes.  
   - **Code File:** [analysis_marketposition.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/Analytics/analysis_marketposition.py).  

5. **Market Potential & Forecasting**  
   - **CSF/KPI:** *Projected Market Growth* and *Forecast Accuracy (SMAPE)*.  
   - **Data Source:** Dynamic RDW API (registrations 2020–2025).  
   - **Computation:** *Holt–Winters Exponential Smoothing* with additive trend and seasonality. Forecasts displayed with ±80% prediction intervals.  
   - **Visualization:** Line chart showing historical vs. predicted registrations for new and second-hand vehicles.  
   - **Code File:** [forecasting.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/Analytics/forecasting.py).  

6. **EV Readiness & What-If Scenarios**  
   - **CSF/KPI:** *Electric Vehicle Readiness* and *Transition Potential*.  
   - **Data Source:** [KiM Atlas van de Auto](https://www.kimnet.nl/atlas-van-de-auto#auto-op-de-kaart) + [Open Charge Map](https://openchargemap.org/) API (charger density, power level).  
   - **Computation:** *K-Means Clustering* on standardized features (income, urbanization, fuel mix, household size).  
     Includes Ridge Regression “what-if” model simulating EV adoption under changing socio-economic conditions.  
   - **Visualization:** Clustered map of EV readiness, table of regional indicators, and interactive slider for scenario testing.  
   - **Code File:** [analysis_ev.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/Analytics/analysis_ev.py).  

7. **Vehicle Profile Explorer**  
   - **CSF/KPI:** Data completeness and quality check of RDW database.  
   - **Data Source:** [RDW classified dataset](https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Final/RDW.csv).  
   - **Computation:** Retrieves model specs (mass, seats, propulsion, price) and compares them with segment averages.  
   - **Visualization:** Model detail card with representative image (Google Custom Search API) and radar diagram summarizing vehicle attributes.  
   - **Code File:** [main.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/App/main.py).  
