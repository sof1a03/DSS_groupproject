# DSS-Group04

## Running the App

To run the frontend:
1. Make sure Docker is installed and running. Refer to [Docker](https://www.docker.com/)
2. Download and unzip Group04Midterm.zip, then open the `App` folder
3. From the project root, run:

   ```
   docker-compose up --build
   ```
4. Once the container is ready, go to `localhost:8501` to view the dashboard.

### Links:
- Codebase: [GitHub](https://github.com/sof1a03/DSS_groupproject/tree/main)
- Datasets: 
	- [RDW](https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Final/RDW.csv)
   	- [REGIONAL] (https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Final/REGIONAL.csv)
   	- [GEO] (https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Geo/CBS_NBH_HEAD_10.csv) 

## Data Collection and Preparation

A unified **ETL pipeline** integrates national open data from the RDW, CBS and KiM. \
The **Extract** phase retrieves raw records and the **Transform** phase cleans, harmonises and standardises the data:
- RDW notebooks classify car models by fuel type, body type and weight ([01_RDW_extraction.ipynb](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/01_RDW_extraction.ipynb), [02_RDW_cleaning.ipynb](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_RDW_cleaning.ipynb), [03_RDW_classification.ipynb](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/03_RDW_classification.ipynb)).
- Regional scripts merge PC6-PC4 areas and calculate proportions and Z-scores([01_REGIONAL_extr_transf.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/01_REGIONAL_extr_transf.py), [02_NBH_REGIONAL_cleaning.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_NBH_REGIONAL_cleaning.py), [02_PC4_REGIONAL_cleaning.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_PC4_REGIONAL_cleaning.py), [02_REGIONAL_cleaning.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_REGIONAL_cleaning.py)). 

The **Load** phase exports the final tables to BigQuery and populates the dashboard via Dockerised Streamlit components.

## Data Sources
- RDW   Rijksdienst Wegverkeer
        Gekentekende Voertuigen (2025) [link](https://opendata.rdw.nl/Voertuigen/Open-Data-RDW-Gekentekende_voertuigen/m9d7-ebf2/about_data)
      
- CBS   Centraal Bureau voor de Statistiek
  - Kerncijfers Wijken en Buurten (2024) [link](https://www.cbs.nl/nl-nl/maatwerk/2025/38/kerncijfers-wijken-en-buurten-2024)
  - Kerncijfers Postcode 4 (2024) [link](https://www.cbs.nl/nl-nl/longread/diversen/2023/statistische-gegevens-per-vierkant-en-postcode-2022-2021-2020-2019)
  - Buurt, Wijk en Gemeente (2023) voor Postcode + Huisnummer [link](https://www.cbs.nl/nl-nl/maatwerk/2023/35/buurt-wijk-en-gemeente-2023-voor-postcode-huisnummer)
  - KiM   Kennisinstituut Mobilitiet, Ministerie van Infrastructuur en Waterstaat. Atlas van de Auto (2024) [link](https://www.kimnet.nl/atlas-van-de-auto#auto-op-de-kaart)

### Existing Indicators and Visualizations

1. **Vehicle Search and Model Details**  
   - **CSF/KPI:** Data completeness, quality, and market dynamics.  
   - **Data Source:** [RDW](https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Final/RDW.csv) vehicle dataset (`RDW.rdw_classified`, BigQuery).  
   - **Computation:** Queries car specs (fuel type, mass, average price) directly; no derived KPI yet.  
   - **Code File:**  [main.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/App/main.py) (functions `get_unique_brands()`, `get_car_details()`).  

2. **Neighbourhood Map (Geo Layer)**  
   - **CSF/KPI:** Data consistency, look-alike matching (future).  
   - **Data Source:** [CBS GeoPackage 2023](https://github.com/sof1a03/DSS_groupproject/blob/main/Data/Geo/CBS_NBH_HEAD_10.csv) (`MAP.geo_neighbourhoods`).  
   - **Computation:** Loads and converts geometries to WGS84; color intensity currently random.  
   - **Code File:**  [main.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/App/main.py) (`query_map_geometry()` and PyDeck `GeoJsonLayer`).  

Other charts and tables are placeholders for KPIs like *Match-Score*, *Niche-Score*, and *Market Potential*. Future iterations will integrate computed KPIs to assess data completeness, consistency, and market dynamics for both new and pre-owned vehicles with more in detail graphs.

