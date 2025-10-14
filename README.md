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
    - RDW   Rijksdienst Wegverkeer
            Gekentekende Voertuigen (2025)
            [link](https://opendata.rdw.nl/Voertuigen/Open-Data-RDW-Gekentekende_voertuigen/m9d7-ebf2/about_data)
      
    - CBS   Centraal Bureau voor de Statistiek
        
        - Kerncijfers Wijken en Buurten (2024)
            [link](https://www.cbs.nl/nl-nl/maatwerk/2025/38/kerncijfers-wijken-en-buurten-2024)
        - Kerncijfers Postcode 4 (2024) [link](https://www.cbs.nl/nl-nl/longread/diversen/2023/statistische-gegevens-per-vierkant-en-postcode-2022-2021-2020-2019)
    - KiM   Kennisinstituut Mobilitiet, Ministerie van Infrastructuur en Waterstaat
            Atlas van de Auto (2024)
            [link](https://www.kimnet.nl/atlas-van-de-auto#auto-op-de-kaart)


## Data Collection and Preparation

A unified **ETL pipeline** integrates national open data from the RDW, CBS and KiM. \
The **Extract** phase retrieves raw records and the **Transform** phase cleans, harmonises and standardises the data:
- RDW notebooks classify car models by fuel type, body type and weight ([01_RDW_extraction.ipynb](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/01_RDW_extraction.ipynb), [02_RDW_cleaning.ipynb](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_RDW_cleaning.ipynb), [03_RDW_classification.ipynb](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/03_RDW_classification.ipynb)).
- Regional scripts merge PC6-PC4 areas and calculate proportions and Z-scores([01_REGIONAL_extr_transf.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/01_REGIONAL_extr_transf.py), [02_NBH_REGIONAL_cleaning.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_NBH_REGIONAL_cleaning.py), [02_PC4_REGIONAL_cleaning.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_PC4_REGIONAL_cleaning.py), [02_REGIONAL_cleaning.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_REGIONAL_cleaning.py)). 

The **Load** phase exports the final tables to BigQuery and populates the dashboard via Dockerised Streamlit components.

## Data Sources
	1. Netherlands Population Register (CBS) - Population_country_of_origin_country_of_birth_1_January_09102024_110602.csv
	
	2. Consumption Expenditure of households (CBS)* - After Cleaning - Consumption_110637_mark_cleaned.csv
	
	* Consumption expenditure was not mentioned in Concept Report

## Existing Indicators and Visualizations (50-150 words)

For each indicator/visualization you have implemented so far, 
        CSF/KPI supported by the indicator
        Original data source(s) used to create this indicator
        If the indicator is computed using multiple attributes or data sources, explain how is it computed
        Which files in your code (python, html, javascript etc) generates the indicator
