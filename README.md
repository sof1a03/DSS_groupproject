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
    - RDW:
    - CBS:
    - KiM:

## Data Collection and Preparation

A unified **ETL pipeline** integrates national open data from the RDW, CBS and KiM. \
The **Extract** phase retrieves raw records and the **Transform** phase cleans, harmonises and standardises the data:
- RDW notebooks classify car models by fuel type, body type and weight ([01_RDW_extraction.ipynb](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/01_RDW_extraction.ipynb), [02_RDW_cleaning.ipynb](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_RDW_cleaning.ipynb), [03_RDW_classification.ipynb](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/03_RDW_classification.ipynb)).
- Regional scripts merge PC6-PC4 areas and calculate proportions and Z-scores([01_REGIONAL_extr_transf.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/01_REGIONAL_extr_transf.py), [02_NBH_REGIONAL_cleaning.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_NBH_REGIONAL_cleaning.py), [02_PC4_REGIONAL_cleaning.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_PC4_REGIONAL_cleaning.py), [02_REGIONAL_cleaning.py](https://github.com/sof1a03/DSS_groupproject/blob/main/Code/ETL/02_REGIONAL_cleaning.py)). 

The **Load** phase exports the final tables to BigQuery and populates the dashboard via Dockerised Streamlit components.


## Existing Indicators and Visualizations (50-150 words)

For each indicator/visualization you have implemented so far, 
        CSF/KPI supported by the indicator
        Original data source(s) used to create this indicator
        If the indicator is computed using multiple attributes or data sources, explain how is it computed
        Which files in your code (python, html, javascript etc) generates the indicator