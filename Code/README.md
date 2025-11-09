# Code Repository Overview

This section of the repository is organized into three main components:

## App
Contains the application layer, including scripts for running the dashboard, APIs, and visualization interfaces. 
It consists of the following documents:
- **Dockerfile**              Builds the Docker image containing the Python environment and application code.
- **analysis.py**             Calculates time series forecasts for RDW vehicle registration data.
- **docker-compose.yml**      Defines the configuration for running the application in a multi-container environment.
- **ev.py**                   Runs EV market analysis, including clustering and what-if scenario simulations for regions.
- **main.py**                 The main Streamlit application file that controls the dashboard layout and page navigation.
- **matching.py**             Computes regional market suitability scores to rank PC4 areas for a specific car model.
- **utils_common.py**         Shared functions for BigQuery connections and configuration.
- **requirements.txt**        Lists the necessary Python libraries for the project to run.

## ETL
Includes Extract–Transform–Load (ETL) scripts for processing and integrating raw datasets.
- **01_RDW_extraction.ipynb** Calls RDW API, retrieves relevant vehicle info. Calls another API to find image of vehicle
- **01_REGIONAL_extr_transf.py** Converts original CBS data (.gpkg), flattens nested coordinates with GeoPandas, exports as CSV
- **02_NBH_REGIONAL_cleaning.py** CBS neighbourhood statistics: redundant attributes omitted, applies pandas NA values
- **02_PC4_REGIONAL_cleaning.py** CBS postal code statistics: redundant attributes omitted, applies pandas NA values
- **02_RDW_cleaning.ipynb** Selects unique brand-model combinations, validates, aggregates, and excludes redundant info
- **02_REGIONAL_cleaning.py** Creates a conversion table to cross-reference neighbourhoods, postal codes and minicipalities
- **03_RDW_classification.ipynb** Classifies car models into six segments (compact, medium, large, mpv, suv, sports)
- **03_REGIONAL_load.py** Creates final  dataset by merging NBH and PC4 datasets, aggregating to PC4

## Features
Stores feature engineering notebooks and scripts created in Google Colab.  
Used for generating derived indicators and preparing datasets for analysis and model training.

---

### Purpose
These folders collectively form the project pipeline:
1. Data ingestion (ETL)
2. Feature creation
3. Application deployment (App).
