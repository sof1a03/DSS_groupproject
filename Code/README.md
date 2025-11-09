# Code Repository Overview

This section of the repository is organized into three main components:

## App
Contains the application layer, including scripts for running the dashboard, APIs, and visualization interfaces. 
It consists of the following documents:
* **Dockerfile**              Builds the Docker image containing the Python environment and application code.
- **analysis.py**             Calculates time series forecasts for RDW vehicle registration data.
- **docker-compose.yml**      Defines the configuration for running the application in a multi-container environment.
- **ev.py**                   Runs EV market analysis, including clustering and what-if scenario simulations for regions.
- **main.py**                 The main Streamlit application file that controls the dashboard layout and page navigation.
- **matching.py**             Computes regional market suitability scores to rank PC4 areas for a specific car model.
- **utils_common.py**         Shared functions for BigQuery connections and configuration.
- **requirements.txt**        Lists the necessary Python libraries for the project to run.

## ETL
Includes Extract–Transform–Load (ETL) scripts for processing and integrating raw datasets.
01_RDW_extraction.ipynb
01_REGIONAL_extr_transf.py
02_NBH_REGIONAL_cleaning.py
02_PC4_REGIONAL_cleaning.py
02_RDW_cleaning.ipynb
02_REGIONAL_cleaning.py
03_RDW_classification.ipynb
03_REGIONAL_load.py
Car_Classification.ipynb

## Features
Stores feature engineering notebooks and scripts created in Google Colab.  
Used for generating derived indicators and preparing datasets for analysis and model training.

---

### Purpose
These folders collectively form the project pipeline — from data ingestion (ETL) to feature creation and final application deployment (App).
