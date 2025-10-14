# REGIONAL & MAP — CBS + KiM to BigQuery

We are building two regional datasets for the frontend based on **open Dutch data sources**. These datasets combine spatial, demographic, and mobility information to support both visualization (MAP) and analytics (REGIONAL):

- **CBS (Centraal Bureau voor de Statistiek)**: provides postal code–level geodata (PC6 and PC4), neighbourhood and municipality statistics, and the conversion tables linking PC4 to neighbourhood codes.  
- **KiM (Netherlands Institute for Transport Policy Analysis)**: provides mobility and vehicle fleet data aggregated at the PC4 level.

The processed datasets are stored in **BigQuery**:

- **MAP**: contains PC6-level data for visualizations and the PC6→PC4→neighbourhood conversion tables required for the frontend heatmap.  
- **REGIONAL**: contains standardized analytical data by combining CBS demographic indicators, neighbourhood-level statistics, and KiM mobility and fleet variables at PC4 level.

---

## Workflow Overview

The process consists of four main stages:

1. Geodata Extraction & Flattening — Convert CBS GeoPackage (PC4) to CSV in order to have the dataset for the map interface.  
2. Cleaning & Transformation — Process PC4, neighbourhood, and conversion PC6/PC4-nbh tables.  
3. Linking & Integration — Merge PC4, NBH, and KiM datasets on shared keys using the conversion PC6/PC4-nbh table.  
4. Final Standardization & Loading — Create final REGIONAL tables with Z-scores.

---

## Scripts & Pipeline

### 01_REGIONAL_extr_transf.py
**Extract** PC4-level geodata from CBS GeoPackage and convert it into CSV for downstream processing.

- **Input:** `data_raw/cbs_pc4_2023.gpkg`
- **Run:**  
  ```bash
  python 01_REGIONAL_extr_transf.py
- **Output:** data_intermediate/cbs_pc4_2023.csv

### 02_PC4_REGIONAL_cleaning.py

**Clean and prepare** PC4-level CBS indicators for integration. Selects demographic, housing, and income indicators. Replaces sentinel NA values and drops unused columns. Prepares data for merging.

- **Input**: data_intermediate/cbs_pc4_2023.csv

- **Run**:
 ```bash
  python 02_PC4_REGIONAL_cleaning.py

- **Output:** data_clean/fact_pc4.csv

### 02_NBH_REGIONAL_cleaning.py

**Process neighbourhood-level (Buurt)** data from CBS Kerncijfers. Filters neighbourhood records. Extracts indicators like WOZ value, household size, and urbanization. Handles NA values and prepares GEO join keys.

- **Input:** data_raw/cbs_kwb_2024.csv

- **Run:**
 ```bash
 python 02_NBH_REGIONAL_cleaning.py

- **Output:** data_clean/fact_neighbourhoods.csv

### 02_REGIONAL_cleaning.py

**Create** PC6→PC4→Buurt/Wijk/Gemeente conversion tables.

- **Inputs:** data_raw/key_conversion/pc6-conversion.csv 

data_raw/key_conversion/gem_2025.csv, wijk_2025.csv, buurt_2025.csv

- **Run:**
 ```bash
 python 02_REGIONAL_cleaning.py

- **Output:** data_clean/pc6-conversion.csv

### 03_REGIONAL_load.py

**Integrate all components** — PC4, NBH, and KiM — into the final REGIONAL dataset. Merges datasets using shared keys. Converts counts to proportions (e.g., age distribution). Renames columns  to English. Applies Z-standardization to all continuous variables. Merge all into a single table. 

- **Inputs:** data_clean/fact_pc4.csv and data_clean/fact_neighbourhoods.csv and data_clean/fact_kim.csv (hand-processed KiM data)

- **Run:**
 ```bash
 python 03_REGIONAL_load.py

- **Output:** 







