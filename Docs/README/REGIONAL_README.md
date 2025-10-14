# REGIONAL & MAP — CBS + KiM to BigQuery

We are building two regional datasets for the frontend from **free Dutch sources**:
- **CBS** (Statistics Netherlands): PC6 geodata, PC4 stats, and Wijken/Buurten.
- **KiM** (Netherlands Institute for Transport Policy Analysis): PC4-level mobility/fleet.

Final databases in **BigQuery**:
- **MAP**: heatmap-ready PC6 + PC6→PC4/admin names.
- **REGIONAL**: analytics (PC4, neighbourhood, KiM) including Z-scores.

---

## Scripts

### 01_REGIONAL_extr_transf.py
Convert CBS GeoPackage to a flat CSV for BigQuery.

- **Input:** `data_raw/cbs_pc6_2023.gpkg`
- **Run:** `python 01_REGIONAL_extr_transf.py`
- **What it does:** Reads the GeoPackage, selects key demographic/housing columns, flattens geometry.
- **Output (for MAP):** `./cbs_pc6_2023.csv` (heatmap base)

---

### 02_REGIONAL_cleaning.py
Build a **PC6→PC4→(buurt/wijk/gemeente)** conversion table with names.

- **Inputs:**  
  `data_raw/key_conversion/pc6-conversion.csv`,  
  `data_raw/key_conversion/gem_2025.csv`, `wijk_2025.csv`, `buurt_2025.csv`
- **Run:** `python 02_REGIONAL_cleaning.py`
- **What it does:** Deduplicates PC6, derives `pc4 = pc6[:4]`, merges admin **codes + names**, cleans NAs.
- **Output (for MAP):** `data_clean/pc6-conversion.csv`

---

### 02_NBH_REGIONAL_cleaning.py
Extract **neighbourhood** (“Buurt”) facts from CBS KWB.

- **Input:** `data_raw/cbs-kwb-2024.csv`
- **Run:** `python 02_NBH_REGIONAL_cleaning.py`
- **What it does:** Filters to `recs == 'Buurt'`, selects indicators (WOZ, household size, urbanization), normalizes NAs.
- **Output (for REGIONAL):** `data_clean/fact_neighbourhoods.csv`

---

### 02_PC4_REGIONAL_cleaning.py
Prepare **PC4**-level CBS indicators.

- **Input:** `data_raw/cbs_pc4_2023.csv`
- **Run:** `python 02_PC4_REGIONAL_cleaning.py`
- **What it does:** Selects PC4 columns (population, age groups, housing, income), replaces CBS NA sentinels, drops geometry.
- **Outputs (for REGIONAL):**  
  `data_clean/fact_pc4.csv`  
  `NA_counts.csv` (quick missingness audit)

---

### 03_REGIONAL_classification.py
Create **final, standardized** CSVs for BigQuery (with `std_` Z-scores).

- **Inputs:**  
  `data_clean/fact_pc4.csv` (CBS),  
  `data_clean/fact_neighbourhoods.csv` (CBS),  
  `data_clean/fact_kim.csv` (**KiM**, PC4-level fleet & income)
- **Run:** `python 03_REGIONAL_classification.py`
- **What it does:**  
  - PC4: convert age **counts → proportions** of total inhabitants; rename to English; Z-score (`std_`*).  
  - Neighbourhood: rename (WOZ, household size, urbanization); Z-score.  
  - KiM: keep fuel mix/body/weight/income; Z-score.
- **Outputs (for REGIONAL):**  
  `final_data/pc4.csv`  
  `final_data/nbh.csv`  
  `final_data/kim.csv`

> **Note (PC4 proportions):**
> ```python
> cols = ["aantal_inwoners_15_tot_25_jaar", "aantal_inwoners_25_tot_45_jaar",
>         "aantal_inwoners_45_tot_65_jaar", "aantal_inwoners_65_jaar_en_ouder"]
> mask = pc4["aantal_inwoners"].notna()
> pc4.loc[mask, cols] = pc4.loc[mask, cols].div(pc4.loc[mask, "aantal_inwoners"], axis=0)
> ```

---

## BigQuery Layout

### Dataset: `MAP`
- **`MAP.cbs_pc6_2023`** — flattened PC6 metrics for the heatmap.
- **`MAP.pc6_conversion`** — `pc6`, `pc4`, neighbourhood/district/municipality **codes + names**.

### Dataset: `REGIONAL`
- **`REGIONAL.kim`** (from **KiM**; PC4):  
  `pc4, p_gasoline, p_diesel, p_electric, p_hybrid, avg_yearly_income_k, p_car_weight_0_to_850, p_car_weight_851_to_1150, p_car_weight_1151_to_1500, p_car_weight_1501_more, body_hatchback, body_station, body_mpv` + `std_*`.
- **`REGIONAL.nbh`** (from **CBS** KWB; neighbourhood):  
  `nbh_code, avg_household_size, avg_house_value_woz, urbanization` + `std_*`.
- **`REGIONAL.pc4`** (from **CBS** PC4):  
  `p_inhb_15_to_25_year, p_inhb_25_to_45_year, p_inhb_45_to_65_year, p_inhb_65_year_older` + `std_*`.

---

## Load (example)

```bash
gcloud config set project YOUR_PROJECT_ID
bq mk --location=EU MAP
bq mk --location=EU REGIONAL

bq load --autodetect --source_format=CSV MAP.cbs_pc6_2023 ./cbs_pc6_2023.csv
bq load --autodetect --source_format=CSV MAP.pc6_conversion ./data_clean/pc6-conversion.csv

bq load --autodetect --source_format=CSV REGIONAL.pc4 ./final_data/pc4.csv
bq load --autodetect --source_format=CSV REGIONAL.nbh ./final_data/nbh.csv
bq load --autodetect --source_format=CSV REGIONAL.kim ./final_data/kim.csv
