# RDW to BigQuery

We are building a personalized database of car models using the free **Dutch registered vehicles dataset (RDW)**. 

Source: https://opendata.rdw.nl/Voertuigen/Open-Data-RDW-Gekentekende_voertuigen/m9d7-ebf2/about_data.

---

### 01_RDW_extraction.ipynb
Fetch RDW vehicles/fuels via **Socrata**, engineer fields, and optionally enrich with images. 

Set YEAR_START/YEAR_END (and RDW_APP_TOKEN/GOOGLE keys if needed), then run all cells in Colab. Outputs CSVs like `brand_model_peryear_with_images.csv`.

For image_url Google API searches, you need two credentials: a **Custom Search API key** and a **Custom Search Engine (CSE) ID**. Follow the steps below to set them up.

Create a Google Cloud project → enable “Custom Search API” → go to **APIs & Services → Credentials** and create an **API key** (this is your `GOOGLE_API_KEY`). 

At https://cse.google.com/cse/ create a Custom Search Engine (choose “Search the entire web” or add specific sites) and copy its **Search engine ID** (`GOOGLE_CSE_ID`).  

Set them in your environment/Colab (e.g., `os.environ["GOOGLE_API_KEY"]="..."`; `os.environ["GOOGLE_CSE_ID"]="..."`) or I can provide both keys in an encrypted file if needed.

---
                                                                                                                   
### 02_RDW_cleaning.ipynb
Load the aggregated CSV, drop coachbuilders/low-frequency brands, and normalize/dedupe model names per brand. Run all cells to print before/after counts and preserve original metrics. 

Save the cleaned table to `df_cleaned_FINAL.csv`.

---

### 03_RDW_classification.ipynb
Read `df_cleaned_FINAL.csv`, filter non-passenger body types, one-hot encode fuels, compute clipped Z-score price, and classify into SUV/MPV/Sports/Compact/Medium/Large using mass/length heuristics. 

Run all cells to view distributions and `body_*` dummies. Export `rdw_classified.csv` for downstream use.

---

### 04_RDW_analysis.ipynb
Load `df_cleaned_FINAL.csv`, engineer `avg_price`, `fuel_bucket`, `intro_year`, share metrics, and build EDA (missingness, distributions, new vs continuing, scatterplots), brand HHI, clustering, naive 2-year brand forecasts, and Spearman correlations. Adjust `CSV` and `YEAR_START/YEAR_END`, then run all cells; plots render inline and tables show key summaries. Use insights to validate classification rules and spot price/segment trends.

The final dataset was added to our **BigQuery** project environment.

---

## BigQuery — Load Examples (RDW)

```bash
gcloud config set project YOUR_PROJECT_ID
bq mk --location=EU RDW

bq load --autodetect --source_format=CSV RDW.rdw_classified ./rdw_classified.csv


