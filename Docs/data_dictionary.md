# Data Dictionary
This document provides a structured description of the data sources used in the DSS-Group04 project.  
It supports the **Data Understanding & Documentation** phase of the CRISP-DM methodology.

## RDW – Gekentekende Voertuigen (Registered Vehicles)

**Source:**  
Rijksdienst voor het Wegverkeer (RDW) – Open Data Catalog  
[Gekentekende Voertuigen (2025)](https://opendata.rdw.nl/Voertuigen/Open-Data-RDW-Gekentekende_voertuigen/m9d7-ebf2/about_data)

**License:**  
Creative Commons Attribution 4.0 (CC-BY 4.0)

**Access Method:**  
- Extracted through RDW Open Data API and exported as CSV (`Data/Final/RDW.csv`)  
- Processed via ETL notebooks:  
  - `01_RDW_extraction.ipynb`  
  - `02_RDW_cleaning.ipynb`  
  - `03_RDW_classification.ipynb`  

**Main Columns:**  
| Column | Description |
|---------|-------------|
| `kenteken` | Vehicle registration number |
| `merk` | Brand / manufacturer |
| `handelsbenaming` | Commercial model name |
| `voertuigsoort` | Vehicle type (e.g., Personenauto, Bedrijfsauto) |
| `brandstof_omschrijving` | Fuel type (e.g., Benzine, Elektrisch, Diesel) |
| `massa_ledig_voertuig` | Vehicle weight (kg) |
| `catalogusprijs` | Catalog price (€) |
| `co2_uitstoot` | CO₂ emissions (g/km) |
| `datum_eerste_toelating` | First registration date |
| `carrosserie_omschrijving` | Body type (e.g., Hatchback, SUV, Sedan) |

**Known Issues:**  
- **Catalog price missingness:** frequent null values for imported, electric, or older vehicles.  
- **CO₂ emission inconsistency:** zero or null for electric models.  
- **Model name heterogeneity:** inconsistent capitalization or spelling.  
- Vehicles deregistered before 2010 may be incomplete.

**Derived Fields:**  
- `fuel_category` and `body_type_category` computed during ETL classification.  
- Average price and weight statistics aggregated at model and fuel type level.


## CBS – Demographic and Geographic Data

**Source:**  
Centraal Bureau voor de Statistiek (CBS) – Statistics Netherlands  

**Sub-datasets:**  
1. [Kerncijfers Wijken en Buurten 2024](https://www.cbs.nl/nl-nl/maatwerk/2025/38/kerncijfers-wijken-en-buurten-2024)  
2. [Kerncijfers Postcode 4 (2024)](https://www.cbs.nl/nl-nl/longread/diversen/2023/statistische-gegevens-per-vierkant-en-postcode-2022-2021-2020-2019)  
3. [Buurt, Wijk en Gemeente (2023) – Postcode + Huisnummer](https://www.cbs.nl/nl-nl/maatwerk/2023/35/buurt-wijk-en-gemeente-2023-voor-postcode-huisnummer)  

**License:**  
Creative Commons Attribution 4.0 (CC-BY 4.0)

**Access Method:**  
- Downloaded CSVs under `Data/Geo/` and `Data/Final/REGIONAL.csv`  
- Processed with:  
  - `01_REGIONAL_extr_transf.py`  
  - `02_NBH_REGIONAL_cleaning.py`  
  - `02_PC4_REGIONAL_cleaning.py`  
  - `02_REGIONAL_cleaning.py`  

**Main Columns:**  
| Column | Description |
|---------|-------------|
| `Wijkcode` / `Buurtcode` | Unique neighbourhood or district code |
| `Wijknaam` / `Buurtnaam` | Neighbourhood name |
| `Inwoners` | Number of inhabitants |
| `Gemiddelde_huishoudensgrootte` | Average household size |
| `Bevolkingsdichtheid` | Population density (inhabitants per km²) |
| `Gemiddeld_inkomen_per_inwoner` | Average income per inhabitant (€) |
| `PC4` | 4-digit postal code |
| `Oppervlakte_km2` | Surface area in square kilometres |
| `geometry` | Spatial geometry of the area (GeoJSON or WGS84) |

**Known Issues:**  
- **Data suppression:** CBS aggregates or suppresses cells for areas with fewer than **5 inhabitants** (privacy protection).  
- **Temporal inconsistency:** some region codes change annually due to administrative restructuring.  
- **Spatial mismatches:** minor geometry differences across CBS years.  

**Derived Fields:**  
- `Z_scores` for population and income distribution (calculated in ETL scripts).  
- `proportions` for car ownership and density indicators per PC4/Wijk.


## KiM – Atlas van de Auto (Mobility and Car Ownership)

**Source:**  
Kennisinstituut voor Mobiliteitsbeleid (KiM), Ministerie van Infrastructuur en Waterstaat  
[Atlas van de Auto 2024](https://www.kimnet.nl/atlas-van-de-auto#auto-op-de-kaart)

**License:**  
Open data for research and public policy; attribution required (CC-BY 4.0 equivalent).

**Access Method:**  
Public web-based dashboard with downloadable tables.  
Extracted manually for integration in the `REGIONAL` dataset as reference indicators.

**Main Columns:**  
| Column | Description |
|---------|-------------|
| `jaar` | Year of observation |
| `auto_per_huishouden` | Average cars per household |
| `auto_per_100_inwoners` | Number of cars per 100 inhabitants |
| `brandstofmix_pers_auto` | Share of fuel types in private cars |
| `leeftijd_autopark` | Average age of vehicle fleet |
| `provincie` / `gemeente` | Administrative region |

**Known Issues:**  
- Aggregated data: no microdata at vehicle level.  
- Possible rounding differences across editions.  
- Regional breakdown not always aligned with CBS boundaries.


## Summary Table

| Source | License | Access Method | Main File | Known Issues |
|--------|----------|----------------|------------|----------------|
| **RDW** | CC-BY 4.0 | API → CSV → BigQuery | `RDW.csv` | Missing catalog prices; inconsistent model names |
| **CBS** | CC-BY 4.0 | CSV/GeoJSON → Python ETL | `REGIONAL.csv`, `CBS_NBH_HEAD_10.csv` | Data suppression <5 inhabitants; code changes |
| **KiM** | CC-BY 4.0 | Web extraction → CSV merge | integrated in `REGIONAL.csv` | Aggregated data; rounding differences |
