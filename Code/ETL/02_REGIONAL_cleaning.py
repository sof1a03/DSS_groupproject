# Create a table for Primary Key conversions (postal code, neighbourhood code, municipality, associated names)
import pandas as pd

# Load/clean pc6 conversion: duplicates, add pc4
c = pd.read_csv("data_raw/key_conversion/pc6-conversion.csv", sep=";", dtype=str)
c.drop(columns="Huisnummer", inplace=True)
c = c[~c.duplicated(subset=["PC6"], keep="first")]
c['PC4'] = c['PC6'].str[:4]

gemnamen = pd.read_csv("data_raw/key_conversion/gem_2025.csv", sep=";", dtype=str)
wijknamen = pd.read_csv("data_raw/key_conversion/wijk_2025.csv", sep=";", dtype=str)
buurtnamen = pd.read_csv("data_raw/key_conversion/buurt_2025.csv", sep=";", dtype=str)

# Merge
c = pd.merge(left=c, right=buurtnamen, on="Buurt2025", how="left")
c = pd.merge(left=c, right=wijknamen, on="Wijk2025", how="left")
c = pd.merge(left=c, right=gemnamen, on="Gemeente2025", how="left")

# clean: NA, drop buurtcode, rename/reorder cols
c = c.fillna(pd.NA)
#c = c.drop(columns=["Buurt2025_y", "Wijk2025_y"])

c.columns = ["pc6", "neighbourhood_code", "district_code", "municipality_code", "pc4", "neighbourhood_name", "district_name", "municipality_name"]
col_order = ["pc6", "pc4", "neighbourhood_code", "district_code", "municipality_code", "neighbourhood_name", "district_name", "municipality_name"]
c = c[col_order]

# save
c.to_csv("data_clean/pc6-conversion.csv", index=False)

check_codes = c[['pc6', 'pc4', 'district_code', 'neighbourhood_code', 'municipality_code']].nunique()
check_names = c[['pc6', 'pc4', 'district_name', 'neighbourhood_name', 'municipality_name']].nunique()