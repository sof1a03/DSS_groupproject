# Create final datasets, 
#       Using only needed columns
#       Transform absolute to relative values
#       Rename variables to English
#       Z-standardize values

import pandas as pd
import numpy as np

# _________1. FUNCTIONS__________
# Z-Standardize values, add to column with prefix "std_"
def standardize(df, ignore_cols=None):
    if ignore_cols is None:
        ignore_cols = []

    output = df.copy()
    cols_to_standardize = [c for c in df.columns if c not in ignore_cols]

    for col in cols_to_standardize:
        mean = df[col].mean()
        std = df[col].std()
        output[f"std_{col}"] = (df[col] - mean) / std

    return output

# _________2. PARAMETERS__________
# Lists with columns to include from datasets
incl_kim = ["pc4",
            "avg_yearly_income_k",
            "p_gasoline", 
            "p_diesel", 
            "p_hybrid", 
            "p_electric", 
            "body_hatchback", 
            "body_station", 
            "body_mpv", 
            "p_car_weight_0_to_850", 
            "p_car_weight_851_to_1150", 
            "p_car_weight_1151_to_1500", 
            "p_car_weight_1501_more"]

incl_pc4 = ["pc4",
            "aantal_inwoners_15_tot_25_jaar", 
            "aantal_inwoners_25_tot_45_jaar", 
            "aantal_inwoners_45_tot_65_jaar", 
            "aantal_inwoners_65_jaar_en_ouder", 
            "aantal_inwoners"]

incl_nbh = ["gwb_code_8",
            "g_wozbag", 
            "g_hhgro", 
            "ste_mvs"]

incl_con = ["pc4",
            "nbh_code"]


# Dictionary for renaming/translating columns
rename_pc4 = {
    "aantal_inwoners_15_tot_25_jaar": "p_inhb_15_to_25_year",
    "aantal_inwoners_25_tot_45_jaar": "p_inhb_25_to_45_year",
    "aantal_inwoners_45_tot_65_jaar": "p_inhb_45_to_65_year",
    "aantal_inwoners_65_jaar_en_ouder": "p_inhb_65_year_older",
    "aantal_inwoners": "inhabitants_total"
}

rename_nbh = {
    "gwb_code_8": "nbh_code",
    "g_wozbag": "avg_house_value_woz",
    "g_hhgro": "avg_household_size",
    "ste_mvs": "urbanization"
}


# _________3. IMPORT__________
pc4 = pd.read_csv("data_clean/fact_pc4.csv", usecols=incl_pc4, dtype={"pc4": str})
nbh = pd.read_csv("data_clean/fact_neighbourhoods.csv", usecols=incl_nbh,  dtype={"nbh_code": str})
kim = pd.read_csv("data_clean/fact_kim.csv", usecols=incl_kim, dtype={"pc4": str})
con = pd.read_csv("data_clean/key_lookup.csv", usecols=incl_con, dtype=str).drop_duplicates(subset="nbh_code")


# _________4. RENAME/TRANSFORM__________
# Transform absolute values to relative
cols = ["aantal_inwoners_15_tot_25_jaar", 
        "aantal_inwoners_25_tot_45_jaar", 
        "aantal_inwoners_45_tot_65_jaar", 
        "aantal_inwoners_65_jaar_en_ouder"]
mask = pc4["aantal_inwoners"].notna() & pc4[cols].notna().any(axis=1)
pc4.loc[mask, cols] = pc4.loc[mask, cols].div(pc4.loc[mask, "aantal_inwoners"], axis=0)

# Rename columns
pc4 = pc4.rename(columns=rename_pc4)
nbh = nbh.rename(columns=rename_nbh)

# _________5. STANDARDIZE__________
pc4 = standardize(pc4, ignore_cols=["pc4", "inhabitants_total"])
nbh = standardize(nbh, ignore_cols=["nbh_code"])
kim = standardize(kim, ignore_cols=["pc4"])

# _________6. MERGE__________
merged = pd.merge(nbh, con, how="right", on="nbh_code")
merged = pd.merge(merged, kim, how="left", on="pc4")
merged = pd.merge(merged, pc4, how="left", on="pc4").dropna()


# _________7. ASSIGN CAR TYPE PROBABILITY SCORE__________

# weights DataFrame
weights = pd.DataFrame({
    "feature": [
        "body_hatchback", "body_station", "body_mpv",
        "avg_household_size", "p_car_weight_0_to_850",
        "p_car_weight_851_to_1150", "p_car_weight_1151_to_1500",
        "p_car_weight_1501_more", "urbanization"
    ],
    "compact": [3, -1.5, 5, -2, 4, 0, 5, 3, -1],
    "medium": [0.23, 0.75, -1.8, 1.5, 0.75, 2.25, -3, -1, 1.67],
    "large": [-1.62, 3, -1.8, 2, -1.5, 0, 2.33, 1, 0],
    "suv": [-3, 1.5, -1.8, 1.5, -3, -3, 3, 3, -3],
    "mpv": [-1.62, 0.75, 3, 3, -1.5, 0.75, 3, -0.33, -0.67],
    "sports": [-3, -3, -3, 0, -0.75, 0, 2.33, 0.33, 1.33]
})

# Columns in merged corresponding to features in weights
std_cols = [
    "std_body_hatchback", "std_body_station", "std_body_mpv",
    "std_avg_household_size", "std_p_car_weight_0_to_850",
    "std_p_car_weight_851_to_1150", "std_p_car_weight_1151_to_1500",
    "std_p_car_weight_1501_more", "std_urbanization"
]

# Make sure merged columns are numeric
merged[std_cols] = merged[std_cols].apply(pd.to_numeric, errors="coerce")

# Compute normalized Manhattan distances
for car_type in ["compact", "medium", "large", "suv", "mpv", "sports"]:
    # Get weight vector in the same order as std_cols
    w = weights.set_index("feature")[car_type].reindex([c.replace("std_", "") for c in std_cols]).to_numpy(dtype=float)
    distances = np.nansum(np.abs(merged[std_cols].to_numpy(dtype=float) - w), axis=1)
    min_d, max_d = distances.min(), distances.max()
    merged[f"p_{car_type}"] = (distances - min_d) / (max_d - min_d)


# _________8. EXPORT__________
merged.to_csv("data_final/regional.csv", index=False)
