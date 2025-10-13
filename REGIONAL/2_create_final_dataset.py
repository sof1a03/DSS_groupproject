# Create final datasets, 
#       Using only needed columns
#       Transform absolute to relative values
#       Rename variables to English
#       Z-standardize values

import pandas as pd

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
pc4 = pd.read_csv("data_clean/fact_pc4.csv", usecols=incl_pc4)
nbh = pd.read_csv("data_clean/fact_neighbourhoods.csv", usecols=incl_nbh)
kim = pd.read_csv("data_clean/fact_kim.csv", usecols=incl_kim)


# _________4. RENAME/TRANSFORM__________
# Transform absolute values to relative
cols = ["aantal_inwoners_15_tot_25_jaar", 
        "aantal_inwoners_25_tot_45_jaar", 
        "aantal_inwoners_45_tot_65_jaar", 
        "aantal_inwoners_65_jaar_en_ouder"]
mask = pc4["aantal_inwoners"].notna() & pc4[cols].notna().any(axis=1)
pc4 = pc4.loc[mask, cols] = pc4.loc[mask, cols].div(pc4.loc[mask, "aantal_inwoners"], axis=0)

# Rename columns
pc4 = pc4.rename(columns=rename_pc4)
nbh = nbh.rename(columns=rename_nbh)

# _________5. STANDARDIZE__________
pc4 = standardize(pc4, ignore_cols=["pc4", "inhabitants_total"])
nbh = standardize(nbh, ignore_cols=["nbh_code"])
kim = standardize(kim, ignore_cols=["pc4"])

# _________6. EXPORT__________
pc4.to_csv("final_data/pc4.csv", index=False)
nbh.to_csv("final_data/nbh.csv", index=False)
kim.to_csv("final_data/kim.csv", index=False)