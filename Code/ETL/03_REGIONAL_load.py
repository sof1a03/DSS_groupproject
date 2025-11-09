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
        # Calculate mean and standard deviation, handling potential NaNs
        mean = df[col].mean()
        std = df[col].std()
        
        # Only standardize if standard deviation is not zero
        if std != 0:
            output[f"std_{col}"] = (df[col] - mean) / std
        else:
            # If std is 0, all values are the same; standardized value is 0 (or NaN if mean is NaN)
            output[f"std_{col}"] = 0.0 if not pd.isna(mean) else np.nan

    return output

# _________2. PARAMETERS (UPDATED)__________
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

# ADDED new columns for imputation from PC4 data
incl_pc4 = ["pc4",
            "aantal_inwoners_15_tot_25_jaar", 
            "aantal_inwoners_25_tot_45_jaar", 
            "aantal_inwoners_45_tot_65_jaar", 
            "aantal_inwoners_65_jaar_en_ouder", 
            "aantal_inwoners",
            "gemiddelde_huishoudensgrootte",
            "gemiddelde_woz_waarde_woning"] 

incl_nbh = ["gwb_code_8",
            "g_wozbag", 
            "g_hhgro", 
            "ste_mvs",
            "a_inw"]

incl_con = ["pc4",
            "nbh_code",
            "municipality_name"]

# Dictionary for renaming/translating columns (UPDATED)
rename_pc4 = {
    "aantal_inwoners_15_tot_25_jaar": "p_inhb_15_to_25_year",
    "aantal_inwoners_25_tot_45_jaar": "p_inhb_25_to_45_year",
    "aantal_inwoners_45_tot_65_jaar": "p_inhb_45_to_65_year",
    "aantal_inwoners_65_jaar_en_ouder": "p_inhb_65_year_older",
    "aantal_inwoners": "inhabitants_total",
    # Temporary names for imputation source columns
    "gemiddelde_huishoudensgrootte": "pc4_avg_household_size",
    "gemiddelde_woz_waarde_woning": "pc4_avg_house_value_woz"
}

rename_nbh = {
    "gwb_code_8": "nbh_code",
    "g_wozbag": "avg_house_value_woz",
    "g_hhgro": "avg_household_size",
    "ste_mvs": "urbanization",
    "a_inw" : "inhabitants_nbh"
}

# Define columns involved in NBH/PC4 imputation (for imputation and clean-up)
impute_cols_raw = ["avg_household_size", "avg_house_value_woz"]
impute_cols_pc4 = ["pc4_avg_household_size", "pc4_avg_house_value_woz"]
impute_cols_std = [f"std_{col}" for col in impute_cols_raw]
impute_cols_pc4_std = [f"std_{col}" for col in impute_cols_pc4]


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

# Rename columns and remove nbh with no population
pc4 = pc4.rename(columns=rename_pc4)
nbh = nbh.rename(columns=rename_nbh)
nbh = nbh[nbh['inhabitants_nbh'] != 0].copy()


# _________5. STANDARDIZE__________
pc4 = standardize(pc4, ignore_cols=["pc4", "inhabitants_total"]) 
nbh = standardize(nbh, ignore_cols=["nbh_code"])
kim = standardize(kim, ignore_cols=["pc4"])


# _________6. MERGE AND IMPUTATION__________
merged = con.copy()
merged = merged.merge(nbh, how="left", on="nbh_code")
merged = merged.merge(kim, how="left", on="pc4") 
merged = merged.merge(pc4, how="left", on="pc4") 

# IMPUTATION LOGIC: Fill missing NBH values with PC4 values 
for raw_col, pc4_col in zip(impute_cols_raw, impute_cols_pc4):
    std_raw_col = f"std_{raw_col}"
    std_pc4_col = f"std_{pc4_col}"

    # Impute raw values, impute standardized values (using the standardized PC4 columns)
    merged[raw_col] = merged[raw_col].fillna(merged[pc4_col])
    merged[std_raw_col] = merged[std_raw_col].fillna(merged[std_pc4_col])


# Drop the temporary PC4 imputation columns 
merged = merged.drop(columns=impute_cols_pc4 + impute_cols_pc4_std, errors='ignore')

# --- Display rows that were either 0 or NA for inhabitants_nbh (for verification) ---
filtered_df = merged[(merged['inhabitants_nbh'] == 0) | (merged['inhabitants_nbh'].isna())]
pc4_values = filtered_df['pc4'].unique()
print("Unique 'pc4' values where 'inhabitants_nbh' is 0 or NA:")
print(pc4_values)


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

# Make sure merged columns are numeric (critical step to prevent previous error)
merged[std_cols] = merged[std_cols].apply(pd.to_numeric, errors="coerce")

# Compute normalized Manhattan distances
for car_type in ["compact", "medium", "large", "suv", "mpv", "sports"]:
    # Get weight vector in the same order as std_cols
    w = weights.set_index("feature")[car_type].reindex([c.replace("std_", "") for c in std_cols]).to_numpy(dtype=float)
    distances = np.nansum(np.abs(merged[std_cols].to_numpy(dtype=float) - w), axis=1)
    min_d, max_d = distances.min(), distances.max()
    # Normalize distances (0 to 1), then subtract from 1 to get "probability" (higher score = better match)
    merged[f"p_{car_type}"] = 1 - ((distances - min_d) / (max_d - min_d)) 

# _________8. AGGREGATE TO PC4-LEVEL, WEIGHTED BY NBH_INHABITANTS__________

numeric_cols = [
    'avg_household_size', 'avg_house_value_woz', 'urbanization', 
    'std_avg_household_size', 'std_avg_house_value_woz', 'std_urbanization',
    'p_gasoline', 'p_diesel', 'p_electric', 'p_hybrid', 'avg_yearly_income_k', 
    'p_car_weight_0_to_850', 'p_car_weight_851_to_1150', 'p_car_weight_1151_to_1500',
    'p_car_weight_1501_more', 'body_hatchback', 'body_station', 'body_mpv',
    'std_p_gasoline', 'std_p_diesel', 'std_p_electric', 'std_p_hybrid',
    'std_avg_yearly_income_k', 'std_p_car_weight_0_to_850',
    'std_p_car_weight_851_to_1150', 'std_p_car_weight_1151_to_1500',
    'std_p_car_weight_1501_more', 'std_body_hatchback', 'std_body_station',
    'std_body_mpv', 'inhabitants_total', 'p_inhb_15_to_25_year',
    'p_inhb_25_to_45_year', 'p_inhb_45_to_65_year', 'p_inhb_65_year_older',
    'std_p_inhb_15_to_25_year', 'std_p_inhb_25_to_45_year',
    'std_p_inhb_45_to_65_year', 'std_p_inhb_65_year_older', 'p_compact',
    'p_medium', 'p_large', 'p_suv', 'p_mpv', 'p_sports'
]

# Function to calculate weighted average
def weighted_avg(group, column, weight_column):
    # Ensure weights are not NaN (NumPy's average handles this implicitly, 
    # but explicit handling is safer for sum check)
    valid_data_mask = group[column].notna() & group[weight_column].notna()
    weights = group.loc[valid_data_mask, weight_column]
    data = group.loc[valid_data_mask, column]
    
    if weights.sum() == 0 or weights.empty:
        # Fallback to unweighted mean if weights are all zero or missing
        return group[column].mean() 
    return np.average(data, weights=weights)

weight_col = 'inhabitants_nbh'

merged_aggregated_weighted = merged.groupby('pc4').apply(
    lambda group: pd.Series({
        col: weighted_avg(group, col, weight_col) for col in numeric_cols
    })
).reset_index()

merged_aggregated_weighted = merged_aggregated_weighted

# _________9. EXPORT__________
merged_aggregated_weighted.to_csv("data_final/REGIONAL_PC4_2.csv", index=False)
print("DONE")
