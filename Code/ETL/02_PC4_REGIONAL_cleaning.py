# Preprocess CBS demographic data for Postal code 4
import pandas as pd

# Load data, set NA values
def load_pc_data(csv_path, usecols=None, drop_geom=True):
    df = pd.read_csv(csv_path, quotechar='"', usecols=usecols)
    df = df.replace(["-99997", "-99995.0", "-99995", -99997, -99995, -99995.0], pd.NA)
    if drop_geom:
        df = df.drop(columns=["geometry"])
    return df

# Count number of NAs
def na_counts(df):
    total_rows = len(df)    
    na_counts = df.isna().sum()
    df_na_counts = na_counts.reset_index()
    df_na_counts.columns = ['field', 'NA_count']
    df_na_counts['NA_percentage'] = (df_na_counts['NA_count'] / total_rows) * 100
    df_na_counts.to_csv("NA_counts.csv", index=False)
    return df_na_counts

# Function to export cleaned data
def export_csv(df, path):
    df.to_csv(path, index=False)

# Specify columns to include
pc4_cols = ["postcode",
            # population
            "aantal_inwoners","omgevingsadressendichtheid","aantal_mannen","aantal_vrouwen",
            # age groups
            "aantal_inwoners_0_tot_15_jaar","aantal_inwoners_15_tot_25_jaar", 
            "aantal_inwoners_25_tot_45_jaar","aantal_inwoners_45_tot_65_jaar",
            "aantal_inwoners_65_jaar_en_ouder",
            # household
            "aantal_part_huishoudens", "gemiddelde_huishoudensgrootte",
            "aantal_eenpersoonshuishoudens", "aantal_meergezins_woningen", 
            "aantal_meerpersoonshuishoudens_zonder_kind",
            "aantal_eenouderhuishoudens","aantal_tweeouderhuishoudens", 
            # income
            "aantal_personen_met_uitkering_onder_aowlft",
             # origin
            "percentage_geb_nederland_herkomst_nederland","percentage_geb_nederland_herkomst_overig_europa", 
            "percentage_geb_nederland_herkomst_buiten_europa","percentage_geb_buiten_nederland_herkomst_europa", 
            "percentage_geb_buiten_nederland_herkmst_buiten_europa",
            # housing
            "gemiddelde_woz_waarde_woning", 
            "aantal_woningen", "percentage_koopwoningen","percentage_huurwoningen", "aantal_huurwoningen_in_bezit_woningcorporaties", 
            "aantal_niet_bewoonde_woningen", "stedelijkheid",
            # construction year
            "aantal_woningen_bouwjaar_voor_1945", "aantal_woningen_bouwjaar_45_tot_65",
            "aantal_woningen_bouwjaar_65_tot_75","aantal_woningen_bouwjaar_75_tot_85", 
            "aantal_woningen_bouwjaar_85_tot_95","aantal_woningen_bouwjaar_95_tot_05",
            "aantal_woningen_bouwjaar_05_tot_15", "aantal_woningen_bouwjaar_15_en_later"]

# Call functions
pc4 = load_pc_data("data_raw/cbs_pc4_2023.csv", pc4_cols, drop_geom=False)
export_csv(pc4, "data_clean/fact_pc4.csv")