# Convert .gpkg files (original filetype CBS) to csv (flattened, columnar) => needed for import in BigQuery
import pandas as pd
import geopandas as gpd

# Reading a GeoPackage file
input_file = "cbs_pc4_2024.gpkg"
data = gpd.read_file(input_file)

cols = (
    'aantal_inwoners', 'percentage_koopwoningen', 'percentage_huurwoningen',
    'aantal_mannen', 'aantal_vrouwen', 'aantal_inwoners_0_tot_15_jaar',
    'aantal_inwoners_15_tot_25_jaar', 'aantal_inwoners_25_tot_45_jaar',
    'aantal_inwoners_45_tot_65_jaar', 'aantal_inwoners_65_jaar_en_ouder',
    'percentage_geb_nederland_herkomst_nederland', 'percentage_geb_nederland_herkomst_overig_europa',
    'percentage_geb_nederland_herkomst_buiten_europa', 'percentage_geb_buiten_nederland_herkomst_europa',
    'percentage_geb_buiten_nederland_herkmst_buiten_europa', 'aantal_part_huishoudens',
    'aantal_eenpersoonshuishoudens', 'aantal_meerpersoonshuishoudens_zonder_kind',
    'aantal_eenouderhuishoudens', 'aantal_tweeouderhuishoudens', 'gemiddelde_huishoudensgrootte',
    'aantal_woningen', 'aantal_woningen_bouwjaar_voor_1945', 'aantal_woningen_bouwjaar_45_tot_65',
    'aantal_woningen_bouwjaar_65_tot_75', 'aantal_woningen_bouwjaar_75_tot_85',
    'aantal_woningen_bouwjaar_85_tot_95', 'aantal_woningen_bouwjaar_95_tot_05',
    'aantal_woningen_bouwjaar_05_tot_15', 'aantal_woningen_bouwjaar_15_en_later',
    'aantal_meergezins_woningen', 'aantal_huurwoningen_in_bezit_woningcorporaties',
    'aantal_niet_bewoonde_woningen', 'gemiddelde_woz_waarde_woning',
    'aantal_personen_met_uitkering_onder_aowlft', 'geometry', 'postcode6'
)

df = pd.DataFrame(data, columns=cols)
df.to_csv("cbs_pc4_2024.csv", index=False)
