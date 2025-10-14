# Preprocess dataset (districts & neighborhoods)
import pandas as pd

# Specify columns to import
import_cols = ['gwb_code_10', 'gwb_code_8', 'recs', 'gwb_code', 'ind_wbi', 'a_inw', 'a_man', 'a_vrouw', 'a_00_14', 'a_15_24', 'a_25_44', 'a_45_64', 'a_65_oo', 'a_ongeh', 'a_gehuwd', 'a_gesch', 'a_verwed', 'a_nl_all', 'a_eur_al', 'a_neu_al', 'a_geb_nl', 'a_geb_eu', 'a_geb_ne', 'a_gbl_eu', 'a_gbl_ne', 'a_geb', 'p_geb', 'a_ste', 'p_ste', 'a_hh', 'a_1p_hh', 'a_hh_z_k', 'a_hh_m_k', 'g_hhgro', 'bev_dich', 'a_woning', 'a_nb_won ', 'a_vastg', 'a_nb_vastg', 'g_wozbag', 'p_1gezw', 'p_1gezw_tw', 'p_1gezw_hw', 'p_1gezw_2w', 'p_1gezw_hvw', 'p_mgezw', 'p_leegsw', 'p_koopw', 'p_huurw', 'p_wcorpw', 'p_ov_hw', 'p_bj_me10', 'p_bj_mi10', 'g_ele', 'g_ele_tr', 'g_gas', 'p_stadsv', 'p_won_z_ag', 'p_won_m_ag ', 'p_won_zs', 'p_won_ev ', 'a_lp_pub', 'a_ons_po', 'a_ons_vovavo', 'a_ons_mbo', 'a_ons_hbo', 'a_ons_wo', 'a_opl_bvm', 'a_opl_hvm', 'a_opl_hw', 'a_arb_wz', 'p_arb_pp', 'p_arb_wn', 'p_arb_wnv', 'p_arb_wnf', 'p_arb_zs', 'a_inkont', 'g_ink_po', 'g_ink_pi', 'p_ink_li', 'p_ink_hi', 'g_hh_sti', 'p_hh_li', 'p_hh_hi', 'p_hh_lkk', 'p_hh_osm', 'p_hh_110', 'p_hh_120', 'm_hh_ver', 'a_soz_wb', 'a_soz_ao', 'a_soz_ww', 'a_soz_ow', 'a_jz_tn', 'p_jz_tn', 'a_wmo_t', 'p_wmo_t', 'a_bedv', 'a_bed_a', 'a_bed_bf', 'a_bed_gi', 'a_bed_hj', 'a_bed_kl', 'a_bed_mn', 'a_bed_oq', 'a_bed_ru', 'a_pau', 'a_bst_b', 'a_bst_nb', 'g_pau_hh', 'g_pau_km', 'a_m2w', 'g_afs_hp', 'g_afs_gs', 'g_afs_kv', 'g_afs_sc', 'g_3km_sc', 'a_opp_ha', 'a_lan_ha', 'a_wat_ha', 'pst_mvp', 'pst_dekp', 'ste_mvs', 'ste_oad']

# Import raw data
nd = pd.read_csv("data_raw/cbs-kwb-2024.csv", 
                  delimiter=";", 
                  decimal=".", 
                  quotechar='"',
                  usecols=import_cols
                  )

# Set consistent NA values
nd = nd.fillna(pd.NA)
nd = nd.replace(["NaN", "       NaN"], pd.NA)

# Create dataframe using only "neighbourhoods"/"buurten" (not "districts"/"wijken") and save
neighbourhoods = nd[nd['recs'] == 'Buurt'].copy()
neighbourhoods = neighbourhoods.drop(columns=["recs"])
neighbourhoods.to_csv("data_clean/fact_neighbourhoods.csv", index=False, decimal=".")