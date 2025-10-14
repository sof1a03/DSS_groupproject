import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import math
import os

# Ensure output folder exists
output_dir = "stat_reports"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv("data_final/REGIONAL.csv")

# Columns to plot
cols_mapping = {
    "df": [
        "std_avg_yearly_income_k", "std_body_hatchback", "std_body_mpv", "std_body_station",
        "std_p_car_weight_0_to_850", "std_p_car_weight_1151-1500", "std_p_car_weight_1501_more",
        "std_p_car_weight_851_to_1150", "std_p_diesel", "std_p_electric",
        "std_p_gasoline", "std_p_hybrid", "std_avg_household_size", "std_urbanization", 
        "std_avg_income_household", "std_p_inhb_15_to_25_year", "std_p_inhb_25_to_45_year",
        "std_p_inhb_45_to_65_year", "std_p_inhb_65_year_older"
    ]
}

# Map dataset names to DataFrames
df_mapping = {"df": df}

# A4 size in inches
a4_width, a4_height = 8.27, 11.69

# Function to export histograms to a PDF per dataset
def export_histograms(df, cols, df_name, output_dir):
    cols_in_df = [c for c in cols if c in df.columns]
    if not cols_in_df:
        print(f"No valid columns to plot for {df_name}")
        return

    output_pdf = os.path.join(output_dir, f"histograms_{df_name}.pdf")
    plots_per_page = 3 * 5
    n_pages = math.ceil(len(cols_in_df) / plots_per_page)

    with PdfPages(output_pdf) as pdf:
        for page in range(n_pages):
            fig, axes = plt.subplots(5, 3, figsize=(a4_width, a4_height))
            axes = axes.flatten()
            fig.suptitle(f"Dataset: {df_name}", fontsize=16, y=0.95)

            for i in range(plots_per_page):
                idx = page * plots_per_page + i
                if idx >= len(cols_in_df):
                    axes[i].axis("off")
                    continue

                col = cols_in_df[idx]
                data = df[col].dropna()
                axes[i].hist(data, bins=20, color='skyblue', edgecolor='black')
                axes[i].set_title(col, fontsize=9)
                axes[i].set_xlabel("")
                axes[i].set_ylabel("")

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved histograms for {df_name} to {output_pdf}")


# Export histograms for each dataset separately
export_histograms(df, cols_mapping["df"], "regional_hist", output_dir)
