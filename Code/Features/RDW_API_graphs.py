import requests
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# -------- Inputs --------
merk = "BMW"
handelsbenaming = "118i"
keys = ["catalogusprijs", "datum_eerste_tenaamstelling_in_nederland", "datum_tenaamstelling"]

# -------- API call --------
url = f"https://opendata.rdw.nl/resource/m9d7-ebf2.json?merk={merk}&handelsbenaming={handelsbenaming}"
response = requests.get(url)
data = response.json()

# -------- Convert to DataFrame --------
df = pd.DataFrame(data)

# Fill missing keys with None
for k in keys:
    if k not in df.columns:
        df[k] = None

# -------- Convert types --------
df['catalogusprijs'] = pd.to_numeric(df['catalogusprijs'], errors='coerce')  # coerce invalid to NaN
df['datum_eerste_tenaamstelling_in_nederland'] = pd.to_datetime(df['datum_eerste_tenaamstelling_in_nederland'], errors='coerce', format='%Y%m%d')
df['datum_tenaamstelling'] = pd.to_datetime(df['datum_tenaamstelling'], errors='coerce', format='%Y%m%d')

# Drop rows with missing catalogusprijs or datum_tenaamstelling
df = df.dropna(subset=['catalogusprijs', 'datum_tenaamstelling'])

# Optional: filter out extremely large/unrealistic catalogusprijs
df = df[df['catalogusprijs'] < 1e7]

# -------- Create Occassion Dummy --------
df['Occasion'] = np.where(
    df['datum_eerste_tenaamstelling_in_nederland'] != df['datum_tenaamstelling'], 1, 0
)

# -------- Extract year --------
df['year'] = df['datum_tenaamstelling'].dt.year
df = df.dropna(subset=['year'])

# -------- Calculate averages --------
avg_price = df.groupby('Occasion')['catalogusprijs'].mean()
avg_new = avg_price.get(0, np.nan)
avg_occ = avg_price.get(1, np.nan)
print(f"Average catalogusprijs New: {avg_new:.2f}")
print(f"Average catalogusprijs Occasion: {avg_occ:.2f}")

# -------- Count records per year & calculate market cap --------
counts = df.groupby(['year', 'Occasion']).agg(
    count=('catalogusprijs','size'),
    avg_price=('catalogusprijs','mean')
).reset_index()

counts['market_cap'] = counts['count'] * counts['avg_price']

# -------- Forecast next 5 years --------
last_year = counts['year'].max()
forecast_years = np.arange(last_year + 1, last_year + 6)

# Linear regression separately for New and Occasion
forecast_dfs = []
for occ in [0,1]:
    sub = counts[counts['Occasion']==occ]
    if len(sub) > 1:  # need at least 2 points for regression
        model = LinearRegression()
        model.fit(sub[['year']], sub['market_cap'])
        pred = model.predict(forecast_years.reshape(-1,1))
        df_forecast = pd.DataFrame({
            'year': forecast_years,
            'Occasion': occ,
            'market_cap': pred,
            'count': np.nan
        })
        forecast_dfs.append(df_forecast)

forecast_df = pd.concat(forecast_dfs, ignore_index=True)
plot_df = pd.concat([counts, forecast_df], ignore_index=True)

# Map labels
plot_df['Type'] = plot_df['Occasion'].map({0:'New', 1:'Occasion'})

# -------- Interactive Plot --------
fig = px.line(
    plot_df, x='year', y='market_cap', color='Type',
    line_dash='Type', markers=True,
    title=f"{merk} {handelsbenaming} Market Cap Over Time",
    hover_data={'year':True, 'count':True, 'market_cap':':.2f'}
)

# Dotted line for forecast
for occ in [0,1]:
    fig.add_scatter(
        x=forecast_df[forecast_df['Occasion']==occ]['year'],
        y=forecast_df[forecast_df['Occasion']==occ]['market_cap'],
        mode='lines', name=f"Forecast {['New','Occasion'][occ]}",
        line=dict(dash='dot')
    )

fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Market Cap (€)',
    hovermode='x unified'
)

fig.show()
