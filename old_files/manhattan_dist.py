import numpy as np
import pandas as pd

# Example format "raw" data
car = pd.DataFrame({
    "price": [35000],
    "luxury": [0],
    "sports": [0],
    "family": [1]
    })


car = pd.DataFrame({
    "price": [35000],
    "luxury": [0],
    "sports": [0],
    "family": [1]
    })

regions = pd.DataFrame({
    "pc4": ["3824", "5212", "1012"],
    "income": [45000, 55000, 50000],
    "luxury": [1, 0, 0],
    "sports": [0, 1, 0],
    "family": [0, 0, 1]
    })

# Lists of colnames to include for z-standardization
region_cols = ["income"]
car_cols = ["price"]

# Nested list of associated colnames (by index number of sublist) to compute L1 with
relation = np.array([
    ["price", "luxury", "sports", "family"],
    ["income", "luxury", "sports", "family"]
    ])

# Dictorionary to provide mean and stdev with, required for standardization of single-object dataframes
car_mean_std = pd.DataFrame({
    "price": {
        "mean": 35000, 
        "std": 5000}
    })

# ------- Load dataframe from csv -------
def load_df(csv_path, usecols=None):
    df = pd.read_csv(csv_path, usecols=usecols, delimiter = ",", decimal=".")
    return df


# ------- Z-standardize values -------
def standardize(df, list_cols, dict_mean_std=None):
        # df                Dataframe to standardize
        # list_cols         List of columns to standardize
        # dict_mean_std     Optional: manual dictionary input of mean/std => {"colname" : {"mean": 100, "std": 10}}

    output = df.copy()
    
    for col in list_cols:
        if dict_mean_std is None:
            mean = df[col].mean()
            std = df[col].std()
        else:
            mean = dict_mean_std[col]["mean"]
            std = dict_mean_std[col]["std"]
        
        output[col] = (df[col] - mean) / std
    return output



# ------- Compute Manhattan Distance (L1) -------
def manhattan_distance(df_a, df_b, PK_b, relation_a_b = None, n_return=None):
    # df_a                  df: single object with multiple z-standardized values (e.g. car model)
    # df_b                  df: multiple objects with multiple z-standardized values (e.g. regions)
    # PK_b                  Primary key of df_b to ID outcomes (e.g. "pc4")
    # relation_a_b          Nested list of columns for both df, (first a then b) aligned for L1 distance computation
    #                           e.g.: list = ["price, "seats"], ["income", "family_size"]

    # Check if relation_a_b is provided, if not use all columns of both df (in order)
    try:
        if relation_a_b == None:
            relation_a_b = np.array([df_a.columns.tolist(), df_b.columns.tolist()])
            df_b = df_b.fillna(0)
            print('"relation_a_b" not provided, using all columns in both df')
    except ValueError:
        df_b = df_b.fillna(0)
        pass

    l1 = []

    # Iterate through columns of df_b (multiple objects)
    for index, col in enumerate(df_b[relation_a_b[1]]):
        temp_list = []

        # Get values for associated column in df_a
        a = df_a[relation_a_b[0][index]]
        
        # Iterate through values in df_b > Subtract a/b values > Store in temporary list (L1's for single column)
        for b in df_b[col]:
            a, b = float(a), float(b)
            temp_list.append(abs(a-b))
        
        # Create a nested list > Sublists with L1 for mactching columns
        l1.append(temp_list)

    # Sum up L1 per matching columns > Compute min/max of summed L1
    l1 = np.array([sum(col) for col in zip(*l1)])
    min = np.min(l1)
    max = np.max(l1)
    
    # Convert summed L1 to dictionary > Add PK_b as Key
    l1 = dict(zip(df_b[PK_b], l1))
    l1 = dict(sorted(l1.items(), key=lambda item: item[1]))

    # Scale L1-values > Range 0 to 1
    l1_minmax = {}
    for key, value in l1.items():
        minmax_scaled = 1 - (float((value - min) / (max - min)))
        l1_minmax[key] = minmax_scaled
    
    if n_return != None:
        from itertools import islice
        l1_minmax = dict(islice(l1_minmax.items(), n_return))

    return l1_minmax
    


# ------- Call Functions -------
#regions_stnd =  standardize(regions, region_cols)
#car_stnd =      standardize(car, car_cols, dict_mean_std = car_mean_std)
#l1 =            manhattan_distance(car_stnd, regions_stnd, "pc4", relation_a_b = relation)


# ------- Example: Find most suitable car for wealthy, urban, single seniors -------
regions_2 = load_df("data_clean/fact_pc4.csv", 
                    usecols=["postcode", 
                             "gemiddelde_woz_waarde_woning", 
                             "stedelijkheid", 
                             "gemiddelde_huishoudensgrootte", 
                             "aantal_inwoners_65_jaar_en_ouder",
                             "aantal_eenpersoonshuishoudens"
                             ])
regions_2 = regions_2.dropna()
regions_2 = regions_2[(regions_2 != 0).all(axis=1)]




region_cols_2 = ["gemiddelde_woz_waarde_woning", 
                 "stedelijkheid", 
                 "gemiddelde_huishoudensgrootte", 
                 "aantal_inwoners_65_jaar_en_ouder", 
                 "aantal_eenpersoonshuishoudens"]

car_2 = pd.DataFrame({
    "price": [90000],
    "urban_score": [70],
    "seats": [3],
    "senior_score": [80],
    "singles_score": [60]
    })

car_cols_2 = ["price", "urban_score", "seats", "senior_score", "singles_score"]

car_mean_std_2 = pd.DataFrame({
    "price": {
        "mean": 35000, 
        "std": 20000},
    "urban_score": {
        "mean": 2.5, 
        "std": 1},
    "seats": {
        "mean": 5, 
        "std": 1},
    "senior_score": {
        "mean": 50, 
        "std": 16},
    "singles_score": {
        "mean": 50, 
        "std": 16}
        })

relation_2 = np.array([
    ["price", "urban_score", "seats", "senior_score", "singles_score"],
    ["gemiddelde_woz_waarde_woning", "stedelijkheid", "gemiddelde_huishoudensgrootte", "aantal_inwoners_65_jaar_en_ouder", "aantal_eenpersoonshuishoudens"]
    ])


regions_stnd_2 =    standardize(regions_2, region_cols_2)
car_stnd_2 =        standardize(car_2, car_cols_2, dict_mean_std = car_mean_std_2)
l1 =                manhattan_distance(car_stnd_2, regions_stnd_2, "postcode", relation_a_b = relation_2, n_return=10)

for i, (key, value) in enumerate(l1.items()):
    print(f"{key} {value}")
