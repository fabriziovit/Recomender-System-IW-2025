import pandas as pd

# Load CSV files specifying data types
df_A = pd.read_csv("movies_with_abstracts_complete.csv")
df_B = pd.read_csv("dataset/links.csv", dtype={"imdbId": str, "tmdbId": str})

# Ensure that dataframe A also has correct types
if "imdbId" in df_A.columns:
    df_A["imdbId"] = df_A["imdbId"].astype(str)
if "tmdbId" in df_A.columns:
    df_A["tmdbId"] = df_A["tmdbId"].astype(str)

# Proceed with update as before
df_A.set_index("movieId", inplace=True)
df_B.set_index("movieId", inplace=True)

try:
    df_A.update(df_B)
except ValueError:
    common_columns = df_A.columns.intersection(df_B.columns)
    df_A.update(df_B[common_columns])
    print("Warning: some columns were not updated because they have different types")

# Save result maintaining correct format
df_A.reset_index(inplace=True)
df_A.to_csv("movies_with_abstracts_updated.csv", index=False)
