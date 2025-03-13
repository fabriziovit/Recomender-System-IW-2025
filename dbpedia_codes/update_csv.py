import pandas as pd

# Carica i file CSV specificando i tipi di dati
df_A = pd.read_csv("movies_with_abstracts_complete.csv")
df_B = pd.read_csv("dataset/links.csv", dtype={'imdbId': str, 'tmdbId': str})

# Assicurati che anche nel dataframe A i tipi siano corretti
if 'imdbId' in df_A.columns:
    df_A['imdbId'] = df_A['imdbId'].astype(str)
if 'tmdbId' in df_A.columns:
    df_A['tmdbId'] = df_A['tmdbId'].astype(str)

# Procedi con l'aggiornamento come prima
df_A.set_index('movieId', inplace=True)
df_B.set_index('movieId', inplace=True)

try:
    df_A.update(df_B)
except ValueError:
    common_columns = df_A.columns.intersection(df_B.columns)
    df_A.update(df_B[common_columns])
    print("Attenzione: alcune colonne non sono state aggiornate perché hanno tipi diversi")

# Salva il risultato mantenendo il formato corretto
df_A.reset_index(inplace=True)
df_A.to_csv("movies_with_abstracts_updated.csv", index=False)