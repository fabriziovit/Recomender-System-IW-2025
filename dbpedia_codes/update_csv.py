import pandas as pd

# Carica i due file CSV
df_A = pd.read_csv("Datasets_dbpedia/movies_with_abstracts_complete.csv")
df_B = pd.read_csv("movies_with_abstract_wikipedia_3.csv")

# Imposta 'movieId' come indice per poter allineare le righe facilmente
df_A.set_index('movieId', inplace=True)
df_B.set_index('movieId', inplace=True)

# Aggiorna il dataframe A con le righe di B (solo per i movieId comuni)
try:
    df_A.update(df_B)
except ValueError:
    # Se ci sono colonne con tipi diversi, aggiorna solo le colonne comuni
    common_columns = df_A.columns.intersection(df_B.columns)
    df_A.update(df_B[common_columns])
    print("Attenzione: alcune colonne non sono state aggiornate perché hanno tipi diversi")

# Se vuoi salvare il risultato in un nuovo file CSV:
df_A.reset_index(inplace=True)
df_A.to_csv("movies_with_abstracts_updated.csv", index=False)