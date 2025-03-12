import pandas as pd

def filter_unavailable_abstracts(output_file):
    # Legge il CSV in un DataFrame
    df = pd.read_csv("Datasets_dbpedia/movies_with_abstracts_complete.csv", dtype=str, quotechar='"', on_bad_lines='skip')
    
    # Filtra le righe dove dbpedia_abstract è "Abstract non disponibile"
    filtered_df = df[(df['dbpedia_abstract'] == "Abstract non disponibile") | (df['dbpedia_abstract'].isnull())]
    
    # Salva il risultato in un nuovo CSV
    filtered_df.to_csv(output_file, index=False)
    
    print(f"File salvato: {output_file}")

# Esempio di utilizzo
output_csv = "output.csv"
filter_unavailable_abstracts(output_csv)
