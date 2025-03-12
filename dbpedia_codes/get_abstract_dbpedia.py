from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions
import pandas as pd
import time
import random
import requests

def get_movie_title_from_wikidata(tdmb_id):
    """Query Wikidata to get the movie title using IMDb ID."""
    if not tdmb_id or pd.isna(tdmb_id):
        return None
    
    print(f"Querying Wikidata for TMDB ID: {tdmb_id}")
    query = f'''
    SELECT ?filmLabel WHERE {{
      ?film wdt:P4947 "{tdmb_id}".
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    '''
    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "MyCustomBot/1.0 (https://example.com; contact@example.com)"}
    params = {"query": query, "format": "json"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        results = response.json()
        return results["results"]["bindings"][0]["filmLabel"]["value"] if results["results"]["bindings"] else None
    except requests.exceptions.RequestException as e:
        print(f"Error querying Wikidata: {e}")


def get_abstract_and_director_from_dbpedia(title, year):
    
    """Query DBpedia to get the movie abstract using the title."""
    if not title or pd.isna(title):
        return None
    
    print(f"Querying DBpedia for title: {title}")
    query = f'''
    SELECT DISTINCT (COALESCE(?abstractDBO, ?abstractRDFS) AS ?abstract) ?directorLabel
    WHERE {{
    {{
    ?movie rdf:type dbo:Film .
    ?movie dbp:title|rdfs:label|dbp:name "{title}"@en .
    ?movie dbo:director ?director .
    ?director rdfs:label ?directorLabel .
  }}
  UNION
  {{
    ?movie rdf:type owl:Thing .
    ?movie dbp:title|rdfs:label|dbp:name "{title}"@en .
    ?movie dbp:director ?director .
    ?director rdfs:label ?directorLabel .
    FILTER NOT EXISTS {{ ?movie rdf:type dbo:Film . }}
  }}
   
  OPTIONAL {{ ?movie dbo:abstract ?abstractDBO . FILTER(LANG(?abstractDBO) = "en") }}
  FILTER(CONTAINS(?abstractDBO, "{year}"))
  OPTIONAL {{ ?movie rdfs:comment ?abstractRDFS . FILTER(LANG(?abstractRDFS) = "en") }}
  FILTER(CONTAINS(?abstractRDFS, "{year}"))
  FILTER(LANG(?directorLabel)="it")
}}
'''
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setTimeout(30000)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        if results["results"]["bindings"]:
            abstract = results["results"]["bindings"][0]["abstract"]["value"]
            director = results["results"]["bindings"][0]["directorLabel"]["value"]
            return abstract, director
        else:
            return None
    except Exception as e:
        print(f"Error querying DBpedia: {e}")
        return None

def process_movies(movies_file, links_file, output_file):
    """Merge movies and links, query Wikidata & DBpedia, then save to CSV."""
    # Carica i dati e assicurati che imdbId sia trattato come stringa
    """movies_df = pd.read_csv(movies_file)[5264:6000]    
    links_df = pd.read_csv(links_file)
    links_df['tmdbId'] = links_df['tmdbId'].apply(lambda x: str(x).split('.0')[0])
    
    # Merge on movieId
    merged_df = movies_df.merge(links_df, on="movieId", how="left")"""
    merged_df = pd.read_csv(movies_file)
    merged_df['tmdbId'] = merged_df['tmdbId'].apply(lambda x: str(x).split('.0')[0])
    
    # Add columns for title from Wikidata and abstract & director from DBpedia
    for index, row in merged_df.iterrows():
        print(f"Processing row {index}")
        title = merged_df.at[index, "title"]
        year = title.split('(')[-1].split(')')[0]  # Extract year from title
        cleaned_title = title.rsplit('(', 1)[0].strip()# Remove year from title
        #print(f"Title: {cleaned_title}, Year: {year}")
          
        r1 = get_abstract_and_director_from_dbpedia(cleaned_title, year)
        if r1 is None:
            print(f"DBpedia query failed for {cleaned_title}. Trying Wikidata.")
            #print(f"Processing row {index}: movieId={row['movieId']}, tmdbId={row['tmdbId']}")
            #merged_df.at[index, "wikidata_title"] = get_movie_title_from_wikidata(str(row['tmdbId']))
            tmdb_title = get_movie_title_from_wikidata(str(row['tmdbId']))
            time.sleep(2 + random.uniform(0, 2))  # Random delay to prevent rate limiting
            
            if pd.notna(tmdb_title):
                result = get_abstract_and_director_from_dbpedia(tmdb_title, year)

                if result is None:  # Controllo se la funzione ha restituito None
                    abstract, director = "Abstract non disponibile", "Regista non disponibile"
                    #abstract = "Abstract non disponibile"
                else:
                    abstract, director = result  # Decomposizione sicura
                merged_df.at[index, "dbpedia_abstract"] = abstract
                merged_df.at[index, "dbpedia_director"] = director
                
        else:
            abstract, director = r1
            merged_df.at[index, "dbpedia_abstract"] = abstract
            merged_df.at[index, "dbpedia_director"] = director

        time.sleep(1 + random.uniform(0, 2))
        # Salva ogni X righe per non perdere progressi
        if index % 5 == 0:  # Salva ogni 10 righe
            merged_df.to_csv(output_file, index=False)
            print(f"Backup salvato: {output_file} (fino a riga {index})")
    
    # Save the result
    merged_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

# Example usage
process_movies("output.csv", "dataset/links.csv", "movies_with_abstracts.csv")