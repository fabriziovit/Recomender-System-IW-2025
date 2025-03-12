import requests
import csv
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict

def normalize_string(s):
    """
    Normalizza una stringa sostituendo vari caratteri speciali e rimuovendo punteggiatura superflua.
    """
    import unicodedata
    import re
    
    # Normalizzazione Unicode (decomposizione seguita da ricomposizione)
    s = unicodedata.normalize('NFKD', s)
    
    # Sostituzione di vari tipi di trattini con un trattino standard
    trattini = ['-', '–', '—', '−', '‐', '‑', '‒', '–', '—', '―']
    for trattino in trattini:
        s = s.replace(trattino, '-')
    
    # Sostituzione di vari tipi di apostrofi e virgolette
    apostrofi = ['\'', ''', ''', '`', '´', '"', '"', '"']
    for apostrofo in apostrofi:
        s = s.replace(apostrofo, "'")
    
    # Rimozione di spazi multipli
    s = re.sub(r'\s+', ' ', s)
    
    # Rimozione di spazi attorno ai trattini
    s = re.sub(r'\s*-\s*', '-', s)
    
    # Conversione a minuscolo (opzionale, attivare se necessario)
    # s = s.lower()
    
    # Rimozione di punteggiatura superflua (opzionale, attivare se necessario)
    # s = re.sub(r'[^\w\s-]', '', s)
    
    # Rimozione di spazi iniziali e finali
    s = s.strip()
    
    return s

def compare_strings(str1, str2, case_sensitive=True, ignore_punctuation=False):
    """
    Confronta due stringhe dopo normalizzazione.
    
    Args:
        str1, str2: Le stringhe da confrontare
        case_sensitive: Se False, ignora differenze tra maiuscole e minuscole
        ignore_punctuation: Se True, rimuove tutta la punteggiatura prima del confronto
    
    Returns:
        bool: True se le stringhe sono considerate uguali dopo la normalizzazione
    """
    import re
    
    # Normalizzazione di base
    norm1 = normalize_string(str1)
    norm2 = normalize_string(str2)
    
    # Opzioni aggiuntive
    if not case_sensitive:
        norm1 = norm1.lower()
        norm2 = norm2.lower()
    
    if ignore_punctuation:
        norm1 = re.sub(r'[^\w\s]', '', norm1)
        norm2 = re.sub(r'[^\w\s]', '', norm2)
    
    # Confronto finale
    return norm1 == norm2

def get_director_by_movie_id(csv_file, movie_id):
    """
    Ottiene il regista di un film dal file CSV dato il movieId.
    
    Args:
        csv_file (str): Percorso del file CSV
        movie_id (int): ID del film da cercare
        
    Returns:
        str: Nome del regista o None se non trovato
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if int(row['movieId']) == movie_id:
                    director = row.get('dbpedia_director') 
                    if director.endswith(')'):
                        director = director.rsplit('(', 1)[0].strip()
                    return director
        return None
    except Exception as e:
        print(f"Errore nella lettura del CSV: {e}")
        return None
    
def get_title_by_movie_id(csv_file, movie_id):
    """
    Ottiene il regista di un film dal file CSV dato il movieId.
    
    Args:
        csv_file (str): Percorso del file CSV
        movie_id (int): ID del film da cercare
        
    Returns:
        str: Nome del regista o None se non trovato
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if int(row['movieId']) == movie_id:
                    movie_title = row.get('title')
                    # Rimuove l'anno dal titolo del film se presente
                    if movie_title.endswith(')'):
                        movie_title = movie_title.rsplit('(', 1)[0].strip()
                    return movie_title
        return None
    except Exception as e:
        print(f"Errore nella lettura del CSV: {e}")
        return None
    
def get_release_year_from_wikidata(film_title, director_name):
    """
    Ottiene l'anno di uscita di un film da Wikidata, filtrando per titolo e regista.
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT DISTINCT ?anno WHERE {{
      ?film wdt:P31 wd:Q11424 ;        # Il film è un'istanza di "film"
            rdfs:label "{film_title}"@en ;  # Titolo del film
            wdt:P577 ?releaseDate ;      # Data di pubblicazione
            wdt:P57 ?director .          # Regista
      ?director rdfs:label "{director_name}"@en .  # Filtra per il regista
      BIND(YEAR(?releaseDate) AS ?anno)
    }}
    LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            return result["anno"]["value"]
    except Exception as e:
        print(f"Errore durante la query Wikidata: {e}")
    return "N/A"

def get_directed_films_with_actors(director_name):
    """
    Ottiene i film diretti da un regista con attori e prova a recuperare l'anno di uscita da Wikidata.
    """
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbp: <http://dbpedia.org/property/>

    SELECT DISTINCT ?film ?titolo ?actorName WHERE {{
      ?director rdfs:label "{director_name}"@en .
      ?film dbo:director ?director ;
            rdfs:label ?titolo .
      FILTER(LANG(?titolo) = "en")
      OPTIONAL {{ ?film dbo:starring ?actor . ?actor rdfs:label ?actorName . FILTER(LANG(?actorName) = "en") }}
    }}
    ORDER BY ?titolo
    LIMIT 100
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        films_dict = defaultdict(lambda: {"titolo": "", "anno": "N/A", "attori": set()})
        
        for result in results["results"]["bindings"]:
            film_uri = result["film"]["value"]
            film_title = result["titolo"]["value"]
            films_dict[film_uri]["titolo"] = film_title
            
            if "actorName" in result:
                films_dict[film_uri]["attori"].add(result["actorName"]["value"])
        
        films_list = []
        for uri, info in films_dict.items():
            title=info["titolo"]
            if title.endswith(')'):
                title_striped = title.rsplit('(', 1)[0].strip()
            else:
                title_striped = title
            anno = get_release_year_from_wikidata(title_striped, director_name)  # Recupera l'anno da Wikidata
            films_list.append({
                "uri": uri,
                "titolo": info["titolo"],
                "anno": anno,
                "attori": sorted(list(info["attori"]))
            })
        
        # Ordina per anno (dal più recente) gestendo "N/A" o None come 0
        films_list.sort(key=lambda x: int(x["anno"]) if x["anno"] not in [None, "N/A"] else 0, reverse=True)
        return films_list
    
    except Exception as e:
        print(f"Errore durante la query DBpedia: {e}")
        return []

def recommend_films_with_actors(director_name, max_actors=5, title=None, movie_title_selected = False):
    """
    Stampa una lista formattata di film raccomandati con attori.
    
    Args:
        director_name (str): Il nome del regista
        max_actors (int): Numero massimo di attori da mostrare per film
    """
    print(f"\nFilm diretti da {director_name} con gli attori principali (max {max_actors}):\n")
    print("-" * 80)
    
    films = get_directed_films_with_actors(director_name)
    
    if not films:
        print(f"Nessun film trovato per {director_name} o si è verificato un errore.")
        return
    
    max_film = 6
    index = 1
    title = normalize_string(title)
    for i, film in enumerate(films, 1):
        if(i < max_film):
            if(movie_title_selected):
                title_query = normalize_string(film["titolo"])
                if(compare_strings(title, title_query)):
                    max_film+=1
                    continue
            anno = film["anno"]
            print(f"{index}. {film['titolo']} ({anno})")
            index += 1
            
            # Stampiamo gli attori se disponibili
            if film["attori"]:
                print("   Attori principali:")
                # Limitiamo a massimo 5 attori per film come richiesto
                for actor in film["attori"][:max_actors]:
                    print(f"   - {actor}")
                
                # Indichiamo se ci sono più attori
                if len(film["attori"]) > max_actors:
                    print(f"   ... e altri {len(film['attori']) - max_actors} attori")
            else:
                print("   Attori: Informazione non disponibile")
            
            print()  # Linea vuota tra i film
        else:
            break
        
    print("-" * 80)
    print(f"Totale film trovati: {len(films)}")

def recommend_by_movie_id(csv_file, movie_id, max_actors=5, movie_title_selected = False):
    """
    Dato un movieId, trova il regista nel CSV e mostra raccomandazioni di altri suoi film.
    
    Args:
        csv_file (str): Percorso del file CSV
        movie_id (int): ID del film di cui cercare il regista
        max_actors (int): Numero massimo di attori da mostrare per film
    """
    director = get_director_by_movie_id(csv_file, movie_id)
    title = get_title_by_movie_id(csv_file, movie_id)
    
    if not director:
        print(f"Nessun regista trovato per il film con ID {movie_id}")
        return
    
    print(f"Trovato regista: {director} per il film con ID {movie_id}")
    recommend_films_with_actors(director, max_actors, title, movie_title_selected)
