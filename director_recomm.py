import csv
import logging
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def normalize_string(s):
    """Normalize a string by replacing various special characters and removing superfluous punctuation."""
    import unicodedata
    import re

    # Unicode normalization (decomposition followed by recomposition)
    s = unicodedata.normalize("NFKD", s)

    # Replace various types of hyphens with a standard hyphen
    trattini = ["-", "–", "—", "−", "‐", "‑", "‒", "–", "—", "―"]
    for trattino in trattini:
        s = s.replace(trattino, "-")

    # Replace various types of apostrophes and quotes
    apostrofi = ["'", """, """, "`", "´", '"', '"', '"']
    for apostrofo in apostrofi:
        s = s.replace(apostrofo, "'")

    # Remove multiple spaces
    s = re.sub(r"\s+", " ", s)

    # Remove spaces around hyphens
    s = re.sub(r"\s*-\s*", "-", s)

    # Convert to lowercase (optional, enable if necessary)
    # s = s.lower()

    # Remove superfluous punctuation (optional, enable if necessary)
    # s = re.sub(r'[^\w\s-]', '', s)

    # Remove leading and trailing spaces
    s = s.strip()

    return s


def compare_strings(str1, str2, case_sensitive=True, ignore_punctuation=False):
    """Compare two strings after normalization.
    Args:
        str1, str2: The strings to compare
        case_sensitive: If False, ignore differences between uppercase and lowercase
        ignore_punctuation: If True, remove all punctuation before comparison
    Returns:
        bool: True if strings are considered equal after normalization"""
    import re

    # Basic normalization
    norm1 = normalize_string(str1)
    norm2 = normalize_string(str2)

    # Additional options
    if not case_sensitive:
        norm1 = norm1.lower()
        norm2 = norm2.lower()

    if ignore_punctuation:
        norm1 = re.sub(r"[^\w\s]", "", norm1)
        norm2 = re.sub(r"[^\w\s]", "", norm2)

    # Final comparison
    return norm1 == norm2


def get_director_by_movie_id(csv_file, movie_id):
    """Get the director of a movie from the CSV file given the movieId.
    Args:
        csv_file (str): Path to the CSV file
        movie_id (int): ID of the movie to search for
    Returns:
        str: Director's name or None if not found"""
    try:
        with open(csv_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if int(row["movieId"]) == movie_id:
                    director = row.get("dbpedia_director")
                    if director.endswith(")"):
                        director = director.rsplit("(", 1)[0].strip()
                    return director
        return None
    except Exception as e:
        logging.info(f"Error reading CSV: {e}")
        return None


def get_title_by_movie_id(csv_file, movie_id):
    """Get the director of a movie from the CSV file given the movieId.
    Args:
        csv_file (str): Path to the CSV file
        movie_id (int): ID of the movie to search for
    Returns:
        str: Director's name or None if not found"""
    try:
        with open(csv_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if int(row["movieId"]) == movie_id:
                    movie_title = row.get("title")
                    # Remove the year from the movie title if present
                    if movie_title.endswith(")"):
                        movie_title = movie_title.rsplit("(", 1)[0].strip()
                    return movie_title
        return None
    except Exception as e:
        logging.info(f"Error reading CSV: {e}")
        return None


def get_release_year_from_wikidata(film_title, director_name):
    """Get the release year of a movie from Wikidata, filtering by title and director."""
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    sparql.addCustomHttpHeader("User-Agent", "MyRecommederSystem/1.0 (your@email.com)")

    query = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT DISTINCT ?anno WHERE {{
      ?film wdt:P31 wd:Q11424 ;        # The film is an instance of "film"
            rdfs:label "{film_title}"@en ;  # Movie title
            wdt:P577 ?releaseDate ;      # Release date
            wdt:P57 ?director .          # Director
      ?director rdfs:label "{director_name}"@en .  # Filter by director
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
        logging.info(f"Error during Wikidata query: {e}")
    return "N/A"


def get_directed_films_with_actors(director_name):
    """Gets films directed by a director with actors and tries to retrieve the release year from Wikidata."""
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
            title = info["titolo"]
            if title.endswith(")"):
                title_striped = title.rsplit("(", 1)[0].strip()
            else:
                title_striped = title
            anno = get_release_year_from_wikidata(title_striped, director_name)  # Retrieve year from Wikidata
            films_list.append({"uri": uri, "titolo": info["titolo"], "anno": anno, "attori": sorted(list(info["attori"]))})

        # Sort by year (from most recent) handling "N/A" or None as 0
        films_list.sort(key=lambda x: int(x["anno"]) if x["anno"] not in [None, "N/A"] else 0, reverse=True)
        return films_list

    except Exception as e:
        logging.info(f"Error during DBpedia query: {e}")
        return []


def recommend_films_with_actors(director_name, max_actors=5, title=None, movie_title_selected=False):
    """Returns a data structure of recommended films with actors.
    Args:
        director_name (str): The director's name
        max_actors (int): Maximum number of actors to show per film
        title (str, optional): Movie title to exclude
        movie_title_selected (bool): If True, excludes the film with corresponding title
    Returns:
        dict: A dictionary containing the list of recommended films and director information"""
    films = get_directed_films_with_actors(director_name)

    if not films:
        return {"success": False, "message": f"No films found for {director_name} or an error occurred.", "director": director_name, "total_films": 0, "films": []}

    recommended_films = []
    max_film = 6
    index = 1

    if title:
        title = normalize_string(title)

    for i, film in enumerate(films, 1):
        if i < max_film:
            if movie_title_selected:
                title_query = normalize_string(film["titolo"])
                if compare_strings(title, title_query):
                    max_film += 1
                    continue

            film_data = {"id": index, "title": film["titolo"], "year": film["anno"], "actors": []}

            # Add actors if available
            if film["attori"]:
                # Limit to maximum max_actors actors per film
                actors_to_show = film["attori"][:max_actors]
                for actor in actors_to_show:
                    film_data["actors"].append(actor)

                # Add information about remaining actors
                film_data["additional_actors_count"] = max(0, len(film["attori"]) - max_actors)

            recommended_films.append(film_data)
            index += 1
        else:
            break

    result = {
        "success": True,
        "director": director_name,
        "max_actors_shown": max_actors,
        "total_films_found": len(films),
        "films_shown": len(recommended_films),
        "films": recommended_films,
    }

    return result


def recommend_by_movie_id(csv_file, movie_id, max_actors=5, movie_title_selected=False):
    """Given a movieId, finds the director in the CSV and shows recommendations of other films by the same director.
    Args:
        csv_file (str): Path to the CSV file
        movie_id (int): ID of the movie to find the director for
        max_actors (int): Maximum number of actors to show per film"""
    director = get_director_by_movie_id(csv_file, movie_id)
    title = get_title_by_movie_id(csv_file, movie_id)

    if not director:
        logging.info(f"No director found for movie with ID {movie_id}")
        return

    logging.info(f"Found director: {director} for movie with ID {movie_id}")
    return recommend_films_with_actors(director, max_actors, title, movie_title_selected)
