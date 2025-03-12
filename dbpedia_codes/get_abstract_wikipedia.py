import csv
import requests
from bs4 import BeautifulSoup
import time
import re
import os
import shutil

def get_wikipedia_data(movie_title, year=None):
    """
    Cerca un film su Wikipedia e ne estrae il primo paragrafo come abstract e il regista.
    
    Args:
        movie_title (str): Titolo del film
        year (str, optional): Anno di uscita del film per migliorare la ricerca
    
    Returns:
        tuple: (abstract, director) - informazioni estratte o valori originali se non trovati
    """
    search_query = movie_title
    year_value = None
    
    if year:
        # Estrai l'anno dal formato (YYYY)
        year_match = re.search(r'\((\d{4})\)', year)
        if year_match:
            year_value = year_match.group(1)
            search_query = f"{movie_title} {year_value} film"
    else:
        # Prova a estrarre l'anno dal titolo se non è stato fornito separatamente
        year_match = re.search(r'\((\d{4})\)', movie_title)
        if year_match:
            year_value = year_match.group(1)
    
    # Modifica il titolo per la ricerca su Wikipedia
    search_query = search_query.replace(' ', '+')
    
    # URL di ricerca di Wikipedia
    search_url = f"https://en.wikipedia.org/w/index.php?search={search_query}"
    
    try:
        # Aggiungi un ritardo per evitare di sovraccaricare Wikipedia
        time.sleep(1)
        
        # Effettua la richiesta HTTP
        response = requests.get(search_url)
        response.raise_for_status()
        
        # Analizza la pagina HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Se siamo in una pagina di ricerca, prendi il primo risultato
        if "Search results" in soup.title.text:
            first_result = soup.select_one(".mw-search-result-heading a")
            if not first_result:
                return "Abstract non disponibile", "Regista non disponibile"
                
            article_url = f"https://en.wikipedia.org{first_result['href']}"
            
            # Effettua una nuova richiesta alla pagina dell'articolo
            time.sleep(1)
            response = requests.get(article_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
        
        # Estrai il primo paragrafo (abstract)
        first_paragraph = soup.select_one(".mw-parser-output p:not(.mw-empty-elt)")
        abstract = first_paragraph.text.strip() if first_paragraph else "Abstract non disponibile"
        
        # CONTROLLO: verifica se l'abstract contiene l'anno del film
        is_valid_abstract = False
        
        if abstract != "Abstract non disponibile":
            # Verifica se contiene l'anno del film (se disponibile)
            if year_value and year_value in abstract:
                is_valid_abstract = True
            # Puoi anche mantenere la verifica della parola "film" come backup
            # elif "film" in abstract.lower():
            #     is_valid_abstract = True
            
            if not is_valid_abstract:
                abstract = "Abstract non disponibile"
        
        # METODO MIGLIORATO PER TROVARE IL REGISTA
        director = "Regista non disponibile"
        
        # Metodo 1: Cerca nella infobox
        infobox = soup.select_one(".infobox")
        if infobox:
            rows = infobox.select("tr")
            for row in rows:
                header = row.select_one("th")
                if header and ("Director" in header.text or "Directed by" in header.text):
                    director_cell = row.select_one("td")
                    if director_cell:
                        # Estrai il testo con tutti i link
                        links = director_cell.select("a")
                        if links:
                            directors = []
                            for link in links:
                                # Verifica che non sia un link a un riferimento o una nota
                                if not link.get('href', '').startswith('#cite') and not link.get('class') and link.text.strip():
                                    directors.append(link.text.strip())
                            if directors:
                                director = ", ".join(directors)
                        else:
                            director = director_cell.text.strip()
                        
                        # Pulisci il testo del regista
                        director = re.sub(r'\[\d+\]', '', director)  # Rimuovi i riferimenti [1], [2], ecc.
                        director = re.sub(r'\s+', ' ', director).strip()  # Rimuovi spazi extra
                        break
        
        # Metodo 2: Cerca nella prima sezione del testo
        if director == "Regista non disponibile":
            # Cerca frasi come "directed by" nel primo paragrafo
            paragraphs = soup.select(".mw-parser-output p")
            for paragraph in paragraphs[:3]:  # Controlla solo i primi 3 paragrafi
                text = paragraph.text.lower()
                if "directed by" in text or "film by" in text or "directed the film" in text:
                    # Trova il regista dopo "directed by" o frasi simili
                    directed_by_match = re.search(r'directed by\s+([^\.]+)', text, re.IGNORECASE)
                    if directed_by_match:
                        director = directed_by_match.group(1).strip().title()
                    else:
                        # Cerca link di persone nel paragrafo
                        links = paragraph.select("a")
                        for link in links:
                            link_text = link.text.strip()
                            if link_text and len(link_text.split()) >= 2:  # Potrebbe essere un nome
                                href = link.get('href', '')
                                if '/wiki/' in href and not href.endswith('film'):
                                    director = link_text
                                    break
                    break
        
        # Metodo 3: Cerca in tutta la pagina per "Director"
        if director == "Regista non disponibile":
            # Cerca nelle tabelle e nelle sezioni cast/crew
            tables = soup.select("table")
            for table in tables:
                rows = table.select("tr")
                for row in rows:
                    cells = row.select("td, th")
                    if len(cells) >= 2:
                        if "director" in cells[0].text.lower():
                            director = cells[1].text.strip()
                            director = re.sub(r'\[\d+\]', '', director)
                            break
        
        return abstract, director
        
    except Exception as e:
        print(f"Errore durante l'elaborazione di {movie_title}: {e}")
        return "Abstract non disponibile", "Regista non disponibile"

def update_csv_file(input_file, output_file):
    """
    Aggiorna il file CSV sostituendo gli abstract e i registi non disponibili
    con informazioni recuperate da Wikipedia.
    
    Args:
        input_file (str): Percorso del file CSV di input
        output_file (str): Percorso del file CSV di output
    """
    updated_count = 0
    total_count = 0
    films_processed = 0
    temp_file = output_file + ".temp"
    
    try:
        # Se esiste già un file temporaneo, potrebbe essere un recupero da un'esecuzione precedente
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            print(f"Trovato file temporaneo. Riprendo da dove ero rimasto.")
            shutil.copy(temp_file, output_file)
        
        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            
            # Leggi il file esistente se presente, altrimenti crea un nuovo file
            existing_data = []
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                with open(output_file, 'r', encoding='utf-8') as existing_file:
                    existing_reader = csv.DictReader(existing_file)
                    existing_data = list(existing_reader)
                    
                    # Verifica che i file abbiano la stessa struttura
                    if existing_reader.fieldnames != fieldnames:
                        raise ValueError("Il file di output esistente ha una struttura diversa dal file di input")
                    
                    # Determina quanti film sono già stati elaborati
                    total_count = len(existing_data)
                    films_processed = total_count
                    print(f"Ripresa elaborazione dal film {films_processed + 1}")
            
            # Se non ci sono dati esistenti, leggi da input
            if not existing_data:
                with open(input_file, 'r', encoding='utf-8') as raw_input:
                    existing_reader = csv.DictReader(raw_input)
                    existing_data = list(existing_reader)
            
            # Apri il file di output per l'aggiornamento
            with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Scrivi i dati già elaborati
                for i, row in enumerate(existing_data):
                    if i < films_processed:
                        writer.writerow(row)
                        continue
                    
                    total_count += 1
                    
                    # Verifica se è necessario aggiornare questa riga
                    if (row['dbpedia_abstract'] == "Abstract non disponibile" or 
                        row['dbpedia_director'] == "Regista non disponibile"):
                        
                        print(f"Elaborazione di: {row['title']}")
                        
                        # Ottieni i dati da Wikipedia
                        abstract, director = get_wikipedia_data(row['title'], row.get('year', None))
                        
                        # Aggiorna i campi se abbiamo trovato nuove informazioni
                        if row['dbpedia_abstract'] == "Abstract non disponibile" and abstract != "Abstract non disponibile":
                            row['dbpedia_abstract'] = abstract
                            print(f"  Abstract aggiornato: {abstract[:50]}...")
                            updated_count += 1
                        
                        if row['dbpedia_director'] == "Regista non disponibile" and director != "Regista non disponibile":
                            row['dbpedia_director'] = director
                            print(f"  Regista trovato: {director}")
                            updated_count += 1
                    
                    writer.writerow(row)
                    films_processed += 1
                    
                    # Salvataggio parziale ogni 5 film elaborati
                    if films_processed % 5 == 0:
                        print(f"✅ Salvataggio parziale dopo {films_processed} film")
                        # Crea una copia di backup
                        outfile.flush()
                        shutil.copy(output_file, temp_file)
                
        # Rimuovi il file temporaneo alla fine
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        print(f"Aggiornamento completato! {updated_count} campi aggiornati su {total_count} righe.")
    
    except Exception as e:
        print(f"Errore durante l'aggiornamento del file CSV: {e}")
        print("Se esisteva un salvataggio parziale, puoi riprendere da lì.")


if __name__ == "__main__":
    # Richiedi all'utente i percorsi dei file
    input_file = 'movies_with_abstracts.csv'
    output_file = 'movies_with_abstract_wikipedia_3.csv'
    
    # Aggiorna il file CSV
    update_csv_file(input_file, output_file)