import csv
import requests
from bs4 import BeautifulSoup
import time
import re
import os
import shutil


def get_wikipedia_data(movie_title, year=None):
    """
    Searches for a movie on Wikipedia and extracts the first paragraph as abstract and director.

    Args:
        movie_title (str): Movie title
        year (str, optional): Movie release year to improve search

    Returns:
        tuple: (abstract, director) - extracted information or original values if not found
    """
    search_query = movie_title
    year_value = None

    if year:
        # Extract year from format (YYYY)
        year_match = re.search(r"\((\d{4})\)", year)
        if year_match:
            year_value = year_match.group(1)
            search_query = f"{movie_title} {year_value} film"
    else:
        # Try to extract year from title if not provided separately
        year_match = re.search(r"\((\d{4})\)", movie_title)
        if year_match:
            year_value = year_match.group(1)

    # Modify title for Wikipedia search
    search_query = search_query.replace(" ", "+")

    # Wikipedia search URL
    search_url = f"https://en.wikipedia.org/w/index.php?search={search_query}"

    try:
        # Add delay to avoid overloading Wikipedia
        time.sleep(1)

        # Make HTTP request
        response = requests.get(search_url)
        response.raise_for_status()

        # Parse HTML page
        soup = BeautifulSoup(response.text, "html.parser")

        # If we're on a search page, take the first result
        if "Search results" in soup.title.text:
            first_result = soup.select_one(".mw-search-result-heading a")
            if not first_result:
                return "Abstract not available", "Director not available"

            article_url = f"https://en.wikipedia.org{first_result['href']}"

            # Make new request to article page
            time.sleep(1)
            response = requests.get(article_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

        # Extract first paragraph (abstract)
        first_paragraph = soup.select_one(".mw-parser-output p:not(.mw-empty-elt)")
        abstract = first_paragraph.text.strip() if first_paragraph else "Abstract not available"

        # CHECK: verify if abstract contains movie year
        is_valid_abstract = False

        if abstract != "Abstract not available":
            # Check if it contains the movie year (if available)
            if year_value and year_value in abstract:
                is_valid_abstract = True
            # You can also keep the "film" word check as backup
            # elif "film" in abstract.lower():
            #     is_valid_abstract = True

            if not is_valid_abstract:
                abstract = "Abstract not available"

        # IMPROVED METHOD TO FIND DIRECTOR
        director = "Director not available"

        # Method 1: Search in infobox
        infobox = soup.select_one(".infobox")
        if infobox:
            rows = infobox.select("tr")
            for row in rows:
                header = row.select_one("th")
                if header and ("Director" in header.text or "Directed by" in header.text):
                    director_cell = row.select_one("td")
                    if director_cell:
                        # Extract text with all links
                        links = director_cell.select("a")
                        if links:
                            directors = []
                            for link in links:
                                # Check that it's not a link to a reference or note
                                if not link.get("href", "").startswith("#cite") and not link.get("class") and link.text.strip():
                                    directors.append(link.text.strip())
                            if directors:
                                director = ", ".join(directors)
                        else:
                            director = director_cell.text.strip()

                        # Clean director text
                        director = re.sub(r"\[\d+\]", "", director)  # Remove references [1], [2], etc.
                        director = re.sub(r"\s+", " ", director).strip()  # Remove extra spaces
                        break

        # Method 2: Search in first text section
        if director == "Director not available":
            # Search for phrases like "directed by" in first paragraph
            paragraphs = soup.select(".mw-parser-output p")
            for paragraph in paragraphs[:3]:  # Check only first 3 paragraphs
                text = paragraph.text.lower()
                if "directed by" in text or "film by" in text or "directed the film" in text:
                    # Find director after "directed by" or similar phrases
                    directed_by_match = re.search(r"directed by\s+([^\.]+)", text, re.IGNORECASE)
                    if directed_by_match:
                        director = directed_by_match.group(1).strip().title()
                    else:
                        # Search for person links in paragraph
                        links = paragraph.select("a")
                        for link in links:
                            link_text = link.text.strip()
                            if link_text and len(link_text.split()) >= 2:  # Could be a name
                                href = link.get("href", "")
                                if "/wiki/" in href and not href.endswith("film"):
                                    director = link_text
                                    break
                    break

        # Method 3: Search entire page for "Director"
        if director == "Director not available":
            # Search in tables and cast/crew sections
            tables = soup.select("table")
            for table in tables:
                rows = table.select("tr")
                for row in rows:
                    cells = row.select("td, th")
                    if len(cells) >= 2:
                        if "director" in cells[0].text.lower():
                            director = cells[1].text.strip()
                            director = re.sub(r"\[\d+\]", "", director)
                            break

        return abstract, director

    except Exception as e:
        print(f"Error during processing of {movie_title}: {e}")
        return "Abstract not available", "Director not available"


def update_csv_file(input_file, output_file):
    """
    Updates CSV file replacing unavailable abstracts and directors
    with information retrieved from Wikipedia.

    Args:
        input_file (str): Input CSV file path
        output_file (str): Output CSV file path
    """
    updated_count = 0
    total_count = 0
    films_processed = 0
    temp_file = output_file + ".temp"

    try:
        # If a temporary file already exists, it might be a recovery from a previous run
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            print(f"Found temporary file. Resuming from where I left off.")
            shutil.copy(temp_file, output_file)

        with open(input_file, "r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames

            # Read existing file if present, otherwise create new file
            existing_data = []
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                with open(output_file, "r", encoding="utf-8") as existing_file:
                    existing_reader = csv.DictReader(existing_file)
                    existing_data = list(existing_reader)

                    # Verify that files have the same structure
                    if existing_reader.fieldnames != fieldnames:
                        raise ValueError("Existing output file has different structure from input file")

                    # Determine how many films have already been processed
                    total_count = len(existing_data)
                    films_processed = total_count
                    print(f"Resuming processing from film {films_processed + 1}")

            # If no existing data, read from input
            if not existing_data:
                with open(input_file, "r", encoding="utf-8") as raw_input:
                    existing_reader = csv.DictReader(raw_input)
                    existing_data = list(existing_reader)

            # Open output file for update
            with open(output_file, "w", encoding="utf-8", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

                # Write already processed data
                for i, row in enumerate(existing_data):
                    if i < films_processed:
                        writer.writerow(row)
                        continue

                    total_count += 1

                    # Check if this row needs updating
                    if row["dbpedia_abstract"] == "Abstract not available" or row["dbpedia_director"] == "Director not available":

                        print(f"Processing: {row['title']}")

                        # Get data from Wikipedia
                        abstract, director = get_wikipedia_data(row["title"], row.get("year", None))

                        # Update fields if we found new information
                        if row["dbpedia_abstract"] == "Abstract not available" and abstract != "Abstract not available":
                            row["dbpedia_abstract"] = abstract
                            print(f"  Abstract updated: {abstract[:50]}...")
                            updated_count += 1

                        if row["dbpedia_director"] == "Director not available" and director != "Director not available":
                            row["dbpedia_director"] = director
                            print(f"  Director found: {director}")
                            updated_count += 1

                    writer.writerow(row)
                    films_processed += 1

                    # Partial save every 5 processed films
                    if films_processed % 5 == 0:
                        print(f"✅ Partial save after {films_processed} films")
                        # Create backup copy
                        outfile.flush()
                        shutil.copy(output_file, temp_file)

        # Remove temporary file at the end
        if os.path.exists(temp_file):
            os.remove(temp_file)

        print(f"Update completed! {updated_count} fields updated on {total_count} rows.")

    except Exception as e:
        print(f"Error during CSV file update: {e}")
        print("If a partial save existed, you can resume from there.")


if __name__ == "__main__":
    # Request file paths from user
    input_file = "movies_with_abstracts.csv"
    output_file = "movies_with_abstract_wikipedia_3.csv"

    # Update CSV file
    update_csv_file(input_file, output_file)
