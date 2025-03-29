const API_URL = 'http://127.0.0.1:5000'; // Replace with your API URL
const TMDB_API_KEY = '017cdbfa0649ce3e66ac957f8136ca80';
const TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w185';

// Show loading indicator
function showLoading(id) {
  document.getElementById(id).style.display = 'block';
}

// Hide loading indicator
function hideLoading(id) {
  document.getElementById(id).style.display = 'none';
}

function shortenTitle(title, maxLength = 50) {
  const match = title.match(/\((\d{4})\)$/); // Trova l'anno tra parentesi
  const year = match ? ` ${match[0]}` : ''; // Se l'anno esiste, lo mantiene

  let cleanTitle = title.replace(/\(\d{4}\)$/, '').trim(); // Rimuove l'anno dal titolo
  if (cleanTitle.length > maxLength) {
    cleanTitle = cleanTitle.substring(0, maxLength).trim() + '...';
  }

  return cleanTitle + year; // Ricompone il titolo con l'anno
}

async function searchMovies() {
  const query = document.getElementById('searchQuery').value;
  if (!query) {
    displayError('searchResults', 'Please enter a movie title');
    return;
  }

  showLoading('searchLoading');
  try {
    const response = await fetch(`${API_URL}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    const results = await response.json();

    for (const result of results) {
      result.title = shortenTitle(result.title, 22);
      if (result.genres.length > 30) {
        result.genres = result.genres.substring(0, 30) + '...';
      }
    }

    displayResults('searchResults', results, ['id', 'title', 'genres']);
  } catch (error) {
    displayError('searchResults', error);
  } finally {
    hideLoading('searchLoading');
  }
}

async function getMovieDetails() {
  const movieId = document.getElementById('movieId').value;
  if (!movieId) {
    displayError('movieDetails', 'Please enter a movie ID');
    return;
  }

  showLoading('movieDetailsLoading');
  try {
    const response = await fetch(`${API_URL}/movie/${movieId}`);
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'movieDetails',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }

    displayDetails('movieDetails', data);

    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (error) {
    displayError('movieDetails', 'Errore di connessione al server');
  } finally {
    hideLoading('movieDetailsLoading');
  }
}

async function getUserMovies() {
  const userId = document.getElementById('userIdMovies').value;
  if (!userId) {
    displayError('userMoviesResults', 'Please enter a user ID');
    return;
  }

  showLoading('userMoviesLoading');

  try {
    response = await fetch(`${API_URL}/user/${userId}`);
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'userMoviesResults',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }

    for (const movie of data.results) {
      movie.title = shortenTitle(movie.title, 17);
    }

    displayUserMovies('userMoviesResults', data);
  } catch (error) {
    displayError('userMoviesResults', 'Errore di connessione al server');
  } finally {
    hideLoading('userMoviesLoading');
  }
}

async function getContentRecommendations() {
  const title = document.getElementById('contentTitle').value;
  if (!title) {
    displayError('contentResults', 'Please enter a movie title');
    return;
  }

  showLoading('contentLoading');
  try {
    const response = await fetch(`${API_URL}/recommend/content`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'contentResults',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }

    displayResults('contentResults', data, ['id', 'title', 'genres']);
  } catch (error) {
    displayError('contentResults', 'Errore di connessione al server');
  } finally {
    hideLoading('contentLoading');
  }
}

async function getContentRecommendationsWithMab() {
  const title = document.getElementById('contentTitleMab').value;
  if (!title) {
    displayError('contentMabResults', 'Please enter a movie title');
    return;
  }

  showLoading('contentMabLoading');
  try {
    const response = await fetch(`${API_URL}/recommend/content_mab`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'contentMabResults',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }
    displayResults('contentMabResults', data, ['id', 'title', 'genres']);
  } catch (error) {
    displayError('contentMabResults', 'Errore di connessione al server');
  } finally {
    hideLoading('contentMabLoading');
  }
}

async function getItemRecommendations() {
  const movieId = document.getElementById('itemId').value;
  if (!movieId) {
    displayError('itemResults', 'Please enter a movie ID');
    return;
  }

  showLoading('itemLoading');
  try {
    const response = await fetch(`${API_URL}/recommend/item`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ movie_id: parseInt(movieId) }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'itemResults',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }

    displayResults('itemResults', data, ['id', 'title', 'genres']);
  } catch (error) {
    displayError('itemResults', 'Errore di connessione al server');
  } finally {
    hideLoading('itemLoading');
  }
}

async function getItemRecommendationsWithMab() {
  const movieId = document.getElementById('itemIdMAB').value;
  if (!movieId) {
    displayError('itemResultsMAB', 'Please enter a movie ID');
    return;
  }

  showLoading('itemLoadingMAB');
  try {
    const response = await fetch(`${API_URL}/recommend/item_mab`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ movie_id: parseInt(movieId) }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'itemResultsMAB',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }

    displayResults('itemResultsMAB', data, ['id', 'title', 'genres']);
  } catch (error) {
    displayError('itemResultsMAB', 'Errore di connessione al server');
  } finally {
    hideLoading('itemLoadingMAB');
  }
}

async function getUserRecommendations() {
  const userId = document.getElementById('userId').value;
  if (!userId) {
    displayError('userResults', 'Please enter a user ID');
    return;
  }

  showLoading('userLoading');
  try {
    const response = await fetch(`${API_URL}/recommend/user`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: parseInt(userId) }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'userResults',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }

    displayResults('userResults', data, ['id', 'title', 'genres', 'values']);
  } catch (error) {
    displayError('userResults', 'Errore di connessione al server');
  } finally {
    hideLoading('userLoading');
  }
}

async function getUserRecommendationsWithMab() {
  const userId = document.getElementById('userIdMAB').value;
  if (!userId) {
    displayError('userResultsMAB', 'Please enter a user ID');
    return;
  }

  showLoading('userLoadingMAB');
  try {
    const response = await fetch(`${API_URL}/recommend/user_mab`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: parseInt(userId) }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'userResultsMAB',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }

    displayResults('userResultsMAB', data, ['id', 'title', 'genres', 'values']);
  } catch (error) {
    displayError(
      'userResultsMAB',
      data.message || 'Si è verificato un errore durante la richiesta'
    );
  } finally {
    hideLoading('userLoadingMAB');
  }
}

async function getDirectorMovieRecommendations() {
  const movieId = document.getElementById('directorMovieId').value;
  if (!movieId) {
    displayError('directorMovieResults', 'Please enter a movie ID');
    return;
  }

  showLoading('directorMovieLoading');
  try {
    const response = await fetch(`${API_URL}/recommend/director/movie`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ movie_id: parseInt(movieId) }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'directorMovieResults',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }

    displayDirectorResults('directorMovieResults', data.result);
  } catch (error) {
    displayError('directorMovieResults', 'Errore di connessione al server');
  } finally {
    hideLoading('directorMovieLoading');
  }
}

async function getDirectorNameRecommendations() {
  const director = document.getElementById('directorName').value;
  if (!director) {
    displayError('directorNameResults', 'Please enter a director name');
    return;
  }

  showLoading('directorNameLoading');
  try {
    const response = await fetch(`${API_URL}/recommend/director/name`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ director }),
    });
    const results = await response.json();
    displayDirectorResults('directorNameResults', results.result);
  } catch (error) {
    displayError('directorNameResults', error);
  } finally {
    hideLoading('directorNameLoading');
  }
}

async function getSGDRecommandation() {
  const UserId = document.getElementById('SGDId').value;
  if (!UserId) {
    displayError('SGDMovieResults', 'Please enter a User ID');
    return;
  }

  showLoading('SGDLoading');
  try {
    const response = await fetch(`${API_URL}/recommend/sgd`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: parseInt(UserId) }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'SGDMovieResults',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }

    displayResults('SGDMovieResults', data, ['id', 'title', 'genres']);
  } catch (error) {
    displayError('SGDMovieResults', 'Errore di connessione al server');
  } finally {
    hideLoading('SGDLoading');
  }
}

async function getSGDRecommandationMabLog() {
  const userId = document.getElementById('SGDMabId').value;
  if (!userId) {
    displayError('SGDMabMovieResults', 'Please enter a User ID');
    return;
  }

  showLoading('SGDMabLoading');
  try {
    const response = await fetch(`${API_URL}/recommend/sgd_mab_log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: parseInt(userId) }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'SGDMabMovieResults',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }

    displayResults('SGDMabMovieResults', data, ['id', 'title', 'genres']);
  } catch (error) {
    displayError('SGDMabMovieResults', 'Errore di connessione al server');
  } finally {
    hideLoading('SGDMabLoading');
  }
}

async function getPrediction() {
  const movieId = document.getElementById('predictionMovieId').value;
  const userId = document.getElementById('predictionUserId').value;
  if (!movieId || !userId) {
    displayError('predictionResult', 'Please enter both User ID and Movie ID');
    return;
  }

  showLoading('predictionLoading');
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: parseInt(userId),
        movie_id: parseInt(movieId),
      }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'predictionResult',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }

    displayPrediction('predictionResult', data);
  } catch (error) {
    displayError('predictionResult', 'Errore di connessione al server');
  } finally {
    hideLoading('predictionLoading');
  }
}

async function getPredictionKNN() {
  const movieId = document.getElementById('predictionMovieIdKNN').value;
  const userId = document.getElementById('predictionUserIdKNN').value;
  if (!movieId || !userId) {
    displayError(
      'predictionResultKNN',
      'Please enter both User ID and Movie ID'
    );
    return;
  }

  showLoading('predictionLoadingKNN');
  try {
    const response = await fetch(`${API_URL}/predict_knn`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: parseInt(userId),
        movie_id: parseInt(movieId),
      }),
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      displayError(
        'predictionResultKNN',
        data.message || 'Si è verificato un errore durante la richiesta'
      );
      return;
    }

    displayPrediction('predictionResultKNN', data);
  } catch (error) {
    displayError('predictionResultKNN', 'Errore di connessione al server');
  } finally {
    hideLoading('predictionLoadingKNN');
  }
}

async function getMoviePoster(tmdbId, imdbId) {
  // Prima prova con tmdbId se disponibile
  if (tmdbId) {
    try {
      const response = await fetch(
        `https://api.themoviedb.org/3/movie/${tmdbId}?api_key=${TMDB_API_KEY}`
      );
      const data = await response.json();

      if (data.poster_path) {
        return `${TMDB_IMAGE_BASE_URL}${data.poster_path}`;
      }
    } catch (error) {
      console.error('Errore nel recupero del poster con tmdbId:', error);
    }
  }

  // Se tmdbId non ha funzionato o non è disponibile, prova con imdbId
  if (imdbId) {
    try {
      // Se imdbId ha zeri iniziali, assicurati che siano preservati
      const formattedImdbId = imdbId.toString().startsWith('tt')
        ? imdbId
        : `tt${imdbId}`;

      const response = await fetch(
        `https://api.themoviedb.org/3/find/${formattedImdbId}?api_key=${TMDB_API_KEY}&external_source=imdb_id`
      );
      const data = await response.json();

      if (
        data.movie_results &&
        data.movie_results.length > 0 &&
        data.movie_results[0].poster_path
      ) {
        return `${TMDB_IMAGE_BASE_URL}${data.movie_results[0].poster_path}`;
      }
    } catch (error) {
      console.error('Errore nel recupero del poster con imdbId:', error);
    }
  }

  // Restituisci un'immagine placeholder se non è stato possibile ottenere la copertina
  return 'https://placehold.co/140x210/png?text=No+Image+Found&font=roboto';
}

async function displayResults(elementId, data, columns) {
  const element = document.getElementById(elementId);
  element.innerHTML = '';

  // Verifica se i dati sono nel nuovo formato con original_movie e recommendations
  const hasOriginalMovie =
    data && data.original_movie && data.original_movie.title;
  const results = hasOriginalMovie
    ? data.recommendations
    : data.results || data;

  if (results && results.length > 0) {
    // Se abbiamo informazioni sul film originale, creiamo un banner in cima
    if (hasOriginalMovie) {
      const originalMovieBanner = document.createElement('div');
      originalMovieBanner.className = 'original-movie-banner';

      const bannerTitle = document.createElement('h3');
      bannerTitle.textContent = 'Raccomandazioni basate su:';

      const movieTitle = document.createElement('div');
      movieTitle.className = 'original-movie-title';
      movieTitle.textContent = data.original_movie.title;

      // Se è disponibile anche l'ID, lo mostriamo
      if (data.original_movie.movieId) {
        const movieId = document.createElement('span');
        movieId.className = 'original-movie-id';
        movieId.textContent = `ID: ${data.original_movie.movieId}`;
        movieTitle.appendChild(movieId);
      }

      originalMovieBanner.appendChild(bannerTitle);
      originalMovieBanner.appendChild(movieTitle);
      element.appendChild(originalMovieBanner);
    } else if (data.userId) {
      const userBanner = document.createElement('div');
      userBanner.className = 'original-movie-banner';

      const bannerTitle = document.createElement('h3');
      bannerTitle.textContent = "Raccomandazioni basate sull'utente:";

      const userIdElement = document.createElement('div');
      userIdElement.className = 'original-movie-title';
      userIdElement.textContent = `User ID: ${data.userId}`;

      userBanner.appendChild(bannerTitle);
      userBanner.appendChild(userIdElement);
      element.appendChild(userBanner);
    }

    const list = document.createElement('ul');
    list.className = 'movie-list';

    // Processa ogni film nella lista dei risultati
    for (const item of results) {
      const listItem = document.createElement('li');
      listItem.className = 'movie-card-with-poster';
      listItem.style.cursor = 'pointer'; // Aggiunge effetto cliccabile
      listItem.onclick = () => getMovieDetailsFromId(item.movieId); // Aggiungi evento click

      // Aggiunta del poster del film
      const posterContainer = document.createElement('div');
      posterContainer.className = 'poster-container';

      const posterImg = document.createElement('img');
      posterImg.className = 'movie-poster';
      posterImg.src = 'https://placehold.co/140x210/png?text=Loading...';
      posterImg.alt = `${item.title || 'Movie'} poster`;

      // Carica poster in modo asincrono
      if (item.tmdbId || item.imdbId) {
        getMoviePoster(item.tmdbId, item.imdbId).then((url) => {
          posterImg.src = url;
        });
      } else {
        posterImg.src = 'https://placehold.co/140x210/png?text=No+Poster';
      }

      posterContainer.appendChild(posterImg);
      listItem.appendChild(posterContainer);

      // Contenitore per le informazioni
      const infoContainer = document.createElement('div');
      infoContainer.className = 'movie-info';

      // Header con titolo
      const header = document.createElement('div');
      header.className = 'movie-header';

      const titleSpan = document.createElement('span');
      titleSpan.className = 'movie-title';
      titleSpan.textContent = item.title || 'Unknown Title';
      header.appendChild(titleSpan);

      if (item.movieId) {
        const idBadge = document.createElement('span');
        idBadge.className = 'movie-id';
        idBadge.textContent = `ID: ${item.movieId}`;
        header.appendChild(idBadge);
      }

      infoContainer.appendChild(header);

      // Metadata del film (anno, durata, genere)
      const metadata = document.createElement('div');
      metadata.className = 'movie-metadata';

      // Anno (se disponibile)
      if (item.year) {
        const yearItem = document.createElement('span');
        yearItem.className = 'metadata-item';
        yearItem.innerHTML = `<i class="far fa-calendar-alt"></i> ${item.year}`;
        metadata.appendChild(yearItem);
      }

      // Generi (se disponibili)
      if (item.genres) {
        const genreItem = document.createElement('span');
        genreItem.className = 'metadata-item';
        genreItem.innerHTML = `<i class="fas fa-film"></i> ${item.genres}`;
        metadata.appendChild(genreItem);
      }

      // Durata (se disponibile)
      if (item.runtime) {
        const runtimeItem = document.createElement('span');
        runtimeItem.className = 'metadata-item';
        runtimeItem.innerHTML = `<i class="far fa-clock"></i> ${item.runtime} min`;
        metadata.appendChild(runtimeItem);
      }

      // Valutazione (se disponibile)
      if (item.avg_rating) {
        const ratingItem = document.createElement('span');
        ratingItem.className = 'metadata-item';
        ratingItem.innerHTML = `<i class="fas fa-star"></i> ${parseFloat(
          item.avg_rating
        ).toFixed(1)}`;
        metadata.appendChild(ratingItem);
      }

      infoContainer.appendChild(metadata);

      // Contenuto principale
      const content = document.createElement('div');
      content.className = 'movie-content';

      // Aggiungi le altre colonne specificate
      columns.forEach((col) => {
        // Salta colonne già mostrate
        if (
          [
            'title',
            'id',
            'genres',
            'year',
            'runtime',
            'avg_rating',
            'values',
          ].includes(col)
        )
          return;

        const detail = document.createElement('div');
        detail.className = 'movie-detail';

        const label = document.createElement('span');
        label.className = 'detail-label';
        label.textContent = col.charAt(0).toUpperCase() + col.slice(1) + ':';

        const value = document.createElement('span');
        value.className = 'detail-value';
        value.textContent = item[col];

        detail.appendChild(label);
        detail.appendChild(value);
        content.appendChild(detail);
      });

      infoContainer.appendChild(content);

      // Aggiungi informazioni aggiuntive (se disponibili)
      if (item.dbpedia_abstract || item.overview) {
        const additionalInfo = document.createElement('div');
        additionalInfo.className = 'additional-movie-info';

        // Aggiungi la sinossi/abstract se disponibile
        if (item.dbpedia_abstract || item.overview) {
          const abstract = document.createElement('p');
          const abstractText = item.dbpedia_abstract || item.overview;
          abstract.textContent =
            abstractText.length > 150
              ? abstractText.substring(0, 150) + '...'
              : abstractText;
          additionalInfo.appendChild(abstract);
        }

        // Aggiungi regista se disponibile
        if (item.dbpedia_director) {
          const director = document.createElement('p');
          director.innerHTML = `<strong>Director:</strong> ${item.dbpedia_director}`;
          additionalInfo.appendChild(director);
        }

        infoContainer.appendChild(additionalInfo);
      }

      listItem.appendChild(infoContainer);
      list.appendChild(listItem);
    }

    element.appendChild(list);

    element.scrollTop = 0;
  } else {
    displayEmptyState(element);

    element.scrollTop = 0;
  }
}

function getMovieDetailsFromId(movieId) {
  if (!movieId) return;

  document.getElementById('movieId').value = movieId; // Aggiorna il campo input con l'ID selezionato
  getMovieDetails(); // Esegui la query per i dettagli del film
}

async function displayDetails(elementId, details) {
  const element = document.getElementById(elementId);
  element.innerHTML = '';

  if (details && Object.keys(details).length > 0) {
    const movieCard = document.createElement('div');
    movieCard.className = 'movie-card-with-poster';

    // Contenitore per il poster
    const posterContainer = document.createElement('div');
    posterContainer.className = 'poster-container';

    const posterImg = document.createElement('img');
    posterImg.className = 'movie-poster';
    posterImg.src = 'https://placehold.co/140x210/png?text=Loading...';
    posterImg.alt = `${details.title || 'Movie'} poster`;

    // Carica poster in modo asincrono
    if (details.tmdbId || details.imdbId) {
      getMoviePoster(details.tmdbId, details.imdbId).then((url) => {
        posterImg.src = url;
      });
    } else {
      posterImg.src = 'https://placehold.co/140x210/png?text=No+Poster';
    }

    posterContainer.appendChild(posterImg);
    movieCard.appendChild(posterContainer);

    // Contenitore per le informazioni
    const infoContainer = document.createElement('div');
    infoContainer.className = 'movie-info';

    // Header con titolo
    const header = document.createElement('div');
    header.className = 'movie-header';

    const titleSpan = document.createElement('span');
    titleSpan.className = 'movie-title';
    titleSpan.textContent = shortenTitle(details.title || 'Unknown Title', 22);
    header.appendChild(titleSpan);

    if (details.movieId) {
      const idBadge = document.createElement('span');
      idBadge.className = 'movie-id';
      idBadge.textContent = `ID: ${details.movieId}`;
      header.appendChild(idBadge);
    }

    infoContainer.appendChild(header);

    // Metadata del film (anno, durata, genere)
    const metadata = document.createElement('div');
    metadata.className = 'movie-metadata';

    // Anno (se disponibile)
    if (details.year) {
      const yearItem = document.createElement('span');
      yearItem.className = 'metadata-item';
      yearItem.innerHTML = `<i class="far fa-calendar-alt"></i> ${details.year}`;
      metadata.appendChild(yearItem);
    }

    // Generi (se disponibili)
    if (details.genres) {
      if (details.genres.length > 46) {
        details.genres = details.genres.substring(0, 46) + '...';
      }
      const genreItem = document.createElement('span');
      genreItem.className = 'metadata-item';
      genreItem.innerHTML = `<i class="fas fa-film"></i> ${details.genres}`;
      metadata.appendChild(genreItem);
    }

    // Durata (se disponibile)
    if (details.runtime) {
      const runtimeItem = document.createElement('span');
      runtimeItem.className = 'metadata-item';
      runtimeItem.innerHTML = `<i class="far fa-clock"></i> ${details.runtime} min`;
      metadata.appendChild(runtimeItem);
    }

    // Valutazione (se disponibile)
    if (details.avg_rating) {
      const ratingItem = document.createElement('span');
      ratingItem.className = 'metadata-item';

      // Create star rating visualization
      let starsHtml = '';
      const starCount = Math.round(details.avg_rating);
      for (let i = 0; i < 5; i++) {
        starsHtml += `<i class="${i < starCount ? 'fas' : 'far'} fa-star"></i>`;
      }

      ratingItem.innerHTML = `${starsHtml} ${parseFloat(
        details.avg_rating
      ).toFixed(1)}`;
      if (details.num_ratings) {
        ratingItem.innerHTML += ` (${details.num_ratings} ratings)`;
      }

      metadata.appendChild(ratingItem);
    }

    infoContainer.appendChild(metadata);

    // Contenuto principale - abstract/sinossi
    if (details.dbpedia_abstract) {
      const abstractSection = document.createElement('div');
      abstractSection.className = 'additional-movie-info';

      const abstractTitle = document.createElement('h4');
      abstractTitle.textContent = 'Abstract';
      abstractTitle.style.marginBottom = '5px';
      abstractTitle.style.marginLeft = '5px';
      abstractSection.appendChild(abstractTitle);

      const abstractText = document.createElement('p');
      abstractText.textContent = details.dbpedia_abstract;
      abstractSection.appendChild(abstractText);

      infoContainer.appendChild(abstractSection);
    }

    // Altre informazioni
    const otherInfo = document.createElement('div');
    otherInfo.className = 'movie-content';

    // Regista (se disponibile)
    if (details.dbpedia_director) {
      const directorDetail = document.createElement('div');
      directorDetail.className = 'movie-detail';

      const directorLabel = document.createElement('span');
      directorLabel.className = 'detail-label';
      directorLabel.textContent = 'Director';

      const directorValue = document.createElement('span');
      directorValue.className = 'detail-value';
      directorValue.textContent = details.dbpedia_director;

      directorDetail.appendChild(directorLabel);
      directorDetail.appendChild(directorValue);
      otherInfo.appendChild(directorDetail);
    }

    // Aggiungi altre proprietà
    for (const key in details) {
      // Salta quelle già visualizzate
      if (
        [
          'title',
          'movieId',
          'dbpedia_abstract',
          'dbpedia_director',
          'avg_rating',
          'num_ratings',
          'genres',
          'year',
          'runtime',
          'tmdbId',
          'imdbId',
        ].includes(key)
      ) {
        continue;
      }

      const detail = document.createElement('div');
      detail.className = 'movie-detail';

      const label = document.createElement('span');
      label.className = 'detail-label';
      label.textContent =
        key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ');

      const value = document.createElement('span');
      value.className = 'detail-value';
      value.textContent = details[key];

      detail.appendChild(label);
      detail.appendChild(value);
      otherInfo.appendChild(detail);
    }

    infoContainer.appendChild(otherInfo);
    movieCard.appendChild(infoContainer);
    element.appendChild(movieCard);
    element.scrollTop = 0;
  } else {
    displayEmptyState(element);
    element.scrollTop = 0;
  }
}

function displayDirectorResults(elementId, result) {
  const element = document.getElementById(elementId);
  element.innerHTML = '';

  if (result && result.success) {
    const resultCard = document.createElement('div');
    resultCard.className = 'result-card';

    // Intestazione con informazioni sul regista
    const header = document.createElement('div');
    header.className = 'result-header';
    header.innerHTML = `<h3>Film diretti da ${result.director}</h3>
                               <p>Mostrati ${result.films_shown} di ${result.total_films_found} film totali</p>`;
    resultCard.appendChild(header);

    // Lista dei film
    if (result.films && result.films.length > 0) {
      const filmsList = document.createElement('div');
      filmsList.className = 'films-list';

      result.films.forEach((film, index) => {
        const filmItem = document.createElement('div');
        filmItem.className = 'film-item';

        // Titolo e anno
        const filmHeader = document.createElement('h4');
        filmHeader.textContent = `${index + 1}. ${film.title} (${film.year})`;
        filmItem.appendChild(filmHeader);

        // Attori
        if (film.actors && film.actors.length > 0) {
          const actorsList = document.createElement('div');
          actorsList.className = 'actors-list';

          const actorsTitle = document.createElement('p');
          actorsTitle.textContent = 'Attori principali:';
          actorsList.appendChild(actorsTitle);

          const actorsUl = document.createElement('ul');
          film.actors.forEach((actor) => {
            const actorLi = document.createElement('li');
            actorLi.textContent = actor;
            actorsUl.appendChild(actorLi);
          });
          actorsList.appendChild(actorsUl);

          // Se ci sono attori aggiuntivi
          if (film.additional_actors_count > 0) {
            const additionalInfo = document.createElement('p');
            additionalInfo.className = 'additional-info';
            additionalInfo.textContent = `... e altri ${film.additional_actors_count} attori`;
            actorsList.appendChild(additionalInfo);
          }

          filmItem.appendChild(actorsList);
        } else {
          const noActors = document.createElement('p');
          noActors.textContent = 'Attori: Informazione non disponibile';
          filmItem.appendChild(noActors);
        }

        filmsList.appendChild(filmItem);
      });

      resultCard.appendChild(filmsList);
    } else {
      const noFilms = document.createElement('p');
      noFilms.textContent = 'Nessun film disponibile';
      resultCard.appendChild(noFilms);
    }
    element.appendChild(resultCard);

    element.scrollTop = 0;
  } else if (result && !result.success) {
    // Gestione caso di errore restituito dal server
    const errorMessage = document.createElement('div');
    errorMessage.className = 'error-message';
    errorMessage.textContent =
      result.message ||
      'Si è verificato un errore durante il recupero dei film.';
    element.appendChild(errorMessage);
    element.scrollTop = 0;
  } else {
    displayEmptyState(element);
    element.scrollTop = 0;
  }
}

function displayUserMovies(elementId, data) {
  const element = document.getElementById(elementId);
  element.innerHTML = '';

  // Verifica se i dati sono nel nuovo formato con original_movie e recommendations
  const hasOriginalMovie =
    data && data.original_movie && data.original_movie.title;
  const results = hasOriginalMovie
    ? data.recommendations
    : data.results || data;

  if (results && results.length > 0) {
    const userBanner = document.createElement('div');
    userBanner.className = 'original-movie-banner';

    const bannerTitle = document.createElement('h3');
    bannerTitle.textContent = "Film visti dall'utente:";

    const userIdElement = document.createElement('div');
    userIdElement.className = 'original-movie-title';
    userIdElement.textContent = `User: ${data.userId}`;

    userBanner.appendChild(bannerTitle);
    userBanner.appendChild(userIdElement);
    element.appendChild(userBanner);

    const list = document.createElement('ul');
    list.className = 'movie-list';

    // Processa ogni film nella lista dei risultati
    for (const item of results) {
      const listItem = document.createElement('li');
      listItem.className = 'movie-card-with-poster';
      listItem.style.cursor = 'pointer'; // Aggiunge effetto cliccabile
      listItem.onclick = () => getMovieDetailsFromId(item.movieId); // Aggiungi evento click

      // Aggiunta del poster del film
      const posterContainer = document.createElement('div');
      posterContainer.className = 'poster-container';

      const posterImg = document.createElement('img');
      posterImg.className = 'movie-poster';
      posterImg.src = 'https://placehold.co/140x210/png?text=Loading...';
      posterImg.alt = `${item.title || 'Movie'} poster`;

      // Carica poster in modo asincrono
      if (item.tmdbId || item.imdbId) {
        getMoviePoster(item.tmdbId, item.imdbId).then((url) => {
          posterImg.src = url;
        });
      } else {
        posterImg.src = 'https://placehold.co/140x210/png?text=No+Poster';
      }

      posterContainer.appendChild(posterImg);
      listItem.appendChild(posterContainer);

      // Contenitore per le informazioni
      const infoContainer = document.createElement('div');
      infoContainer.className = 'movie-info';

      // Header con titolo
      const header = document.createElement('div');
      header.className = 'movie-header';

      const titleSpan = document.createElement('span');
      titleSpan.className = 'movie-title';
      titleSpan.textContent = item.title || 'Unknown Title';
      header.appendChild(titleSpan);

      if (item.movieId) {
        const idBadge = document.createElement('span');
        idBadge.className = 'movie-id';
        idBadge.textContent = `ID: ${item.movieId}`;
        header.appendChild(idBadge);
      }

      infoContainer.appendChild(header);

      // Metadata del film (anno, durata, genere)
      const metadata = document.createElement('div');
      metadata.className = 'movie-metadata';

      // Anno (se disponibile)
      if (item.year) {
        const yearItem = document.createElement('span');
        yearItem.className = 'metadata-item';
        yearItem.innerHTML = `<i class="far fa-calendar-alt"></i> ${item.year}`;
        metadata.appendChild(yearItem);
      }

      // Generi (se disponibili)
      if (item.genres) {
        if (item.genres.length > 40) {
          item.genres = item.genres.substring(0, 40) + '...';
        }
        const genreItem = document.createElement('span');
        genreItem.className = 'metadata-item';
        genreItem.innerHTML = `<i class="fas fa-film"></i> ${item.genres}`;
        metadata.appendChild(genreItem);
      }

      infoContainer.appendChild(metadata);

      // Contenuto principale
      const content = document.createElement('div');
      content.className = 'movie-content';

      const detail = document.createElement('div');
      detail.className = 'movie-detail';

      const label = document.createElement('span');
      label.className = 'detail-label';
      label.textContent = 'Rated:';

      const value = document.createElement('span');
      value.className = 'metadata-item';
      let starsHtml = '';
      let index = 0;
      const starCount = Math.floor(item.rating);
      for (let i = 0; i < starCount; i++) {
        starsHtml += `<i class="fas fa-star"></i>`;
        index = i;
      }
      if (item.rating - starCount > 0) {
        starsHtml += `<i class="fas fa-star-half-alt"></i>`;
        index++;
      }
      for (let i = index + 1; i < 5; i++) {
        starsHtml += `<i class="far fa-star"></i>`;
      }
      if (starCount == 0) {
        starsHtml += `<i class="far fa-star"></i>`;
      }

      value.innerHTML = `${starsHtml} ${parseFloat(item.rating).toFixed(1)}`;
      //value.textContent = item.rating;

      detail.appendChild(label);
      detail.appendChild(value);
      content.appendChild(detail);

      infoContainer.appendChild(content);

      listItem.appendChild(infoContainer);
      list.appendChild(listItem);
    }

    element.appendChild(list);

    element.scrollTop = 0;
  } else {
    displayEmptyState(element);

    element.scrollTop = 0;
  }
}

function getMovieDetailsFromId(movieId) {
  if (!movieId) return;

  document.getElementById('movieId').value = movieId; // Aggiorna il campo input con l'ID selezionato
  getMovieDetails(); // Esegui la query per i dettagli del film
}

async function displayDetails(elementId, details) {
  const element = document.getElementById(elementId);
  element.innerHTML = '';

  if (details && Object.keys(details).length > 0) {
    const movieCard = document.createElement('div');
    movieCard.className = 'movie-card-with-poster';

    // Contenitore per il poster
    const posterContainer = document.createElement('div');
    posterContainer.className = 'poster-container';

    const posterImg = document.createElement('img');
    posterImg.className = 'movie-poster';
    posterImg.src = 'https://placehold.co/140x210/png?text=Loading...';
    posterImg.alt = `${details.title || 'Movie'} poster`;

    // Carica poster in modo asincrono
    if (details.tmdbId || details.imdbId) {
      getMoviePoster(details.tmdbId, details.imdbId).then((url) => {
        posterImg.src = url;
      });
    } else {
      posterImg.src = 'https://placehold.co/140x210/png?text=No+Poster';
    }

    posterContainer.appendChild(posterImg);
    movieCard.appendChild(posterContainer);

    // Contenitore per le informazioni
    const infoContainer = document.createElement('div');
    infoContainer.className = 'movie-info';

    // Header con titolo
    const header = document.createElement('div');
    header.className = 'movie-header';

    const titleSpan = document.createElement('span');
    titleSpan.className = 'movie-title';
    titleSpan.textContent = shortenTitle(details.title || 'Unknown Title', 24);
    header.appendChild(titleSpan);

    if (details.movieId) {
      const idBadge = document.createElement('span');
      idBadge.className = 'movie-id';
      idBadge.textContent = `ID: ${details.movieId}`;
      header.appendChild(idBadge);
    }

    infoContainer.appendChild(header);

    // Metadata del film (anno, durata, genere)
    const metadata = document.createElement('div');
    metadata.className = 'movie-metadata';

    // Anno (se disponibile)
    if (details.year) {
      const yearItem = document.createElement('span');
      yearItem.className = 'metadata-item';
      yearItem.innerHTML = `<i class="far fa-calendar-alt"></i> ${details.year}`;
      metadata.appendChild(yearItem);
    }

    // Generi (se disponibili)
    if (details.genres) {
      if (details.genres.length > 46) {
        details.genres = details.genres.substring(0, 46) + '...';
      }
      const genreItem = document.createElement('span');
      genreItem.className = 'metadata-item';
      genreItem.innerHTML = `<i class="fas fa-film"></i> ${details.genres}`;
      metadata.appendChild(genreItem);
    }

    // Durata (se disponibile)
    if (details.runtime) {
      const runtimeItem = document.createElement('span');
      runtimeItem.className = 'metadata-item';
      runtimeItem.innerHTML = `<i class="far fa-clock"></i> ${details.runtime} min`;
      metadata.appendChild(runtimeItem);
    }

    // Valutazione (se disponibile)
    if (details.avg_rating) {
      const ratingItem = document.createElement('span');
      ratingItem.className = 'metadata-item';

      // Create star rating visualization
      let starsHtml = '';
      let avg_temp = details.avg_rating;
      let index = 0;

      if (
        details.avg_rating - Math.floor(details.avg_rating) > 0.25 &&
        details.avg_rating - Math.floor(details.avg_rating) < 0.75
      ) {
        avg_temp = Math.floor(details.avg_rating) + 0.5;
      } else if (details.avg_rating - Math.floor(details.avg_rating) >= 0.75) {
        avg_temp = Math.floor(details.avg_rating) + 1;
      }

      const starCount = Math.floor(avg_temp);

      for (let i = 0; i < starCount; i++) {
        starsHtml += `<i class="fas fa-star"></i>`;
        index = i;
      }

      if (avg_temp - starCount > 0) {
        starsHtml += `<i class="fas fa-star-half-alt"></i>`;
        index++;
      }

      for (let i = index + 1; i < 5; i++) {
        starsHtml += `<i class="far fa-star"></i>`;
      }

      if (starCount == 0) {
        starsHtml += `<i class="far fa-star"></i>`;
      }

      ratingItem.innerHTML = `${starsHtml} ${parseFloat(
        details.avg_rating
      ).toFixed(1)}`;
      if (details.num_ratings) {
        ratingItem.innerHTML += ` (${details.num_ratings} ratings)`;
      }

      metadata.appendChild(ratingItem);
    }

    infoContainer.appendChild(metadata);

    // Contenuto principale - abstract/sinossi
    if (details.dbpedia_abstract) {
      const abstractSection = document.createElement('div');
      abstractSection.className = 'additional-movie-info';

      const abstractTitle = document.createElement('h4');
      abstractTitle.textContent = 'Abstract';
      abstractTitle.style.marginBottom = '5px';
      abstractTitle.style.marginLeft = '5px';
      abstractSection.appendChild(abstractTitle);

      const abstractText = document.createElement('p');
      abstractText.textContent = details.dbpedia_abstract;
      abstractSection.appendChild(abstractText);

      infoContainer.appendChild(abstractSection);
    }

    // Altre informazioni
    const otherInfo = document.createElement('div');
    otherInfo.className = 'movie-content';

    // Regista (se disponibile)
    if (details.dbpedia_director) {
      const directorDetail = document.createElement('div');
      directorDetail.className = 'movie-detail';

      const directorLabel = document.createElement('span');
      directorLabel.className = 'detail-label';
      directorLabel.textContent = 'Director';

      const directorValue = document.createElement('span');
      directorValue.className = 'detail-value';
      directorValue.textContent = details.dbpedia_director;

      directorDetail.appendChild(directorLabel);
      directorDetail.appendChild(directorValue);
      otherInfo.appendChild(directorDetail);
    }

    // Aggiungi altre proprietà
    for (const key in details) {
      // Salta quelle già visualizzate
      if (
        [
          'title',
          'movieId',
          'dbpedia_abstract',
          'dbpedia_director',
          'avg_rating',
          'num_ratings',
          'genres',
          'year',
          'runtime',
          'tmdbId',
          'imdbId',
        ].includes(key)
      ) {
        continue;
      }

      const detail = document.createElement('div');
      detail.className = 'movie-detail';

      const label = document.createElement('span');
      label.className = 'detail-label';
      label.textContent =
        key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ');

      const value = document.createElement('span');
      value.className = 'detail-value';
      value.textContent = details[key];

      detail.appendChild(label);
      detail.appendChild(value);
      otherInfo.appendChild(detail);
    }

    infoContainer.appendChild(otherInfo);
    movieCard.appendChild(infoContainer);
    element.appendChild(movieCard);
    element.scrollTop = 0;
  } else {
    displayEmptyState(element);
    element.scrollTop = 0;
  }
}

function displayDirectorResults(elementId, result) {
  const element = document.getElementById(elementId);
  element.innerHTML = '';

  if (result && result.success) {
    const resultCard = document.createElement('div');
    resultCard.className = 'result-card';

    // Intestazione con informazioni sul regista
    const header = document.createElement('div');
    header.className = 'result-header';
    header.innerHTML = `<h3>Film diretti da ${result.director}</h3>
                               <p>Mostrati ${result.films_shown} di ${result.total_films_found} film totali</p>`;
    resultCard.appendChild(header);

    // Lista dei film
    if (result.films && result.films.length > 0) {
      const filmsList = document.createElement('div');
      filmsList.className = 'films-list';

      result.films.forEach((film, index) => {
        const filmItem = document.createElement('div');
        filmItem.className = 'film-item';

        // Titolo e anno
        const filmHeader = document.createElement('h4');
        filmHeader.textContent = `${index + 1}. ${film.title} (${film.year})`;
        filmItem.appendChild(filmHeader);

        // Attori
        if (film.actors && film.actors.length > 0) {
          const actorsList = document.createElement('div');
          actorsList.className = 'actors-list';

          const actorsTitle = document.createElement('p');
          actorsTitle.textContent = 'Attori principali:';
          actorsList.appendChild(actorsTitle);

          const actorsUl = document.createElement('ul');
          film.actors.forEach((actor) => {
            const actorLi = document.createElement('li');
            actorLi.textContent = actor;
            actorsUl.appendChild(actorLi);
          });
          actorsList.appendChild(actorsUl);

          // Se ci sono attori aggiuntivi
          if (film.additional_actors_count > 0) {
            const additionalInfo = document.createElement('p');
            additionalInfo.className = 'additional-info';
            additionalInfo.textContent = `... e altri ${film.additional_actors_count} attori`;
            actorsList.appendChild(additionalInfo);
          }

          filmItem.appendChild(actorsList);
        } else {
          const noActors = document.createElement('p');
          noActors.textContent = 'Attori: Informazione non disponibile';
          filmItem.appendChild(noActors);
        }

        filmsList.appendChild(filmItem);
      });

      resultCard.appendChild(filmsList);
    } else {
      const noFilms = document.createElement('p');
      noFilms.textContent = 'Nessun film disponibile';
      resultCard.appendChild(noFilms);
    }
    element.appendChild(resultCard);

    element.scrollTop = 0;
  } else if (result && !result.success) {
    // Gestione caso di errore restituito dal server
    const errorMessage = document.createElement('div');
    errorMessage.className = 'error-message';
    errorMessage.textContent =
      result.message ||
      'Si è verificato un errore durante il recupero dei film.';
    element.appendChild(errorMessage);
    element.scrollTop = 0;
  } else {
    displayEmptyState(element);
    element.scrollTop = 0;
  }
}

function displayPrediction(elementId, results) {
  const resultElement = document.getElementById(elementId);
  resultElement.innerHTML = '';

  if (!results || results.error) {
    displayError(elementId, results?.error || 'Unknown error occurred');
    return;
  }

  // Extract data from results
  const { userId, movie } = results;

  if (!movie || movie.length === 0) {
    displayError(elementId, 'Movie information not found');
    return;
  }

  const movieInfo = movie[0];
  const alreadyRated = movieInfo.already_rated;
  const prediction = movieInfo.prediction_rating;

  // Create a card to display the prediction and related information
  const card = document.createElement('div');
  card.className = 'prediction-card';

  // Format prediction to 1 decimal place
  const formattedPrediction = prediction.toFixed(1);

  // Create HTML structure with conditional message based on already_rated
  card.innerHTML = `
          <div class="prediction-header">
            <h3>${movieInfo.title || 'Unknown Movie'}</h3>
            <div class="prediction-score">
              <i class="fas fa-star"></i> 
              <span class="prediction-value">${formattedPrediction}</span>/5
            </div>
          </div>
          
          <div class="prediction-message">
            ${
              alreadyRated
                ? `<p class="already-rated">Questo film è già stato visto ed ha il seguente rating: <b>${formattedPrediction}</b></p>`
                : `<p class="prediction-text">La predizione per il seguente film è: <b>${formattedPrediction}</b></p>`
            }
          </div>
          
          <div class="prediction-details">
            <div class="movie-details">
              <h4>Informazioni Film</h4>
              ${
                movieInfo.release_year
                  ? `<p><strong>Anno:</strong> ${movieInfo.release_year}</p>`
                  : ''
              }
              ${
                movieInfo.genres
                  ? `<p><strong>Generi:</strong> ${movieInfo.genres}</p>`
                  : ''
              }
              ${
                movieInfo.avg_rating
                  ? `<p><strong>Valutazione Media:</strong> ${movieInfo.avg_rating.toFixed(
                      1
                    )} (${movieInfo.num_ratings} valutazioni)</p>`
                  : ''
              }
            </div>
            
            <div class="user-details">
              <h4>Informazioni Utente</h4>
              <p><strong>ID Utente:</strong> ${userId}</p>
            </div>
          </div>
        `;

  resultElement.appendChild(card);
  resultElement.scrollTop = 0;

  // Add CSS class to style based on prediction value
  const scoreClass =
    prediction >= 4
      ? 'high-score'
      : prediction >= 3
      ? 'medium-score'
      : 'low-score';
  card.querySelector('.prediction-value').classList.add(scoreClass);
}

function displayError(elementId, errorMessage) {
  const resultElement = document.getElementById(elementId);
  resultElement.innerHTML = '';

  const errorCard = document.createElement('div');
  errorCard.className = 'error-card';

  errorCard.innerHTML = `
        <div class="error-icon">
            <i class="fas fa-exclamation-circle"></i>
        </div>
        <div class="error-content">
            <h3>Errore</h3>
            <p>${errorMessage}</p>
        </div>
    `;

  resultElement.appendChild(errorCard);
}

function displayEmptyState(element) {
  const emptyDiv = document.createElement('div');
  emptyDiv.className = 'empty-state';

  const icon = document.createElement('i');
  icon.className = 'fas fa-film';

  const message = document.createElement('p');
  message.textContent = 'No results found. Try a different search.';

  emptyDiv.appendChild(icon);
  emptyDiv.appendChild(message);
  element.appendChild(emptyDiv);
}

// Add event listeners for Enter key
document.getElementById('searchQuery').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') searchMovies();
});

document.getElementById('movieId').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getMovieDetails();
});

document.getElementById('userIdMovies').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getUserMovies();
});

document.getElementById('contentTitle').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getContentRecommendations();
});

document.getElementById('itemId').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getItemRecommendations();
});

document.getElementById('userId').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getUserRecommendations();
});

document.getElementById('directorMovieId').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getDirectorMovieRecommendations();
});

document.getElementById('directorName').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getDirectorNameRecommendations();
});

document.getElementById('SGDId').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getSGDRecommandation();
});

document.getElementById('SGDMabId').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getSGDRecommandationMabLog();
});

document
  .getElementById('predictionMovieId')
  .addEventListener('keydown', (e) => {
    if (e.key === 'Enter') getPrediction();
  });

document.getElementById('predictionUserId').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getPrediction();
});

document.getElementById('contentTitleMab').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getContentRecommendationsWithMab();
});

document.getElementById('itemIdMAB').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getItemRecommendationsWithMab();
});

document.getElementById('userIdMAB').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') getUserRecommendationsWithMab();
});

document
  .getElementById('predictionUserIdKNN')
  .addEventListener('keydown', (e) => {
    if (e.key === 'Enter') getPredictionKNN();
  });

document
  .getElementById('predictionMovieIdKNN')
  .addEventListener('keydown', (e) => {
    if (e.key === 'Enter') getPredictionKNN();
  });
