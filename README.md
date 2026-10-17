# MovieMind — Recommendation System for Digital Platforms

MovieMind is a local Flask movie-discovery application that demonstrates how user ratings, movie genres, similarity, and evaluation metrics can support explainable recommendations. It is designed as a student-friendly Major Project: the core model is small, modular, and easy to discuss in a viva.

## Problem and objective

Large movie catalogs make it hard to find relevant titles. This project studies MovieLens user–movie ratings and genre metadata, then provides movie similarity, genre preference recommendations, catalog search, basic user taste capture, analytics, and a reproducible offline evaluation.

## Features

- Responsive, dark streaming-inspired interface branded MovieMind.
- Home page with featured titles and an explanation of the pipeline.
- Movie discovery with partial title search, genre filter, and rating/popularity/title sorting.
- Movie detail page and similar-title recommendations with a reason and similarity score.
- Recommendations from a liked title or selected genres.
- Session-only profile ratings and personalized genre suggestions (no account required).
- Analytics dashboard with dataset counts, rating distribution, top genres, most-rated movies, and average rating by genre.
- Model evaluation with Precision@10, Recall@10, MAE, RMSE, and catalog coverage calculated from a deterministic train/test split.
- JSON endpoints at `/api/movies`, `/api/recommend`, and `/api/stats`.
- Built-in sample catalog fallback so the application can launch without data files.

## Technology

Python 3.10+, Flask, pandas, NumPy, scikit-learn, HTML, CSS, JavaScript, Bootstrap Icons, and Chart.js. Charts load Chart.js from a CDN; all core routes and recommendations work without it.

## Dataset

The app is currently configured with the MovieLens `ml-latest-small` files supplied by the user. `data/movies.csv` and `data/ratings.csv` power the recommender; `links.csv` and `tags.csv` are included for possible extensions. MovieLens is provided by GroupLens Research. See [GroupLens MovieLens datasets](https://grouplens.org/datasets/movielens/) and the included source README for dataset-specific terms and details.

The app cleans duplicate movie IDs and user/movie rating pairs, drops rows missing required fields, converts IDs and ratings to numeric values, and excludes ratings outside 0.5–5. If you need to restore the data, download and extract the latest-small archive, then place `movies.csv` and `ratings.csv` in `data/`. A small in-code sample dataset is used if either is absent; the UI labels that source explicitly. CSVs are ignored by Git.

## Architecture

```text
app.py                   Flask routes and shared application setup
src/data_loader.py       CSV loading, cleanup, sample fallback
src/recommender.py       Cached genre features, search, recommendation logic
src/analytics.py         Dataset summaries for the dashboard/API
src/evaluation.py        Train/test recommendation and rating metrics
templates/               Jinja pages
static/css/style.css     Responsive MovieMind design system
static/js/app.js         Small progressive-enhancement script
data/                    MovieLens CSV inputs
```

## Recommendation algorithm

1. Movie genre strings become TF-IDF feature vectors once when the app starts.
2. Cosine similarity compares a selected movie vector with the rest of the catalog. The selected movie is removed, and the closest genre profiles are returned with an explanation and similarity score.
3. Genre recommendations rank titles by overlap with selected genres, then use a Bayesian-smoothed rating score to avoid letting a single rating dominate.
4. Profile recommendations infer positive genre preferences from the user's session ratings. Ratings stay in the browser session and are not written to a database.

This is a content-based hybrid with community-rating popularity signals. It does not train a full collaborative-filtering model; the MovieLens rating interactions are used for preferences, popularity, analytics, and evaluation. Features are prepared at startup rather than recomputing the feature matrix on every request.

## Evaluation metrics

Evaluation uses an 80/20 random rating holdout (`random_state=42`) and a relevance threshold of 4 stars. The recommendation metric uses a user's training-set genre preferences (or a popularity fallback) to form a top-10 list.

- **Precision@10:** relevant held-out movies found in the list divided by the number of users × 10.
- **Recall@10:** relevant held-out movies found divided by all relevant held-out movies.
- **MAE:** mean absolute difference between held-out ratings and training-only movie mean estimates.
- **RMSE:** square root of mean squared rating error; larger misses count more.
- **Coverage:** unique movies recommended to evaluated users divided by the catalog size.

Metrics are computed from the loaded dataset; the app does not display hardcoded scores. Because this is a simple baseline and implicit recommendation quality depends on the relevance threshold and split, use the methodology note alongside the values in a project presentation.

## Installation and VS Code setup

Open this folder in VS Code. In the integrated terminal, create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for this terminal, use `venv\Scripts\activate.bat` from Command Prompt, or run the interpreter directly as `venv\Scripts\python.exe`.

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Open **http://127.0.0.1:5000**. The app uses the included MovieLens CSVs automatically. For a persistent Flask session key outside a local demo, set the `SECRET_KEY` environment variable before running.

## Pages and example use

- `/` — start page and popular movies.
- `/movies` — search, filter, and sort the catalog. Try `?q=matrix`.
- `/movie/<id>` — movie profile and similar titles.
- `/recommend` — choose a liked movie or preferred genres; review match reasons.
- `/profile` — rate movies and see session-based personalized suggestions.
- `/dashboard` — inspect dataset statistics and charts.
- `/evaluation` — inspect calculated ranking and rating metrics.

The same app exposes JSON at `/api/movies?q=matrix`, `/api/recommend` (POST JSON such as `{"movie_id": 1, "limit": 10}` or `{"genres": ["Comedy"]}`), and `/api/stats`.

## Limitations and future scope

MovieLens genres are broad metadata, not full descriptions or current availability. There are no licensed poster images; the interface generates typographic poster cards locally. The profile is browser-session-only, collaborative filtering is not implemented, and the holdout evaluation is a simple baseline. Possible extensions include persisted user profiles, user-user/item-item collaborative filtering, temporal splits, richer text features, and licensed poster metadata.

## Project pipeline

MovieLens CSVs → data loading/cleaning → interaction and genre analysis → TF-IDF feature engineering → cosine similarity and rating-aware ranking → web recommendations → offline evaluation.
