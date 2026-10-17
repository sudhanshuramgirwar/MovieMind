"""MovieMind: an explainable MovieLens recommendation demo."""
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
from src.data_loader import load_dataset
from src.analytics import build_analytics
from src.evaluation import evaluate_model
from src.recommender import Recommender

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(32)
# Load the data and build genre features once so page requests stay quick.
DATA = load_dataset()
ENGINE = Recommender(DATA['movies'], DATA['ratings'])
EVALUATION = None

@app.context_processor
def shared_context():
    return {'nav_movies': len(DATA['movies']), 'nav_ratings': len(DATA['ratings']), 'data_source': DATA['source']}

@app.route('/')
def home():
    featured = ENGINE.popular(8)
    hero_movie = featured[0] if featured else None
    hero_related = ENGINE.similar(hero_movie['movieId'], 2) if hero_movie else []
    return render_template('index.html', featured=featured, hero_movie=hero_movie, hero_related=hero_related)

@app.route('/movies')
def movies():
    query = request.args.get('q', '').strip()
    genre = request.args.get('genre', '')
    sort = request.args.get('sort', 'rating')
    found = ENGINE.search(query=query, genre=genre, sort=sort)
    return render_template('movies.html', movies=found[:120], genres=ENGINE.genres, query=query, selected_genre=genre, sort=sort)

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    movie = ENGINE.movie(movie_id)
    if movie is None:
        flash('We could not find that movie.', 'warning')
        return redirect(url_for('movies'))
    return render_template('movie.html', movie=movie, recommendations=ENGINE.similar(movie_id, 6))

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    chosen = request.values.get('movie_id', type=int)
    results, seed, selected_genres = [], None, []
    # Movie detail pages can open this route with a title already selected.
    if request.method == 'GET' and chosen and ENGINE.movie(chosen):
        seed = ENGINE.movie(chosen)
        results = ENGINE.similar(chosen, 10)
    if request.method == 'POST':
        if chosen and ENGINE.movie(chosen):
            seed = ENGINE.movie(chosen)
            results = ENGINE.similar(chosen, 10)
        else:
            selected_genres = request.form.getlist('genres')
            results = ENGINE.for_genres(selected_genres, 10)
            if not results: flash('Choose at least one genre to get recommendations.', 'info')
    return render_template('recommendations.html', genres=ENGINE.genres, results=results, seed=seed, chosen=chosen, selected_genres=selected_genres)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'POST':
        movie_id = request.form.get('movie_id', type=int)
        rating = request.form.get('rating', type=float)
        if not movie_id or not ENGINE.movie(movie_id) or rating is None or not 0.5 <= rating <= 5:
            flash('Choose a valid movie and a rating from 0.5 to 5.', 'warning')
        else:
            ratings = session.get('ratings', {})
            # Keep these demo ratings in the browser session instead of a user table.
            ratings[str(movie_id)] = rating
            session['ratings'] = ratings
            flash('Your rating was saved for this browser session.', 'success')
        return redirect(url_for('profile'))
    rated = []
    for key, value in session.get('ratings', {}).items():
        item = ENGINE.movie(int(key))
        if item:
            rated.append({**item, 'user_rating': value})
    preferences = ENGINE.user_preferences(session.get('ratings', {}))
    personal = ENGINE.for_genres(preferences, 8, exclude=[int(x) for x in session.get('ratings', {})]) if preferences else []
    return render_template('profile.html', movies=ENGINE.search(sort='title')[:250], rated=rated, preferences=preferences, results=personal)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', stats=build_analytics(DATA['movies'], DATA['ratings']))

@app.route('/evaluation')
def evaluation():
    global EVALUATION
    # The holdout metrics are expensive, so reuse them after the first visit.
    if EVALUATION is None:
        EVALUATION = evaluate_model(DATA['movies'], DATA['ratings'], ENGINE)
    return render_template('evaluation.html', metrics=EVALUATION)

@app.get('/api/movies')
def api_movies():
    return jsonify(ENGINE.search(request.args.get('q',''), request.args.get('genre',''), request.args.get('sort','rating'))[:50])

@app.get('/api/stats')
def api_stats():
    return jsonify(build_analytics(DATA['movies'], DATA['ratings']))

@app.post('/api/recommend')
def api_recommend():
    payload = request.get_json(silent=True) or {}
    movie_id = payload.get('movie_id')
    if movie_id is not None:
        try:
            movie_id = int(movie_id)
            limit = min(max(int(payload.get('limit', 10)), 1), 20)
        except (TypeError, ValueError):
            return jsonify({'error': 'movie_id and limit must be whole numbers'}), 400
        if not ENGINE.movie(movie_id):
            return jsonify({'error': 'Movie not found'}), 404
        return jsonify(ENGINE.similar(movie_id, limit))
    return jsonify(ENGINE.for_genres(payload.get('genres',[]), 10))

if __name__ == '__main__':
    app.run(debug=True)
