from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'

def _demo_data():
    rows = [
        (1,'The Matrix (1999)','Action|Sci-Fi'), (2,'Inception (2010)','Action|Adventure|Sci-Fi|Thriller'),
        (3,'Interstellar (2014)','Adventure|Drama|Sci-Fi'), (4,'The Dark Knight (2008)','Action|Crime|Drama|Thriller'),
        (5,'Arrival (2016)','Drama|Mystery|Sci-Fi'), (6,'Blade Runner 2049 (2017)','Drama|Mystery|Sci-Fi'),
        (7,'The Shawshank Redemption (1994)','Drama'), (8,'Pulp Fiction (1994)','Crime|Drama|Thriller'),
        (9,'Spirited Away (2001)','Adventure|Animation|Fantasy'), (10,'The Grand Budapest Hotel (2014)','Comedy|Drama'),
        (11,'Mad Max: Fury Road (2015)','Action|Adventure|Sci-Fi'), (12,'Eternal Sunshine of the Spotless Mind (2004)','Drama|Romance|Sci-Fi'),
        (13,'Parasite (2019)','Comedy|Drama|Thriller'), (14,'Whiplash (2014)','Drama'),
        (15,'The Lord of the Rings: The Fellowship of the Ring (2001)','Adventure|Fantasy'),
        (16,'Get Out (2017)','Horror|Mystery|Thriller'), (17,'Coco (2017)','Adventure|Animation|Family'),
        (18,'The Social Network (2010)','Drama'), (19,'Jurassic Park (1993)','Action|Adventure|Sci-Fi'),
        (20,'La La Land (2016)','Comedy|Drama|Musical|Romance')]
    movies = pd.DataFrame(rows, columns=['movieId','title','genres'])
    ratings = pd.DataFrame([(u, mid, float(3 + ((u*mid)%5)*.5), 0) for u in range(1,81) for mid,_,_ in rows if (u+mid)%4 != 0], columns=['userId','movieId','rating','timestamp'])
    return movies, ratings

def load_dataset():
    """Load MovieLens CSVs; fall back to clearly labeled built-in demo data."""
    movie_path, rating_path = DATA_DIR/'movies.csv', DATA_DIR/'ratings.csv'
    if movie_path.exists() and rating_path.exists():
        movies, ratings = pd.read_csv(movie_path), pd.read_csv(rating_path)
        source = 'MovieLens files'
    else:
        movies, ratings = _demo_data()
        source = 'built-in sample (add MovieLens CSVs for full dataset)'
    # Drop duplicate or incomplete rows before IDs and ratings enter the model.
    movies = movies.drop_duplicates('movieId').dropna(subset=['movieId','title']).copy()
    ratings = ratings.drop_duplicates(['userId','movieId']).dropna(subset=['userId','movieId','rating']).copy()
    ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
    ratings = ratings[ratings.rating.between(.5,5)].copy()
    movies['movieId'] = movies.movieId.astype(int); ratings['movieId'] = ratings.movieId.astype(int)
    return {'movies':movies, 'ratings':ratings, 'source':source}
