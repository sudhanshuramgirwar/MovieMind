import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    """Cached genre TF-IDF similarity plus ratings-aware popularity ranking."""
    def __init__(self, movies, ratings):
        self.movies = movies.copy().reset_index(drop=True)
        self.ratings = ratings.copy()
        self.movies['genres'] = self.movies['genres'].fillna('(no genres listed)')
        self.genres = sorted({g for value in self.movies.genres for g in value.split('|') if g and g != '(no genres listed)'})
        self.index = {int(mid): i for i,mid in enumerate(self.movies.movieId)}
        self.genre_positions = {genre: np.flatnonzero(self.movies.genres.str.split('|').map(lambda values: genre in values).to_numpy()) for genre in self.genres}
        # Reuse these vectors for every similarity request instead of rebuilding them.
        self.vectorizer = TfidfVectorizer(token_pattern=r'[^|]+')
        self.features = self.vectorizer.fit_transform(self.movies.genres)
        self.stats = self.ratings.groupby('movieId').rating.agg(['mean','count'])
        self.global_mean = float(self.ratings.rating.mean()) if len(self.ratings) else 3.5
        # Bayesian shrinkage keeps a single high rating from dominating popular lists.
        self.movies['rating'] = self.movies.movieId.map(self.stats['mean']).fillna(self.global_mean)
        self.movies['rating_count'] = self.movies.movieId.map(self.stats['count']).fillna(0).astype(int)
        self.movies['score'] = (self.movies.rating_count*self.movies.rating + 10*self.global_mean)/(self.movies.rating_count+10)

    def _record(self, row, reason='', similarity=None):
        return {'movieId':int(row.movieId),'title':str(row.title),'genres':str(row.genres).replace('|',', '),'rating':round(float(row.rating),2),'rating_count':int(row.rating_count),'reason':reason,'similarity':round(float(similarity),3) if similarity is not None else None}
    def movie(self, movie_id):
        if movie_id not in self.index: return None
        row=self.movies.iloc[self.index[movie_id]]
        return self._record(row)
    def popular(self, limit=10):
        return [self._record(r, 'Popular with MovieLens users') for _,r in self.movies.sort_values(['score','rating_count'],ascending=False).head(limit).iterrows()]
    def similar(self, movie_id, limit=10):
        if movie_id not in self.index: return []
        # Compare the selected film with the cached genre vectors, then skip itself.
        idx=self.index[movie_id]; sims=cosine_similarity(self.features[idx],self.features).ravel()
        ranked=np.argsort(-sims)
        out=[]
        for i in ranked:
            mid=int(self.movies.iloc[i].movieId)
            if mid==movie_id: continue
            out.append(self._record(self.movies.iloc[i], 'Shares genre features with '+str(self.movies.iloc[idx].title), sims[i]))
            if len(out)>=limit: break
        return out
    def for_genres(self, genres, limit=10, exclude=None):
        chosen=set(genres or []); exclude=set(exclude or [])
        if not chosen: return []
        # More shared genres rank first; the smoothed rating score breaks ties.
        overlap_counts=np.zeros(len(self.movies),dtype=np.int16)
        for genre in chosen:
            if genre in self.genre_positions:
                overlap_counts[self.genre_positions[genre]] += 1
        valid=np.flatnonzero(overlap_counts>0)
        if exclude:
            excluded=np.isin(self.movies.movieId.to_numpy()[valid],list(exclude))
            valid=valid[~excluded]
        scores=self.movies.score.to_numpy()
        order=np.lexsort((scores[valid],overlap_counts[valid]))[::-1]
        results=[]
        for pos in valid[order[:limit]]:
            matched=sorted(chosen.intersection(str(self.movies.iloc[pos].genres).split('|')))
            results.append(self._record(self.movies.iloc[pos], 'Matches your preferred '+', '.join(matched)+' genres'))
        return results
    def search(self, query='', genre='', sort='rating'):
        df=self.movies
        if query: df=df[df.title.str.contains(re.escape(query),case=False,na=False)]
        if genre: df=df[df.genres.str.split('|').apply(lambda values: genre in values)]
        keys={'rating':('score',False),'popular':('rating_count',False),'title':('title',True)}
        key,ascending=keys.get(sort,keys['rating']); df=df.sort_values(key,ascending=ascending)
        return [self._record(r) for _,r in df.iterrows()]
    def user_preferences(self, ratings_by_movie):
        totals={}
        for mid,value in ratings_by_movie.items():
            movie=self.movie(int(mid))
            if not movie: continue
            # A rating above 2.5 adds weight; one below 2.5 reduces it.
            for genre in movie['genres'].split(', '): totals[genre]=totals.get(genre,0)+(float(value)-2.5)
        return sorted((g for g,v in totals.items() if v>0),key=lambda g:totals[g],reverse=True)[:5]
