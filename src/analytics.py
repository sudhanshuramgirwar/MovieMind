def build_analytics(movies, ratings):
    import pandas as pd
    merged=ratings[['rating','movieId']].merge(movies[['movieId','genres']],on='movieId',how='left')
    dist=ratings.rating.value_counts().sort_index()
    genre_counts={}
    for genres in movies.genres.fillna(''):
        for genre in genres.split('|'):
            if genre and genre != '(no genres listed)': genre_counts[genre]=genre_counts.get(genre,0)+1
    top=ratings.groupby('movieId').agg(mean=('rating','mean'),count=('rating','size')).sort_values('count',ascending=False).head(8).reset_index().merge(movies[['movieId','title']],on='movieId',how='left')
    # Split multi-genre movies into one row per genre before averaging ratings.
    genre_rows=merged.assign(genres=merged.genres.fillna('').str.split('|')).explode('genres')
    genre_rows=genre_rows[~genre_rows.genres.isin(['','(no genres listed)'])]
    genre_means=genre_rows.groupby('genres').rating.mean().round(2)
    by_genre=genre_means.to_dict()
    return {'total_movies':int(movies.movieId.nunique()),'total_users':int(ratings.userId.nunique()),'total_ratings':int(len(ratings)),'average_rating':round(float(ratings.rating.mean()),2) if len(ratings) else 0,
      'ratings_distribution':{'labels':[str(x) for x in dist.index],'values':[int(x) for x in dist.values]},
      'top_genres':{'labels':list(dict(sorted(genre_counts.items(),key=lambda x:x[1],reverse=True)[:10]).keys()),'values':list(dict(sorted(genre_counts.items(),key=lambda x:x[1],reverse=True)[:10]).values())},
      'most_rated':{'labels':top.title.fillna('').tolist(),'values':top['count'].astype(int).tolist()},
      'genre_ratings':{'labels':list(by_genre),'values':list(by_genre.values())},
      'algorithm':'Genre TF-IDF + cosine similarity; user genre preferences and Bayesian popularity ranking'}
