import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(movies, ratings, engine, k=10):
    """Deterministic per-user holdout; relevant means held-out rating >= 4."""
    if len(ratings)<20:
        return {'precision':0,'recall':0,'mae':None,'rmse':None,'coverage':0,'k':k,'users_evaluated':0,'note':'At least 20 ratings are needed for a meaningful holdout. Add MovieLens ratings.'}
    train, test=train_test_split(ratings, test_size=.2, random_state=42)
    # Fit rating/popularity statistics on training data only to avoid test leakage.
    from src.recommender import Recommender
    train_engine = Recommender(movies, train)
    relevant=[]; hit=0; total_relevant=0; covered=set(); actual=[]; predicted=[]
    # Content recommendations are evaluated against held-out liked movies for the user's strongest genres.
    train_groups=train.groupby('userId')
    for user, group in test.groupby('userId'):
        known=train_groups.get_group(user) if user in train_groups.groups else train.iloc[0:0]
        liked=known[known.rating>=4]
        pref={}
        for mid in liked.movieId:
            item=engine.movie(int(mid))
            if item:
                for g in item['genres'].split(', '): pref[g]=pref.get(g,0)+1
        genres=sorted(pref,key=pref.get,reverse=True)[:3]
        recs=train_engine.for_genres(genres,k,exclude=known.movieId.tolist()) if genres else train_engine.popular(k)
        rec_ids={r['movieId'] for r in recs}; covered.update(rec_ids)
        # Four-star test ratings count as relevant for precision and recall.
        truth=set(group.loc[group.rating>=4,'movieId'].astype(int)); hit+=len(truth&rec_ids); total_relevant+=len(truth)
        actual.extend(group.rating.astype(float).tolist()); predicted.extend([float(train_engine.movie(int(mid))['rating']) if train_engine.movie(int(mid)) else train_engine.global_mean for mid in group.movieId])
    n_users=max(1,test.userId.nunique())
    return {'precision':round(hit/(n_users*k),4),'recall':round(hit/total_relevant,4) if total_relevant else 0,'mae':round(float(mean_absolute_error(actual,predicted)),3),'rmse':round(float(np.sqrt(mean_squared_error(actual,predicted))),3),'coverage':round(len(covered)/max(1,movies.movieId.nunique()),4),'k':k,'users_evaluated':int(n_users),'note':'Fixed 80/20 rating holdout (random_state=42); relevance threshold is rating ≥ 4. Popularity-based rating estimates are used for MAE/RMSE.'}
