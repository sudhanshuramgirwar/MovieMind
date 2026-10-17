# MovieMind Google Colab Guide

## Open the notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook** and choose **MovieMind_Recommendation_System.ipynb** from this project folder, or upload the notebook to Google Drive and open it with Colab.
3. In Colab, select **Runtime → Run all**. Accept the prompt to run notebook code.

## Dataset

The notebook downloads the public MovieLens latest-small dataset from GroupLens into **/content** and extracts it automatically. It uses **movies.csv** and **ratings.csv**, matching the files used by the MovieMind application.

If the automatic download is unavailable:

1. Download MovieLens latest-small from [GroupLens](https://grouplens.org/datasets/movielens/).
2. Extract the ZIP on your computer.
3. In the notebook dataset cell, use the upload prompt to upload both **movies.csv** and **ratings.csv**.

The notebook does not need your local VS Code paths, Flask server, virtual environment, or model artifact. The algorithm is included as notebook code and follows **src/recommender.py**.

## What the notebook runs

- Cleans and summarizes MovieLens movie and rating tables.
- Charts rating counts, genre frequency, and most-rated titles.
- Uses MovieMind's TF-IDF genre vectors, cosine similarity, genre preference matching, and Bayesian-smoothed popularity ranking.
- Shows title-similarity and genre-based recommendations.
- Calculates Precision@10, Recall@10, catalog coverage, and rating-estimation MAE/RMSE from a deterministic 80/20 holdout.
- Handles unmatched titles and invalid movie IDs without terminating the notebook.

Metrics are computed from the dataset loaded by the notebook and are not fixed sample values. MAE/RMSE evaluate smoothed item-average rating estimates, while Precision/Recall/Coverage evaluate recommendation lists.

## Dependencies

Google Colab provides the notebook's standard dependencies: Python, NumPy, pandas, Matplotlib, and scikit-learn. A commented optional pip install command is included in the import cell. The existing project requirements file is for the Flask web app and does not need notebook-only changes.
