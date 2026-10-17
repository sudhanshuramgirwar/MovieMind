# MovieLens data

The application currently uses the user's MovieLens `ml-latest-small` CSV files in this folder: `movies.csv`, `ratings.csv`, `links.csv`, and `tags.csv`. The recommendation and analytics pipeline uses `movies.csv` and `ratings.csv`; links and tags are retained for optional future features.

To set up another copy, download MovieLens latest-small from [GroupLens](https://grouplens.org/datasets/movielens/), extract it, and copy `movies.csv` and `ratings.csv` into this folder. The app falls back to a small, clearly labeled built-in example catalog if those two files are missing. Dataset files are excluded from Git.
