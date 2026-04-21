# Task 2 – Movie Rating Prediction
## CodSoft Data Science Internship

### Dataset
Download from: https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies
Save the CSV as `IMDb Movies India.csv` in this folder.

### Setup & Run
```bash
pip install -r requirements.txt
python movie_rating.py
```

### What the script does
1. Loads and cleans IMDb India movies data (year, duration, votes parsing)
2. Engineers features: log-votes, director/actor frequency scores, genre flags, movie age
3. Trains Ridge Regression, Random Forest, Gradient Boosting regressors
4. Evaluates with RMSE, MAE, R²
5. Saves `eda_movies.png` and `movie_model_results.png`
