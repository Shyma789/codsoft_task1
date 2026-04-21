"""
TASK 2 – MOVIE RATING PREDICTION WITH PYTHON
CodSoft Data Science Internship
Dataset: https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import warnings
import os

warnings.filterwarnings('ignore')

# Handle file paths for VS Code "Run" button
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "IMDb Movies India.csv")

print("=" * 60)
print("  MOVIE RATING PREDICTION - STARTING")
print("=" * 60)

# Load Data
try:
    df = pd.read_csv(file_path, encoding='latin-1')
except FileNotFoundError:
    print(f"Error: Could not find 'IMDb Movies India.csv' in {script_dir}")
    exit()

# 2. CLEANING
def clean_dataset(data):
    df_c = data.copy()
    df_c['Rating'] = pd.to_numeric(df_c['Rating'], errors='coerce')
    df_c.dropna(subset=['Rating'], inplace=True)
    df_c['Year'] = df_c['Year'].astype(str).str.extract(r'(\d{4})').astype(float)
    df_c['Duration'] = df_c['Duration'].astype(str).str.extract(r'(\d+)').astype(float)
    df_c['Votes'] = (df_c['Votes'].astype(str)
                     .str.replace(',', '', regex=False)
                     .str.extract(r'(\d+)')[0]
                     .astype(float))
    df_c['Year'] = df_c['Year'].fillna(df_c['Year'].median())
    df_c['Duration'] = df_c['Duration'].fillna(df_c['Duration'].median())
    df_c['Votes'] = df_c['Votes'].fillna(0)
    return df_c

df = clean_dataset(df)

# 3. FEATURE ENGINEERING
def frequency_encode(series):
    freq = series.value_counts(normalize=True)
    return series.map(freq).fillna(0)

# FIXED: Proper string splitting for primary genre
df['Genre_primary'] = df['Genre'].astype(str).str.split(',').str[0].str.strip()
df['Log_Votes'] = np.log1p(df['Votes'])
df['Director_freq'] = frequency_encode(df['Director'].astype(str))
df['Actor_avg_freq'] = (frequency_encode(df['Actor 1'].astype(str)) + 
                        frequency_encode(df['Actor 2'].astype(str))) / 2
df['Movie_age'] = 2026 - df['Year']

# 4. MODEL TRAINING
feature_cols = ['Log_Votes', 'Duration', 'Movie_age', 'Director_freq', 'Actor_avg_freq']
X = df[feature_cols]
y = df['Rating']

imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. DASHBOARD
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Rating Density Analysis", "Top Genres by Average Rating", 
                    "Model: Actual vs Predicted", "Impact of Popularity (Log Votes)"),
    vertical_spacing=0.2,
    horizontal_spacing=0.12
)

fig.add_trace(go.Violin(y=df['Rating'], name="Rating", box_visible=True, line_color='#FFA500'), row=1, col=1)

genre_res = df.groupby('Genre_primary')['Rating'].mean().sort_values(ascending=False).head(10)
fig.add_trace(go.Bar(x=genre_res.index, y=genre_res.values, 
                     marker=dict(color=genre_res.values, colorscale='Viridis')), row=1, col=2)

fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', marker=dict(color='#2980b9', opacity=0.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=[2,10], y=[2,10], mode='lines', line=dict(color='red', dash='dash')), row=2, col=1)

fig.add_trace(go.Scattergl(x=df['Log_Votes'], y=df['Rating'], mode='markers', 
                           marker=dict(color=df['Rating'], colorscale='Plasma', size=4)), row=2, col=2)

fig.update_layout(title_text="MOVIE RATING PREDICTION DASHBOARD", template="plotly_dark", height=900, showlegend=False)

# Labels with angles to fix the "merged words" issue
fig.update_xaxes(title_text="Rating Score", row=1, col=1)
fig.update_xaxes(title_text="Genre", tickangle=45, row=1, col=2)
fig.update_xaxes(title_text="Actual Rating", row=2, col=1)
fig.update_xaxes(title_text="Log(Votes)", row=2, col=2)

print("\n Success! Opening dashboard in your browser...")
fig.show()

print("\n" + "=" * 60)
print("  TASK 2 COMPLETE")
print("=" * 60)