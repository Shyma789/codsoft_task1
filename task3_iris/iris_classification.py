"""
TASK 3 – IRIS FLOWER CLASSIFICATION
CodSoft Data Science Internship
Dataset: https://www.kaggle.com/datasets/arshid/iris-flower-dataset
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import webbrowser

# 1. ROBUST DATA LOADING
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "IRIS.csv")

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading IRIS.csv: {e}")
    exit()

# 2. PREPROCESSING & MODEL ENGINE
le = LabelEncoder()
df['species_id'] = le.fit_transform(df['species'])
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species_id']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# 3. CONSTRUCTING THE UNIFIED DASHBOARD
fig = make_subplots(
    rows=2, cols=2,
    row_heights=[0.5, 0.5],
    vertical_spacing=0.12,
    specs=[[{"type": "heatmap"}, {"type": "bar"}],
           [{"type": "scatter"}, {"type": "table"}]],
    subplot_titles=("<b>AI Confusion Matrix (Accuracy)</b>", "<b>Feature Importance Analysis</b>", 
                    "<b>Species Clustering (Petal Stats)</b>", "<b>Performance Metrics Summary</b>")
)

# --- CHART 1: ACCURACY HEATMAP ---
fig.add_trace(go.Heatmap(
    z=cm, x=list(le.classes_), y=list(le.classes_),
    colorscale='YlGnBu', text=cm, texttemplate="%{text}",
    showscale=False), row=1, col=1)

# --- CHART 2: FEATURE IMPORTANCE ---
importances = model.feature_importances_
fig.add_trace(go.Bar(
    x=list(X.columns), y=importances,
    marker=dict(color=importances, colorscale='Viridis'),
    text=[f"{i:.2f}" for i in importances], textposition='auto'), row=1, col=2)

# --- CHART 3: SPECIES SCATTER ---
for species in le.classes_:
    mask = df['species'] == species
    fig.add_trace(go.Scatter(
        x=df.loc[mask, 'petal_length'], y=df.loc[mask, 'petal_width'],
        mode='markers', name=species, marker=dict(size=10)), row=2, col=1)

# --- CHART 4: PERFORMANCE TABLE ---
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
metrics_df = pd.DataFrame(report).transpose().reset_index().iloc[:3, :4]
fig.add_trace(go.Table(
    header=dict(values=["Class", "Precision", "Recall", "F1"], fill_color='#2c3e50', font=dict(color='white')),
    cells=dict(values=[metrics_df['index'], metrics_df['precision'].round(2), 
                       metrics_df['recall'].round(2), metrics_df['f1-score'].round(2)],
               fill_color='#f8f9fa')), row=2, col=2)

# 4. DASHBOARD THEME & FINISHING
fig.update_layout(
    title=dict(text=f"IRIS FLOWER CLASSIFICATION SYSTEM | FINAL SCORE: {acc*100:.1f}%", 
               x=0.5, font=dict(size=22)),
    template="plotly_white", height=850, showlegend=True
)

# 5. FORCE EXPORT & OPEN
output_name = "Task3_Final_Dashboard.html"
output_path = os.path.join(script_dir, output_name)
fig.write_html(output_path)

print(f"\n✅ SUCCESS! Accuracy: {acc*100:.2f}%")
print(f"✅ Dashboard saved at: {output_path}")

# This will force the browser to open the saved file immediately
webbrowser.open('file://' + os.path.realpath(output_path))