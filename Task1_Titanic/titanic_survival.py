"""
TASK 1 - TITANIC SURVIVAL PREDICTION
CodSoft Data Science Internship
Dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("=" * 60)
print("  TITANIC SURVIVAL PREDICTION")
print("=" * 60)

df = pd.read_csv("Titanic-Dataset.csv")
print(f"\n Dataset shape: {df.shape}")
print(f"\n First 5 rows:\n{df.head()}")
print(f"\n Data types:\n{df.dtypes}")
print(f"\n Missing values:\n{df.isnull().sum()}")

# ─────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic – Exploratory Data Analysis", fontsize=16, fontweight='bold')

# Survival count
survival_counts = df['Survived'].value_counts()
axes[0, 0].bar(['Did Not Survive', 'Survived'], survival_counts.values,
               color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[0, 0].set_title('Overall Survival Count')
axes[0, 0].set_ylabel('Number of Passengers')
for i, v in enumerate(survival_counts.values):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Survival by gender
gender_survival = df.groupby(['Sex', 'Survived']).size().unstack()
gender_survival.plot(kind='bar', ax=axes[0, 1], color=['#e74c3c', '#2ecc71'],
                     edgecolor='black', rot=0)
axes[0, 1].set_title('Survival by Gender')
axes[0, 1].set_xlabel('Gender')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend(['Did Not Survive', 'Survived'])

# Survival by passenger class
pclass_survival = df.groupby(['Pclass', 'Survived']).size().unstack()
pclass_survival.plot(kind='bar', ax=axes[0, 2], color=['#e74c3c', '#2ecc71'],
                     edgecolor='black', rot=0)
axes[0, 2].set_title('Survival by Passenger Class')
axes[0, 2].set_xlabel('Class')
axes[0, 2].set_ylabel('Count')
axes[0, 2].legend(['Did Not Survive', 'Survived'])

# Age distribution
df['Age'].dropna().hist(bins=30, ax=axes[1, 0], color='#3498db', edgecolor='black')
axes[1, 0].set_title('Age Distribution')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Frequency')

# Fare distribution (log scale)
df['Fare'].hist(bins=40, ax=axes[1, 1], color='#9b59b6', edgecolor='black')
axes[1, 1].set_title('Fare Distribution')
axes[1, 1].set_xlabel('Fare')
axes[1, 1].set_ylabel('Frequency')

# Survival rate by age group
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior'])
age_survival = df.groupby('AgeGroup')['Survived'].mean() * 100
age_survival.plot(kind='bar', ax=axes[1, 2], color='#f39c12', edgecolor='black', rot=20)
axes[1, 2].set_title('Survival Rate (%) by Age Group')
axes[1, 2].set_ylabel('Survival Rate (%)')

plt.tight_layout()
plt.savefig("eda_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n EDA plot saved as 'eda_analysis.png'")

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────
def build_features(data):
    df_fe = data.copy()

    # Title extraction from Name
    df_fe['Title'] = df_fe['Name'].str.extract(r',\s*([^\.]+)\.')
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Capt': 'Rare',
        'Sir': 'Rare', 'Mme': 'Mrs'
    }
    df_fe['Title'] = df_fe['Title'].map(title_map).fillna('Rare')

    # Family size
    df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
    df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)

    # Cabin known indicator
    df_fe['HasCabin'] = df_fe['Cabin'].notna().astype(int)

    # Age * class interaction
    df_fe['Age_x_Pclass'] = df_fe['Age'].fillna(df_fe['Age'].median()) * df_fe['Pclass']

    # Fare per person
    df_fe['FarePerPerson'] = df_fe['Fare'] / df_fe['FamilySize']

    return df_fe


df = build_features(df)

# ─────────────────────────────────────────
# 4. PREPROCESSING
# ─────────────────────────────────────────
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'Title', 'FamilySize', 'IsAlone', 'HasCabin', 'Age_x_Pclass', 'FarePerPerson']

X = df[features].copy()
y = df['Survived']

# Encode categoricals
for col in ['Sex', 'Embarked', 'Title']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=features)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n Training samples : {X_train.shape[0]}")
print(f" Testing  samples : {X_test.shape[0]}")

# ─────────────────────────────────────────
# 5. MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                   random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150,
                                                       learning_rate=0.08,
                                                       max_depth=4, random_state=42),
}

results = {}
print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cv   = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy').mean()

    results[name] = {'model': model, 'accuracy': acc, 'auc': auc, 'cv_acc': cv,
                     'y_pred': y_pred, 'y_prob': y_prob}

    print(f"\n {name}")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   ROC-AUC   : {auc:.4f}")
    print(f"   CV Acc(5) : {cv:.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]['auc'])
best = results[best_name]
print(f"\n Best Model: {best_name}  (AUC = {best['auc']:.4f})")

# ─────────────────────────────────────────
# 6. DETAILED EVALUATION
# ─────────────────────────────────────────
print(f"\n Classification Report – {best_name}:\n")
print(classification_report(y_test, best['y_pred'],
                             target_names=['Did Not Survive', 'Survived']))

# ─────────────────────────────────────────
# 7. VISUALISATION – MODEL RESULTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Model Results – {best_name}", fontsize=15, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Did Not Survive', 'Survived'],
            yticklabels=['Did Not Survive', 'Survived'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1].set_title('ROC Curves')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=8)

# Feature importances (Random Forest)
rf_model = results["Random Forest"]['model']
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
importances.tail(10).plot(kind='barh', ax=axes[2], color='#3498db', edgecolor='black')
axes[2].set_title('Top 10 Feature Importances (RF)')
axes[2].set_xlabel('Importance')

plt.tight_layout()
plt.savefig("model_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n Model results plot saved as 'model_results.png'")

# ─────────────────────────────────────────
# 8. SAMPLE PREDICTIONS
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  SAMPLE PREDICTIONS (first 10 test passengers)")
print("=" * 60)
sample_pred = best['y_pred'][:10]
sample_true = y_test.values[:10]
sample_prob = best['y_prob'][:10]

for i, (pred, true, prob) in enumerate(zip(sample_pred, sample_true, sample_prob)):
    status = "✓" if pred == true else "✗"
    label  = "Survived" if pred == 1 else "Did Not Survive"
    print(f"  Passenger {i+1:2d}: Predicted={label:18s}  Prob={prob:.3f}  {status}")

print("\n" + "=" * 60)
print("  TASK 1 COMPLETE")
print("=" * 60)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Style
sns.set_style("whitegrid")
plt.figure(figsize=(14, 10))

# -------------------------------
# 1. Survival Count (Better Bar)
# -------------------------------
plt.subplot(2, 2, 1)
sns.countplot(x='Survived', data=df)
plt.title("Survival Count (0 = Not Survived, 1 = Survived)")

# -------------------------------
# 2. Survival by Gender
# -------------------------------
plt.subplot(2, 2, 2)
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")

# -------------------------------
# 3. Survival by Passenger Class
# -------------------------------
plt.subplot(2, 2, 3)
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")

# -------------------------------
# 4. Age Distribution (Smooth)
# -------------------------------
plt.subplot(2, 2, 4)
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True)
plt.title("Age Distribution by Survival")

# Adjust layout
plt.tight_layout()

# Show graphs
plt.show()