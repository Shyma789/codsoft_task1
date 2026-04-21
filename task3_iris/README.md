# Task 3 – Iris Flower Classification
## CodSoft Data Science Internship

### Dataset
Download from: https://www.kaggle.com/datasets/arshid/iris-flower-dataset
Save the CSV as `IRIS.csv` in this folder.

### Setup & Run
```bash
pip install -r requirements.txt
python iris_classification.py
```

### What the script does
1. Loads and explores the Iris dataset (EDA with scatter plots, box plots, correlation heatmap)
2. Preprocesses: label encoding + standard scaling
3. Trains and compares 4 classifiers: KNN, Decision Tree, Random Forest, SVM
4. Evaluates with accuracy, 5-fold CV, and classification report
5. Demonstrates predictions on 3 custom flower measurements
6. Saves `iris_eda.png` and `iris_results.png`
