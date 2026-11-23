"""Train the symptom-based disease predictor.
Dataset: Kaggle "Disease Symptom Prediction" -> data/dataset.csv
(columns: Disease, Symptom_1 ... Symptom_17). Optional: data/Symptom-severity.csv
"""
import pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("data/dataset.csv")
sym_cols = [c for c in df.columns if c.startswith("Symptom")]
for c in sym_cols:
    df[c] = df[c].astype(str).str.strip().str.replace(" ", "_").replace("nan", np.nan)

symptoms = sorted({s for c in sym_cols for s in df[c].dropna().unique()})
idx = {s: i for i, s in enumerate(symptoms)}

X = np.zeros((len(df), len(symptoms)), dtype=np.uint8)
for r, row in enumerate(df[sym_cols].values):
    for s in row:
        if isinstance(s, str):
            X[r, idx[s]] = 1
y = df["Disease"].str.strip().values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
print(classification_report(yte, clf.predict(Xte)))

joblib.dump({"model": clf, "symptoms": symptoms}, "models/symptom_model.joblib")
print("Saved models/symptom_model.joblib")
