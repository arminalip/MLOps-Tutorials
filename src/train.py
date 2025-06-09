# src/train.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import wandb
import sys

mlflow.set_experiment("Iris-Classification")
wandb.init(project="iris-tutorial")

# Load Data
df = pd.read_csv('data/iris.csv')
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Logging
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 3)
mlflow.log_metric("accuracy", acc)

wandb.log({"accuracy": acc})

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(clf, f)

mlflow.log_artifact("models/model.pkl")
artifact = wandb.Artifact("model", type="model")
artifact.add_file("models/model.pkl")
wandb.log_artifact(artifact)

wandb.finish()
