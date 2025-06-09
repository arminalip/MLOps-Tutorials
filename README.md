# 🧠 MLOps Iris Classifier Pipeline

This project demonstrates a production-grade MLOps workflow using:

- **Python Virtual Environments (`venv`)**
- **DVC** for data and model versioning
- **MLflow** for experiment tracking
- **Weights & Biases (W&B)** for artifact logging and visualization
- A simple **RandomForest** classifier trained on the Iris dataset

---

## 📁 Project Structure
```
├── dvc.yaml # DVC pipeline definition
├── dvc.lock # DVC pipeline lock file
├── requirements.txt # Python dependencies
├── src/
│ ├── download.py # Downloads iris dataset
│ └── train.py # Trains model and logs metrics/artifacts
├── data/
│ └── iris.csv.dvc # DVC-tracked raw dataset (removed)
├── models/
│ └── model.pkl.dvc # DVC-tracked trained model (removed)
└── .venv/ # Python virtual environment (ignored)
```
---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/arminalip/MLOps-Tutorials.git
cd MLOps-Tutorials
