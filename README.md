# Machine Learning + Quantum Machine Learning Lab

An interactive research dashboard that brings classical Machine Learning and Quantum Machine Learning into one project.

## Overview

This project combines a FastAPI-based interactive dashboard with classical ML models, model evaluation tools, real Qiskit quantum circuits, and a Variational Quantum Classifier workflow.

The goal is to provide a visual research environment where classical and quantum approaches can be explored, compared, and benchmarked in one place.

## Features

- Random Forest classification
- Neural Network classification with scikit-learn
- Live training-style metrics and visualizations
- Feature importance analysis
- Confusion matrices
- Precision, Recall and F1 evaluation
- ROC / AUC analysis
- Classical model comparison
- Qiskit quantum circuit simulation
- Quantum state visualization
- Variational Quantum Classifier (VQC)
- QML confusion matrix
- VQC training convergence visualization
- Classical vs Quantum benchmark comparison
- QML leaderboard and research dashboard

## Tech Stack

- Python
- FastAPI
- Uvicorn
- scikit-learn
- NumPy
- pandas
- Plotly
- Qiskit
- Qiskit Aer
- Qiskit Machine Learning
- Qiskit Algorithms

## Project Structure

```text
machine-learning-quantum-lab/
├── app.py
├── qml_test.py
├── qml_result.json
├── lab2.ipynb
├── build_gif.py
└── README.md
```

## Run Locally

Install the required Python packages, then start the dashboard:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

## QML Workflow

The dashboard can load validated QML results from `qml_result.json`.

To refresh the Quantum Machine Learning experiment locally, run:

```bash
python qml_test.py
```

The result can then be used by the dashboard for QML metrics and convergence visualizations.

## Research Focus

This lab is designed around practical experimentation with:

- Classical vs Quantum model benchmarking
- Variational quantum circuits
- Quantum feature maps
- Hybrid AI / Quantum workflows
- Model evaluation and visualization

## Author

**Tamás Németh**

AI & Quantum Research Lab
