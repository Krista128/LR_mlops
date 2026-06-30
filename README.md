# Employee Attrition Prediction MLOps Pipeline

End-to-end MLOps project for employee attrition prediction demonstrating the complete lifecycle of a machine learning system, from data preparation and model training to deployment, monitoring and drift detection.

The project includes:

* reproducible data preparation;
* model registration tracking with MLflow;
* FastAPI prediction service;
* PostgreSQL storage for inference history;
* automated drift detection and model retrain;
* Prometheus and Grafana for metrics collection and dashboards;
* Kubernetes deployment (Minikube);
* CI/CD with GitHub Actions.

---

# Project Organization

```
.
├── .github/
│   └── workflows/
│       └── ci.yml              <- Continuous Integration pipeline
│
├── data/
│   ├── raw/                    <- Original dataset
│   ├── interim/                <- Temporary transformed data
│   ├── processed/              <- Final datasets used for training
│   └── external/               <- Downloaded external datasets
│
├── docs/                       <- Project documentation
│
├── k8s/                        <- Kubernetes manifests
│   ├── predictor.yaml          <- Prediction service deployment
│   ├── postgres.yaml           <- PostgreSQL deployment
│   ├── mlflow.yaml             <- MLflow server
│   ├── prometheus.yaml         <- Prometheus configuration
│   ├── grafana.yaml            <- Grafana deployment
│   └── ui.yaml                 <- Monitoring web interface
│
├── notebooks/                  <- Jupyter notebooks for experiments,
│                                  EDA and model development
│
├── reports/
│   └── figures/                <- Generated plots and figures
│
├── tests/                      <- Unit tests
│
├── src/
│   ├── __init__.py
│   ├── config.py               <- Global project configuration
│   ├── utils.py                <- Shared utility functions
│   │
│   ├── app/
│   │   ├── predictor_app.py    <- FastAPI prediction service
│   │   ├── drift_detector_app.py <- Drift detection and retrain
│   │   │                            FastAPI service
│   │   ├── ui_app.py           <- Web UI backend
│   │   └── db_queries.py       <- PostgreSQL interaction
│   │
│   ├── data/
│   │   ├── download_dataset.py <- Dataset downloading
│   │   ├── make_dataset.py     <- Data preparation
│   │   └── make_split.py       <- Train/validation/test split
│   │
│   ├── features/
│   │   └── build_features.py   <- Feature engineering
│   │
│   ├── models/
│   │   ├── train_model.py      <- Model training and MLflow logging
│   │   └── predict_model.py    <- Model loading and inference
│   │
│   └── visualization/
│       └── visualize.py        <- Visualization utilities
│
├── Dockerfile.predictor        <- Predictor container
├── Dockerfile.ui               <- UI container
├── Dockerfile.drift            <- Drift-detector container
├── requirements.txt            <- Predictor dependencies
├── requirements-dev.txt        <- develop dependencies
├── requirements-ui.txt         <- UI dependencies
├── pyproject.toml              <- Ruff and project configuration
└── README.md
```

---

# System Architecture

The application consists of several independent services: UI, prediction apps (1+), database, mlflow server and drift-retrain app.

### Predictor app

FastAPI application providing prediction endpoints.

Responsibilities:

* loads the production model from MLflow (loading during operation is allowed);
* performs inference;
* stores prediction history into PostgreSQL;
* exposes Prometheus metrics;
* provides OpenAPI documentation.

### Drift Detector

Background service periodically analysing accumulated inference history.

Calculates:

* Data Drift;
* Target Drift;
* Concept Drift.

When enough labelled data becomes available, the corresponding statistical tests are executed automatically.

### MLflow

MLflow is responsible for:

* experiment tracking;
* models storage and registry

### PostgreSQL

Stores:
* prediction requests;
* prediction results;
* delayed ground-truth labels;
* drift history and service metadata required for drift detection.

### Prometheus

Collects runtime metrics from Predictor and Drift Detector.

### Grafana

Visualizes collected metrics and monitoring dashboards.

---

# Running locally

minimal scheme:
* clone repository;
* create python environment;
* pip install requirements.txt;
* run mlflow, train or put some models in it;
* make shure model has tag "production: True";
* make shure urls in srv/config.py match your ones;
* run src/predictor_app.py;
* run in src\ui: uvicorn ui_app:app --host 0.0.0.0 --port 8080;

---

# Kubernetes deployment

Deploy all services

create secret.yaml:
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secrets
  namespace: empl-attr
type: Opaque
stringData:
  POSTGRES_PASSWORD: <your_password>

ruun:
kubectl apply -f namaspace.yaml
kubectl apply -f secret.yaml
kubectl apply -f k8s/

Database must be first time filled manual with trained data.
Tables defenition can be found in docs\sql_tables.txt

Services can be accessed using

```bash
kubectl port-forward
```

# Monitoring

Prometheus continuously collects application metrics.

The project exposes metrics including:

* prediction latency;
* prediction request rate;
* prediction count;
* application health;
* drift statistics.

Grafana dashboards visualize these metrics in real time.

---

# CI/CD

GitHub Actions automatically performs:

* Ruff linting;
* unit tests;
* Docker image build;
* deployment after merge into the main branch.

---

# Technologies

* Python
* FastAPI
* MLflow
* PostgreSQL
* SQLAlchemy
* Prometheus
* Grafana
* Docker
* Kubernetes (Minikube)
* GitHub Actions
* DVC

---

Project initially follows the Cookiecutter Data Science project template and extends it into a complete production-oriented MLOps pipeline.
