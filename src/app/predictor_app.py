from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
import os
import time
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from src.models.predict_model import predictorClass
from src.app.db_querriees import db_connector
from src.config import MLFLOW_TRACKING_URI, PIPELINE_SCHEMA_VERSION
import mlflow
import requests
import uvicorn
import click


class PredictRequest(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EmployeeCount: int
    EmployeeNumber: int
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    Over18: str
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StandardHours: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int


class PredictResponse(BaseModel):
    Attrition: int


class test_model:
    def __init__(self):
        pass

    def predict(self, x: any = None):
        return {"Attrition": -100}


@click.command()
def main():
    mlflow_availibility = False
    attempts = 0
    while not mlflow_availibility and attempts < 12:
        try:
            requests.get(MLFLOW_TRACKING_URI, timeout=2.0)
            mlflow_availibility = True
        except requests.RequestException:
            mlflow_availibility = False
        time.sleep(10)
        attempts += 1

    if mlflow_availibility:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        runs = mlflow.search_runs(
            search_all_experiments=True,
            filter_string=(
                "tags.production = 'True' AND "
                "tags.model_saved = 'True' AND "
                f"tags.pipeline_schema_version = '{PIPELINE_SCHEMA_VERSION}'"
            ),
            order_by=["metrics.roc_auc DESC"],
            max_results=1,
        )

        if len(runs) == 0:
            print("There are no suitable model on MLflow server")
            print("App will start in test mode, without model")
            status = "alive, in testing mode"
            predictor = test_model()
            run_id = "-100"
        else:
            run_id = runs.iloc[0]["run_id"]
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            predictor = predictorClass(model=model)
            status = "alive"
    else:
        print("MLflow server unavailible, cant load model")
        print(f"({attempts} attempts to connect)")
        print("App will start in test mode, without model")
        status = "alive, in testing mode"
        predictor = test_model()
        run_id = "-100"

    host = os.getenv("DB_HOST")
    db_name = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db_port = os.getenv("DB_PORT")
    db_url = (
        f"postgresql+psycopg2://{user}:{password}@{host}:{db_port}/{db_name}"
    )
    postgres_connector = db_connector(db_url)

    db_availibility = False
    attempts = 0
    while not db_availibility and attempts < 12:
        db_availibility = postgres_connector.connection_alive()
        time.sleep(10)
        attempts += 1

    if not db_availibility:
        print(f"Database unavailible ({attempts} attempts to connect)")
        print("App will start without saving predict results")
        status = status + ", no connection to database"

    predict_requests = Counter(
        "predict_requests_total", "Total number of predict requests"
    )

    predict_errors = Counter(
        "predict_errors_total", "Total number of failed predict requests"
    )

    predict_latency = Histogram(
        "predict_latency_seconds", "Predict request latency in seconds"
    )

    app = FastAPI(title="attr_prediction_api")

    @app.get("/health")
    def health():
        return {"status": status}

    @app.post("/predict", response_model=PredictResponse)
    def make_prediction(request: PredictRequest):
        predict_requests.inc()
        start = time.time()
        try:
            row = request.model_dump()
            prediction = predictor.predict(row)
            row["model_train_run_id"] = run_id
            row["Attrition"] = prediction["Attrition"]
            if db_availibility:
                postgres_connector.insert_history(row)
            return prediction
        except Exception as e:
            predict_errors.inc()
            print(e)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            predict_latency.observe(time.time() - start)

    @app.get("/metrics")
    def metrics():
        return Response(
            content=generate_latest(), media_type=CONTENT_TYPE_LATEST
        )

    @app.get("/pull_new_model")
    def pull_new_model():
        nonlocal predictor, status
        mlflow_availibility = False
        attempts = 0
        while not mlflow_availibility and attempts < 12:
            try:
                requests.get(MLFLOW_TRACKING_URI, timeout=10.0)
                mlflow_availibility = True
            except requests.RequestException:
                mlflow_availibility = False
            time.sleep(10)
            attempts += 1

        if mlflow_availibility:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            runs = mlflow.search_runs(
                search_all_experiments=True,
                filter_string=(
                    "tags.production = 'True' AND "
                    "tags.model_saved = 'True' AND "
                    "tags.pipeline_schema_version = "
                    f"'{PIPELINE_SCHEMA_VERSION}'"
                ),
                order_by=["metrics.roc_auc DESC"],
                max_results=1,
            )

            if len(runs) != 0:
                run_id = runs.iloc[0]["run_id"]
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.sklearn.load_model(model_uri)
                predictor = predictorClass(model=model)

                if ", no connection to database" in status:
                    status = "alive, no connection to database"
                else:
                    status = "alive"
            else:
                print("There are still no suitable model on MLflow server")
                print("App will use the old one")
        else:
            print(("MLflow server unavailible, cant load new model"))
            print(
                f"({attempts} attempts to connect). App will use the old one"
            )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
