from fastapi import FastAPI
from pydantic import BaseModel
from src.models.predict_model import predictorClass
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


@click.command()
def main():
    class test_model:
        def __init__(self):
            pass

        def predict(self, x: any = None):
            return {"Attrition": -100}

    mlflow_availibility = True
    try:
        requests.get(MLFLOW_TRACKING_URI, timeout=10.0)
    except requests.RequestException:
        print("MLflow server unavailible, cant load model")
        print("App will start in test mode, without model")
        status = "alive, in testing mode"
        mlflow_availibility = False
        predictor = test_model()

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
        else:
            run_id = runs.iloc[0]["run_id"]
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            predictor = predictorClass(model=model)
            status = "alive"

    app = FastAPI(title="attr_prediction_api")

    @app.get("/health")
    def health():
        return {"status": status}

    @app.post("/predict", response_model=PredictResponse)
    def make_prediction(request: PredictRequest):
        return predictor.predict(request.model_dump())

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
