from src.models.predict_model import predictorClass
from src.config import MLFLOW_TRACKING_URI, PIPELINE_SCHEMA_VERSION
import mlflow
import requests


def test_predictorClass():
    try:
        requests.get(MLFLOW_TRACKING_URI, timeout=10.0)
    except requests.RequestException:
        print("MLflow server unavailible, ", end="")
        print("predictorClass tests are skipped")
        return None

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.pipeline_schema_version = \
'{PIPELINE_SCHEMA_VERSION}' AND tags.model_saved = 'True'",
    )

    if len(runs) == 0:
        print("There are no suitable models in mlflow, ", end="")
        print("predictorClass tests are skipped")
        return None
    else:
        run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        predictor = predictorClass(model=model)

        test_request = {
            "Age": 56,
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 441,
            "Department": "Research & Development",
            "DistanceFromHome": 14,
            "Education": 4,
            "EducationField": "Life Sciences",
            "EmployeeCount": 1,
            "EmployeeNumber": 161,
            "EnvironmentSatisfaction": 2,
            "Gender": "Female",
            "HourlyRate": 72,
            "JobInvolvement": 3,
            "JobLevel": 1,
            "JobRole": "Research Scientist",
            "JobSatisfaction": 2,
            "MaritalStatus": "Married",
            "MonthlyIncome": 4963,
            "MonthlyRate": 4510,
            "NumCompaniesWorked": 9,
            "Over18": "Y",
            "OverTime": "Yes",
            "PercentSalaryHike": 18,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 1,
            "StandardHours": 80,
            "StockOptionLevel": 3,
            "TotalWorkingYears": 7,
            "TrainingTimesLastYear": 2,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 5,
            "YearsInCurrentRole": 4,
            "YearsSinceLastPromotion": 4,
            "YearsWithCurrManager": 3,
        }

        test_request2 = {
            "Unnamed: 0": 33,
            "Age": 39,
            "Attrition": "Yes",
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 895,
            "Department": "Sales",
            "DistanceFromHome": 5,
            "Education": 3,
            "EducationField": "Technical Degree",
            "EmployeeCount": 1,
            "EmployeeNumber": 42,
            "EnvironmentSatisfaction": 4,
            "Gender": "Male",
            "HourlyRate": 56,
            "JobInvolvement": 3,
            "JobLevel": 2,
            "JobRole": "Sales Representative",
            "JobSatisfaction": 4,
            "MaritalStatus": "Married",
            "MonthlyIncome": 2086,
            "MonthlyRate": 3335,
            "NumCompaniesWorked": 3,
            "Over18": "Y",
            "OverTime": "No",
            "PercentSalaryHike": 14,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StandardHours": 80,
            "StockOptionLevel": 1,
            "TotalWorkingYears": 19,
            "TrainingTimesLastYear": 6,
            "WorkLifeBalance": 4,
            "YearsAtCompany": 1,
            "YearsInCurrentRole": 0,
            "YearsSinceLastPromotion": 0,
            "YearsWithCurrManager": 0,
        }

        ans = predictor.predict(test_request)
        assert ans["Attrition"] in [0, 1]

        ans2 = predictor.predict(test_request2)
        assert ans2["Attrition"] in [0, 1]
