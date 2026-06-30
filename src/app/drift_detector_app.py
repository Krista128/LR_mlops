import time
import mlflow
import requests
import os
import asyncio
import uvicorn


import numpy as np
import pandas as pd

from contextlib import asynccontextmanager
from src.app.db_querriees import db_connector
from fastapi import FastAPI, Response, HTTPException

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.features.build_features import (
    drop_features,
    add_new_features,
    make_dummies,
)
from src.config import (
    MLFLOW_TRACKING_URI,
    PIPELINE_SCHEMA_VERSION,
    PREDICTOR_ADRESS,
)

MLFLOW_TRACKING_URI = "http://localhost:5000"
PREDICTOR_ADRESS = "http://localhost:8000"


from scipy.stats import ks_2samp, chi2_contingency
from sklearn.base import BaseEstimator, clone
from src.data.make_split import make_split
from src.models.train_model import verify_model, train_model
from datetime import datetime

status = {"mlflow": "rolling", "postgres": "rolling"}
WIN_LEN = 1470
CONCEPT_DRIFT_ROC_AUC_DELTA = 0.035
CHECK_INTERVAL_SECONDS = 10


loop_iterations_total = Counter(
    "drift_detector_loop_iterations_total",
    "Total number of drift detector loop iterations",
)

loop_errors_total = Counter(
    "drift_detector_loop_errors_total",
    "Total number of drift detector loop errors",
)

loop_duration_seconds = Histogram(
    "drift_detector_loop_duration_seconds",
    "Duration of one drift detector loop iteration in seconds",
)

last_success_timestamp = Gauge(
    "drift_detector_last_success_timestamp",
    "Unix timestamp of last successful loop iteration",
)

postgres_up = Gauge(
    "drift_detector_postgres_up",
    "Postgres availability: 1 = up, 0 = down",
)

mlflow_up = Gauge(
    "drift_detector_mlflow_up",
    "MLflow availability: 1 = up, 0 = down",
)

unseen_requests_count = Gauge(
    "drift_detector_unseen_requests_count",
    "Number of unseen requests waiting for data drift check",
)

data_drift_detected = Gauge(
    "drift_detector_data_drift_detected",
    "Data drift flag: 1 = detected, 0 = not detected",
)

target_drift_detected = Gauge(
    "drift_detector_target_drift_detected",
    "Target drift flag: 1 = detected, 0 = not detected",
)

concept_drift_detected = Gauge(
    "drift_detector_concept_drift_detected",
    "Concept drift flag: 1 = detected, 0 = not detected",
)

drift_p_value = Gauge(
    "drift_detector_p_value",
    "P-value of drift statistical test",
    ["drift_type", "feature", "test"],
)

drift_statistic = Gauge(
    "drift_detector_statistic",
    "Statistic value of drift statistical test",
    ["drift_type", "feature", "test"],
)

drift_feature_detected = Gauge(
    "drift_detector_feature_drift_detected",
    "Feature-level drift flag: 1 = detected, 0 = not detected",
    ["drift_type", "feature", "test"],
)

concept_roc_auc_train = Gauge(
    "drift_detector_concept_roc_auc_train",
    "ROC-AUC on train reference window",
)

concept_roc_auc_window = Gauge(
    "drift_detector_concept_roc_auc_window",
    "ROC-AUC on current labeled window",
)

concept_roc_auc_delta = Gauge(
    "drift_detector_concept_roc_auc_delta",
    "ROC-AUC drop between train and current labeled window",
)

retrain_triggered_total = Counter(
    "drift_detector_retrain_triggered_total",
    "Total number of retraining attempts triggered",
)

predictor_pull_total = Counter(
    "drift_detector_predictor_pull_total",
    "Total number of predictor pull_new_model requests",
)


def try_to_connect_db(connector) -> bool:
    global status
    db_availibility = False
    attempts = 0
    while not db_availibility and attempts < 12:
        db_availibility = connector.connection_alive()
        time.sleep(10)
        attempts += 1

    if not db_availibility:
        print(f"Database unavailible ({attempts} attempts to connect)")
        status["postgres"] = "No connection"
        return True
    else:
        status["postgres"] = "Ready"
        return False


def load_model() -> dict | None:
    global status
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
            status["mlflow"] = "Ready, no suitable model at last attempt"
            return None
        else:
            run_id = runs.iloc[0]["run_id"]
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            status["mlflow"] = "Ready"
            result = {"model": model, "run_id": run_id}
            return result
    else:
        print("MLflow server unavailible, cant load model")
        print(f"({attempts} attempts to connect)")
        print("App will start in test mode, without model")
        status["mlflow"] = "No connection"
        return None


CAT_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

USEFULL_COLUMNS = (
    "Age",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyRate",
    "NumCompaniesWorked",
    "OverTime",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsSinceLastPromotion",
)


def compare_feature_drift(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    feature_columns: tuple[str] = USEFULL_COLUMNS,
    categorical_columns: tuple[str] = CAT_COLS,
    alpha: float = 0.05,
    min_expected_count: int = 5,
) -> pd.DataFrame:
    """
    Сравнивает распределения признаков в real_df и synthetic_df.

    Для числовых признаков:
        KS-test.

    Для категориальных признаков:
        Chi-square test.

    Возвращает DataFrame с p-value и флагом drift_detected.
    """

    results = []

    categorical_columns = set(categorical_columns)

    for col in feature_columns:
        if col not in real_df.columns or col not in synthetic_df.columns:
            results.append(
                {
                    "feature": col,
                    "test": "missing_column",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "drift_detected": None,
                    "note": "column missing in one of dataframes",
                }
            )
            continue

        real_s = real_df[col].dropna()
        synth_s = synthetic_df[col].dropna()

        if len(real_s) == 0 or len(synth_s) == 0:
            results.append(
                {
                    "feature": col,
                    "test": "empty_column",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "drift_detected": None,
                    "note": "empty after dropna",
                }
            )
            continue

        if col in categorical_columns:
            all_categories = sorted(
                set(real_s.astype(str).unique())
                | set(synth_s.astype(str).unique())
            )

            real_counts = (
                real_s.astype(str)
                .value_counts()
                .reindex(all_categories, fill_value=0)
            )
            synth_counts = (
                synth_s.astype(str)
                .value_counts()
                .reindex(all_categories, fill_value=0)
            )

            contingency = np.vstack([real_counts.values, synth_counts.values])

            try:
                stat, p_value, _, expected = chi2_contingency(contingency)

                note = ""
                if (expected < min_expected_count).any():
                    note = "some expected counts are low; chi-square may be unstable"

                results.append(
                    {
                        "feature": col,
                        "test": "chi2",
                        "statistic": stat,
                        "p_value": p_value,
                        "drift_detected": bool(p_value < alpha),
                        "note": note,
                    }
                )

            except ValueError as e:
                results.append(
                    {
                        "feature": col,
                        "test": "chi2",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "drift_detected": None,
                        "note": str(e),
                    }
                )

        else:
            real_num = pd.to_numeric(real_s, errors="coerce").dropna()
            synth_num = pd.to_numeric(synth_s, errors="coerce").dropna()

            if len(real_num) == 0 or len(synth_num) == 0:
                results.append(
                    {
                        "feature": col,
                        "test": "ks",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "drift_detected": None,
                        "note": "non-numeric values after conversion",
                    }
                )
                continue

            stat, p_value = ks_2samp(real_num, synth_num)

            results.append(
                {
                    "feature": col,
                    "test": "ks",
                    "statistic": stat,
                    "p_value": p_value,
                    "drift_detected": bool(p_value < alpha),
                    "note": "",
                }
            )

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by=["drift_detected", "p_value"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    return result_df


def booler(x):
    if x in ["No", 0, False]:
        return 0
    elif x in ["Yes", 1, True]:
        return 1


def preproc(df):
    df = add_new_features(df)
    df = make_dummies(df)
    df = drop_features(df)
    return df


def update_drift_report_metrics(report_df: pd.DataFrame, drift_type: str):
    for _, row in report_df.iterrows():
        feature = str(row["feature"])
        test = str(row["test"])

        p_value = row["p_value"]
        statistic = row["statistic"]
        drift_detected = row["drift_detected"]

        if pd.notna(p_value):
            drift_p_value.labels(
                drift_type=drift_type,
                feature=feature,
                test=test,
            ).set(float(p_value))

        if pd.notna(statistic):
            drift_statistic.labels(
                drift_type=drift_type,
                feature=feature,
                test=test,
            ).set(float(statistic))

        if drift_detected is not None and pd.notna(drift_detected):
            drift_feature_detected.labels(
                drift_type=drift_type,
                feature=feature,
                test=test,
            ).set(int(bool(drift_detected)))


def check_data_drift(conn: db_connector):
    unseen_req = conn.count_unseen_requests()
    if unseen_req is None:
        unseen_req = 0
        return False
    print(f"count_unseen_requests returned {unseen_req}")
    unseen_requests_count.set(unseen_req)
    if unseen_req >= WIN_LEN:
        win_data = conn.get_unseen_requests()
        if len(win_data) == WIN_LEN:
            run_id = win_data.iloc[0]["model_train_run_id"]
            train_data = conn.get_train_data(run_id)

            print("===" * 20, "DATA", "===" * 20)
            print(train_data.columns)
            print("len(train_data)", len(train_data))
            print("===" * 20)
            print(win_data.columns)
            print("len(train_data)", len(win_data))
            print("===" * 20)

            data_drift_report = compare_feature_drift(
                real_df=train_data,
                synthetic_df=win_data,
                alpha=0.05,
                feature_columns=USEFULL_COLUMNS,
                categorical_columns=CAT_COLS,
            )
            update_drift_report_metrics(data_drift_report, drift_type="data")

            report = {}
            report["w_start"] = int(win_data["row_id"].iloc[0])
            report["w_stop"] = int(win_data["row_id"].iloc[-1])
            report["run_id"] = run_id
            report["data_drift"] = bool(
                data_drift_report["drift_detected"].any()
            )
            report["target_drift"] = None
            report["concept_drift"] = None
            report["trained"] = False

            if report["data_drift"]:
                print("data_drift detected!")
            else:
                print("No data_drift")
            conn.insert_drift_report(report)
            return True
        else:
            return False
    else:
        return False


def pull_model_predictors(new_run_id):
    for jj in range(60):
        requests.get(PREDICTOR_ADRESS + "/pull_new_model")
        predictor_pull_total.inc()
        if jj > 0:
            time.sleep(0.5)
        jj += 1
        try:
            response = requests.get(PREDICTOR_ADRESS + "/health")
            response = response.json()
            response = response["run_id"]
        except:
            pass


def check_target_and_concept_drift(conn: db_connector):
    # check target drift
    window_metainfo = conn.get_earliest_window_with_new_labels()
    if len(window_metainfo) > 0:
        window_metainfo = window_metainfo.iloc[0].to_dict()
        win_data = conn.get_window_rows(
            window_metainfo["w_start"], window_metainfo["w_stop"]
        )
        print(f"win_data for target and concept drift: {len(win_data)}")
        train_data = conn.get_train_data(window_metainfo["run_id"])

        print("===" * 20, "TARGET", "===" * 20)
        print("WINIDD ", window_metainfo["window_id"])
        print(train_data.columns)
        print("len(train_data)", len(train_data))
        print("===" * 20)
        print(win_data.columns)
        print("len(win_data)", len(win_data))

        print("===" * 20)

        target_drift_report = compare_feature_drift(
            real_df=train_data.copy(),
            synthetic_df=win_data.copy(),
            alpha=0.05,
            feature_columns=["GT_Attrition"],
            categorical_columns=["GT_Attrition"],
        )

        update_drift_report_metrics(target_drift_report, drift_type="target")

        report = {}
        report["window_id"] = window_metainfo["window_id"]
        report["target_drift"] = bool(
            target_drift_report.iloc[0]["drift_detected"]
        )
        target_drift_detected.set(int(report["target_drift"]))
        if report["target_drift"]:
            print("target_drift detected!")
        else:
            print("NO target_drift")

        del target_drift_report

        # check concept drift
        model_dict = load_model()
        if model_dict is not None:
            model = model_dict["model"]
            target_train = train_data["GT_Attrition"].copy().apply(booler)
            target_win = win_data["GT_Attrition"].copy().apply(booler)

            train_data = train_data.drop(
                columns=[
                    "row_id",
                    "model_train_run_id",
                    "pipeline_shema_version",
                    "GT_Attrition",
                ]
            )
            win_data = win_data.drop(columns=["row_id", "GT_Attrition"])

            train_data = preproc(train_data)
            win_data = preproc(win_data)

            print("===" * 20, "CONCEPT", "===" * 20)
            print(train_data.columns)
            print("len(train_data)", len(train_data))
            print("===" * 20)
            print("len(target_train)", len(target_train))
            print("===" * 20)
            print(win_data.columns)
            print("len(win_data)", len(win_data))
            print("===" * 20)
            print("len(target_win)", len(target_win))
            print("===" * 20)

            mertics_train = verify_model(
                model, train_data, target_train, verbose=False
            )
            metrics_win = verify_model(
                model, win_data, target_win, verbose=False
            )

            delta = mertics_train["roc_auc"] - metrics_win["roc_auc"]
            report["concept_drift"] = (
                True if delta > CONCEPT_DRIFT_ROC_AUC_DELTA else False
            )

            concept_roc_auc_train.set(float(mertics_train["roc_auc"]))
            concept_roc_auc_window.set(float(metrics_win["roc_auc"]))
            concept_roc_auc_delta.set(float(delta))
            concept_drift_detected.set(int(report["concept_drift"]))

            if report["concept_drift"]:
                print("Concept_drift DETECTED!")
            else:
                print("NO concept drift")

            conn.update_drift_report(
                window_id=report["window_id"],
                target_drift=report["target_drift"],
                concept_drift=report["concept_drift"],
            )
        return True
    else:
        return False


def check_retrain(conn: db_connector):
    window_metainfo = conn.get_window_for_retrain()
    if len(window_metainfo) > 0:
        model_dict = load_model()
        if model_dict is not None:
            print("Training model")
            retrain_triggered_total.inc()
            window_metainfo = window_metainfo.iloc[0].to_dict()
            win_data = conn.get_window_rows(
                window_metainfo["w_start"], window_metainfo["w_stop"]
            )
            print(f"window_for_retrain  {len(win_data)}")
            win_data = win_data.drop(columns=["Attrition", "row_id"])
            win_data = preproc(
                win_data.rename(columns={"GT_Attrition": "Attrition"})
            )
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = make_split(
                win_data
            )

            x_train["_target"] = y_train
            x_val["_target"] = y_val

            model = model_dict["model"]
            old_run_id = model_dict["model"]
            run = mlflow.get_run(old_run_id)
            exp_id = run.info.experiment_id
            exp = mlflow.get_experiment(exp_id)
            exp_name = exp.name
            if "__retrain" not in exp_name:
                exp_name = exp_name + "__retrain"

            new_run_id = train_model(
                models=[model],
                train_dataset=x_train,
                valid_dataset=x_val,
                experiment_name=exp_name,
                repeats=1,
                production="True",
            )

            with mlflow.start_run(run_id=old_run_id):
                mlflow.set_tag("production", "deprecated")
                mlflow.set_tag(
                    "deprecation_date",
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                )

            pull_model_predictors(new_run_id)


def main():
    host = os.getenv("DB_HOST")
    db_name = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db_port = os.getenv("DB_PORT")
    # db_url = (
    #    f"postgresql+psycopg2://{user}:{password}@{host}:{db_port}/{db_name}"
    # )
    db_url = "postgresql+psycopg2://user1:paswarddd@localhost:7000/attr_db"
    postgres_db = db_connector(db_url)
    try_to_connect_db(postgres_db)

    async def loop():
        while True:
            start_time = time.time()
            loop_iterations_total.inc()
            try:
                postgres_up.set(1 if postgres_db.connection_alive() else 0)
                mlflow_up.set(1 if status.get("mlflow") == "Ready" else 0)

                while check_data_drift(postgres_db):
                    pass
                while check_target_and_concept_drift(postgres_db):
                    pass
                check_retrain(postgres_db)
                last_success_timestamp.set(time.time())
            except Exception as e:
                loop_errors_total.inc()
                print(e)
            finally:
                loop_duration_seconds.observe(time.time() - start_time)

            await asyncio.sleep(CHECK_INTERVAL_SECONDS)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(loop())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="drift_detector-app", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": status}

    @app.get("/check_connection")
    def check_connection():
        return {"status": try_to_connect_db(postgres_db)}

    @app.get("/metrics")
    def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8500,
        reload=False,
    )


if __name__ == "__main__":
    main()
