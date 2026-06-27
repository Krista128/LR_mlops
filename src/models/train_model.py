import mlflow
import mlflow.sklearn
import subprocess
import click
import logging
import time

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
)
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from itertools import product
from collections.abc import Iterable
from src.config import PIPELINE_SCHEMA_VERSION, MLFLOW_TRACKING_URI
from src.utils import SimpleTimeFormater


def get_git_info() -> dict[str, str]:
    """
    return git info for logging
    """

    def run(command: list[str]) -> str:
        return subprocess.check_output(command).decode().strip()

    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_is_dirty": str(bool(run(["git", "status", "--porcelain"]))),
    }


def verify_model(
    model: Pipeline | BaseEstimator, x_test: pd.DataFrame, y_test: pd.DataFrame
) -> dict[str, float]:
    """
    evaluate model on given x_test dataset

    returns dict with keys: f1, precision, recall, roc_auc, pr_auc
    """
    pred_proba = model.predict(x_test)

    precision, recall, thresholds = precision_recall_curve(y_test, pred_proba)
    f1_scores = (
        2
        * (precision[:-1] * recall[:-1])
        / (precision[:-1] + recall[:-1] + 1e-10)
    )
    best_threshold = thresholds[np.argmax(f1_scores)]

    print(f"Оптимальный порог по f1-score: {best_threshold:.4f}")

    pred_binary = (pred_proba >= best_threshold).astype(int)

    f1 = f1_score(y_test, pred_binary)
    precision = precision_score(y_test, pred_binary)
    recall = recall_score(y_test, pred_binary)
    roc_auc = roc_auc_score(y_test, pred_proba)
    pr_auc = average_precision_score(y_test, pred_proba)
    conf_matrix = confusion_matrix(y_test, pred_binary)

    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")
    print(f"PR AUC: {pr_auc}")
    print(f"Confusion matrix: \n {conf_matrix}")

    return {
        "f1_Score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def make_model_grid():
    """
    defines a generator that returns models with the desired hyperparameters
    """

    features_to_normalize = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "Education",
        "EnvironmentSatisfaction",
        "HourlyRate",
        "JobInvolvement",
        "JobLevel",
        "JobSatisfaction",
        "MonthlyRate",
        "NumCompaniesWorked",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StockOptionLevel",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsSinceLastPromotion",
        "TenureRatio",
        "RoleEngagementScore",
        "RoleStabilityRatio",
        "ManagerStabilityRatio",
        "PromotionRate",
        "IncomePerYear",
        "SalaryHikeExpectation",
    ]
    scaler = ColumnTransformer(
        transformers=[
            ("columns scaler", StandardScaler(), features_to_normalize)
        ],
        remainder="passthrough",
    ).set_output(transform="pandas")
    c_grid = (0.5, 1.0)

    for c in c_grid:
        yield Pipeline(
            [
                ("scaler", scaler),
                (
                    "clf",
                    svm.LinearSVC(
                        C=c, max_iter=10000, class_weight="balanced"
                    ),
                ),
            ]
        )

    gamma_grid = (0.7, 1.0)

    for c, gamma in product(c_grid, gamma_grid):
        yield Pipeline(
            [
                ("scaler", scaler),
                (
                    "clf",
                    svm.SVC(
                        kernel="rbf", gamma=gamma, C=c, class_weight="balanced"
                    ),
                ),
            ]
        )

    degree_grid = [2]
    c_grid = [0.5]

    for c, degree in product(c_grid, degree_grid):
        yield Pipeline(
            [
                ("scaler", scaler),
                (
                    "clf",
                    svm.SVC(
                        kernel="poly",
                        degree=degree,
                        gamma="auto",
                        C=c,
                        class_weight="balanced",
                    ),
                ),
            ]
        )


def train_model(
    models: Iterable,
    train_dataset: pd.DataFrame,
    valid_dataset: pd.DataFrame,
    experiment_name: str,
    repeats: int = 1,
    **params_to_log,
):
    """
    models: must be an iterable object of models to train

    fits models and validate it with logging in MLflow
    """

    train_dataset = train_dataset.copy()
    y_train = train_dataset["_target"]
    x_train = train_dataset.drop(columns=["_target"])

    y_valid = valid_dataset["_target"]
    x_valid = valid_dataset.drop(columns=["_target"])

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    best_metric = 0

    mlflow.set_experiment(experiment_name)
    for jj, base_model in enumerate(models):
        nontrained_model = clone(base_model)
        for rep in range(repeats):
            with mlflow.start_run(
                run_name=f"{experiment_name}_model_{jj + 1}_run_{rep + 1}"
            ) as run:
                mlflow.set_tags(get_git_info())
                mlflow.set_tag(
                    "pipeline_schema_version", PIPELINE_SCHEMA_VERSION
                )
                model = clone(nontrained_model)
                mlflow.set_tag("model_class", model.__class__.__name__)

                model.fit(x_train, y_train)

                metrics_to_log = verify_model(model, x_valid, y_valid)

                target_metric = metrics_to_log["roc_auc"]

                if target_metric > best_metric:
                    best_metric = target_metric
                    best_run_id = run.info.run_id
                    best_model = model

                for param in metrics_to_log:
                    mlflow.log_metric(param, metrics_to_log[param])

                for param in params_to_log:
                    mlflow.log_param(param, params_to_log[param])

                mlflow.log_params(model.get_params())

    with mlflow.start_run(run_id=best_run_id):
        mlflow.sklearn.log_model(best_model, name="model")
        mlflow.set_tag("model_saved", True)


@click.command()
@click.argument("train_dataset_csv_path", type=click.STRING)
@click.argument("valid_dataset_csv_path", type=click.STRING)
@click.argument("experiment_name", type=click.STRING)
@click.argument("repeats", type=click.INT, required=False, default=1)
@click.argument(
    "params_to_log", type=click.STRING, required=False, default=None
)
def main(
    train_dataset_csv_path: str,
    valid_dataset_csv_path: str,
    experiment_name: str,
    repeats: int = 1,
    params_to_log: str = None,
):
    """
    Runs model training with logging in MLflow

    train_dataset_csv_path: path to csv file in Project/data/..
    example: processed/dataset_train.csv

    valid_dataset_csv_path: path to csv file in Project/data/..
    example: processed/dataset_val.csv

    experiment_name: name of experiment to log in MLflow

    repeats: num of runs of each model

    params_to_log: A string matching the template "param1=v1 param2=v2"
    that defines additional parameters for logging in mlflow
    """
    logger = logging.getLogger(__name__)
    project_dir = Path(__file__).resolve().parents[2]
    train = pd.read_csv(
        project_dir
        / "data"
        / Path(train_dataset_csv_path).with_suffix(".csv"),
        index_col="Unnamed: 0",
    )
    valid = pd.read_csv(
        project_dir
        / "data"
        / Path(valid_dataset_csv_path).with_suffix(".csv"),
        index_col="Unnamed: 0",
    )

    logger.info("datasets loaded")

    params_to_log_dict = {}
    if params_to_log is not None:
        params_to_log = params_to_log.split(" ")
        for param in params_to_log:
            key, value = param.split("=")
            params_to_log_dict[key] = value
        logger.info("keyword params parsed")

    logger.info("Starting train")
    start_time = time.time()

    train_model(
        models=make_model_grid(),
        train_dataset=train,
        valid_dataset=valid,
        experiment_name=experiment_name,
        repeats=repeats,
        **params_to_log_dict,
    )

    end_time = time.time() - start_time
    logger.info(
        f"Training complete, computation time: {SimpleTimeFormater(end_time)}"
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
