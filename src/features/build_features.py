import numpy as np
import pandas as pd
import click
import logging
from pathlib import Path


def add_new_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 7 new features
    """
    data = data.copy()
    # показывает «лояльность» сотрудника к компании
    data["TenureRatio"] = np.where(
        data["TotalWorkingYears"] > 0,
        data["YearsAtCompany"] / data["TotalWorkingYears"],
        0,
    )

    data["RoleEngagementScore"] = (
        data["JobLevel"] * data["JobInvolvement"]
    )  # комбинирует статус и вовлечённость

    # измеряет «специализацию» или «застой»
    data["RoleStabilityRatio"] = np.where(
        data["TotalWorkingYears"] > 0,
        data["YearsInCurrentRole"] / data["TotalWorkingYears"],
        0,
    )
    # индикатор стабильности отношений
    data["ManagerStabilityRatio"] = np.where(
        data["YearsInCurrentRole"] > 0,
        data["YearsWithCurrManager"] / data["YearsInCurrentRole"],
        1,
    )

    # как быстро сотрудник растёт в компании
    data["PromotionRate"] = np.where(
        data["YearsAtCompany"] > 0,
        data["JobLevel"] / data["YearsAtCompany"],
        0,
    )
    # Доход на единицу опыта
    data["IncomePerYear"] = np.where(
        data["TotalWorkingYears"] > 0,
        data["MonthlyIncome"] / data["TotalWorkingYears"],
        0,
    )

    # oжидания vs реальность
    data["SalaryHikeExpectation"] = (
        data["PercentSalaryHike"] - data["PerformanceRating"]
    )

    return data


def dummies_maker(df: pd.DataFrame, dummies_dict: dict) -> pd.DataFrame:
    """
    supportive func for make_dummies
    """
    df = df.copy()
    for col_name in dummies_dict:
        for value in dummies_dict[col_name]:
            df[f"{col_name}_{value}"] = df[col_name].apply(
                lambda x: 1 if x == value else 0
            )

    df.drop(columns=dummies_dict.keys(), inplace=True)
    return df


def make_dummies(data: pd.DataFrame) -> pd.DataFrame:
    """
    implements one-hot-encoding to selected features
    """
    dummies_dict = {
        "Attrition": ["Yes", "No"],
        "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
        "Department": ["Sales", "Research & Development", "Human Resources"],
        "EducationField": [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
        ],
        "Gender": ["Female", "Male"],
        "JobRole": [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources",
        ],
        "MaritalStatus": ["Single", "Married", "Divorced"],
        "OverTime": ["Yes", "No"],
    }

    return dummies_maker(data, dummies_dict)


def drop_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    returns: DataFrame without selected columns
    """
    data = data.copy()
    data = data.drop(
        columns=[
            "MonthlyIncome",
            "TotalWorkingYears",
            "YearsInCurrentRole",
            "YearsWithCurrManager",
            "PercentSalaryHike",
            "Over18",
            "EmployeeCount",
            "StandardHours",
            "EmployeeNumber",
            "Attrition_No",
        ]
    )
    return data


@click.command()
@click.argument("input_csv_path", type=click.STRING)
@click.argument("output_csv_path", type=click.STRING)
def main(input_csv_path: str, output_csv_path: str):
    """
    Runs data processing scripts to turn raw data into more realible shape

    input_csv_path: path to csv file in Project/data/..
    example: raw/many_nan.csv

    output_csv_path: path in Project/data/..
    example: interim/no_nan.csv
    """
    logger = logging.getLogger(__name__)
    project_dir = Path(__file__).resolve().parents[2]
    csv_in_path = project_dir / "data" / input_csv_path
    df = pd.read_csv(csv_in_path, index_col="Unnamed: 0")
    logger.info("csv loaded")
    df = add_new_features(df)
    logger.info("new features added")
    df = make_dummies(df)
    logger.info("OHE completed")
    df = drop_features(df)
    logger.info("features droped")
    csv_path = (project_dir / "data" / output_csv_path).with_suffix(".csv")
    df.to_csv(csv_path)
    logger.info(f"saved in {csv_path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
