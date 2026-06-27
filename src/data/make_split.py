import click
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def make_split(df: pd.DataFrame):
    df = df.copy()
    df.dropna()
    y = df["Attrition_Yes"]
    X = df.drop(columns=["Attrition_Yes"])
    x_train, x_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test,
        y_val_test,
        test_size=0.5,
        stratify=y_val_test,
        random_state=42,
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


@click.command()
@click.argument("input_csv_path", type=click.STRING)
@click.argument("output_csv_path", type=click.STRING)
def main(input_csv_path: str, output_csv_path: str):
    """
    Split given data in train/val/test
    Saves result in 3 files:
        <given_name>_train.csv
        <given_name>_val.csv
        <given_name>_test.csv
    where _target column represents labels

    input_csv_path: path to csv file in Project/data/.. with data
    example: interim/dataset.csv

    output_csv_path: path in Project/data/.. to save resulting files
    example: processed/dataset -> processed/dataset_train.csv, ...
    """
    logger = logging.getLogger(__name__)
    project_dir = Path(__file__).resolve().parents[2]
    csv_in_path = project_dir / "data" / input_csv_path
    logger.info(f"loading .csv file from {csv_in_path}")
    df = pd.read_csv(csv_in_path, index_col="Unnamed: 0")
    logger.info("spliting data ..")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = make_split(df)

    x_train["_target"] = y_train
    x_val["_target"] = y_val
    x_test["_target"] = y_test

    csv_path = (
        project_dir / "data" / (output_csv_path + "_train")
    ).with_suffix(".csv")
    logger.info(f"saving {csv_path}")
    x_train.to_csv(csv_path)

    csv_path = (project_dir / "data" / (output_csv_path + "_val")).with_suffix(
        ".csv"
    )
    logger.info(f"saving {csv_path}")
    x_val.to_csv(csv_path)

    csv_path = (
        project_dir / "data" / (output_csv_path + "_test")
    ).with_suffix(".csv")
    logger.info(f"saving {csv_path}")
    x_test.to_csv(csv_path)

    logger.info("saving completed")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
