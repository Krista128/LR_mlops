import pandas as pd
import click
import logging
from pathlib import Path


def read_csvs_and_concat(path: Path) -> list[pd.DataFrame]:
    """
    loads all .csv from from given path into Dataframes
    and tries to concat them by matching columns

    path: path to .csv files

    returns: list of DataFrames with differ columns
    """
    result = []

    for candidate in path.iterdir():
        if candidate.suffix == ".csv":
            df = pd.read_csv(candidate)

            jj = 0
            while jj < len(result) and (
                set(result[jj].columns) != set(df.columns)
            ):
                jj += 1
            if jj < len(result):
                result[jj] = pd.concat([result[jj], df], ignore_index=True)
            else:
                result.append(df)

    return result


def make_single_dataframe(
    list_of_dataframes: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Combines multiple dataframes into one

    list_of_dataframes: list of dataframes

    returns: dataframe combined of input ones
    """

    return list_of_dataframes[0]


@click.command()
@click.argument("input_csv_path", type=click.STRING)
@click.argument("output_csv_path", type=click.STRING)
def main(input_csv_path: str, output_csv_path: str):
    """
    Runs data processing scripts to turn several .csv into single .csv

    input_csv_path: path to folder with csv files in Project/data/..
    example: external

    output_csv_path: path in Project/data/.. to save resulting file
    example: interim/dataset.csv
    """
    logger = logging.getLogger(__name__)
    project_dir = Path(__file__).resolve().parents[2]

    csv_in_path = project_dir / "data" / input_csv_path
    logger.info(f"loading .csv files from {csv_in_path}")
    list_of_dfs = read_csvs_and_concat(csv_in_path)
    logger.info("all .csv loaded, concatenating them into singe DataFame")
    final_df = make_single_dataframe(list_of_dfs)

    project_dir = Path(__file__).resolve().parents[2]
    csv_path = (project_dir / "data" / output_csv_path).with_suffix(".csv")
    logger.info("saving dataframe")
    final_df.to_csv(csv_path)
    logger.info(f"saved in {csv_path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
