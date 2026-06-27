# -*- coding: utf-8 -*-
import click
import logging
import kagglehub

from pathlib import Path


def load_dataset() -> Path:
    """
    Downloads IBM HR attrition dataset from kagglehub into data/external folder

    Returns: path to saved data.
    """
    project_dir = Path(__file__).resolve().parents[2]
    external_dir = project_dir / "data" / "external"
    external_dir.mkdir(exist_ok=True)
    path = kagglehub.dataset_download(
        "pavansubhasht/ibm-hr-analytics-attrition-dataset",
        output_dir=external_dir,
        force_download=True,
    )
    dvc_keep = path / ".dvckeep"
    dvc_keep.touch()

    return Path(path)


@click.command()
def main():
    """
    Turns load_dataset() function into console command
    """
    logger = logging.getLogger(__name__)
    logger.info("loading raw dataset")
    path = load_dataset()
    logger.info(f"Dataset files saved in: {path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
