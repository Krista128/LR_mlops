### Основные возможности

    
- ML-модель для прогноза увольнения — определяет вероятность ухода сотрудника на основе его характеристик
- ML-модели под версионированием — DVC для управления моделями
- Docker-поддержка — быстрое развёртывание в любом окружении

==============================

### Технологический стек

Framework: FastAPI

Web-сервер: ???

ML: Scikit-learn

Версионирование моделей	DVC (локальная папка Яндекс Диска)

Контейнеризация: Docker

Python: 3.9+

==============================

### Требования

- Python 3.9 или выше
- Windows
- Git 
- Docker

==============================

### Установка зависимостей

Клонируйте репозиторий:

    git clone https://github.com/Krista128/LR_mlops.git
    cd LR_mlops
    pip install -r requirements.txt

### Скачивание ML-модели

Модель хранится в DVC и привязана к Яндекс Диску. Для её получения:

Вариант 1: Скачать напрямую
    Перейдите по ссылке на модель - https://disk.yandex.ru/d/FxsH88yKKJqYLw
    Скачайте файл и поместите в директорию models/ проекта

Вариант 2: Использовать DVC (если у вас установлен Яндекс диск локально)

    установить Яндекс Диск
    сохранить папку с моделью по ссылке выше
    привязать dvc к локальной папке Яндекс диска:
    dvc remote add -d myremote /path/to/local/folder

Mlops_lr
==============================

Lab work by MLOps

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── raw            <- The original, immutable data dump.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   └── processed      <- The final, canonical data sets for modeling.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements-dev.txt   <- The requirements for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── requirements.txt   <- The requirements file for app's container
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── config.py      <- global project parameters
    │   ├── utils.py       <- shared utility functions used across the project
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── download_dataset.py   <- Downloads the data to data/external
    │   │   ├── make_dataset.py       <- Merging data from multiple files into one .csv file
    │   │   └── make_split.py         <- Split data into train/val/test
    │   │
    │   ├── features       
    │   │   └── build_features.py <- Turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
