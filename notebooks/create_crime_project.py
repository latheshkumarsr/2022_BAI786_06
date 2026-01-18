#!/usr/bin/env python3
"""
create_crime_project.py

Creates the directory tree and placeholder files for the
crime_pattern_prediction project.

Usage:
    python create_crime_project.py
    python create_crime_project.py --target /path/to/crime_pattern_prediction
"""

import argparse
from pathlib import Path
import json
import textwrap

DEFAULT_TARGET = Path.cwd() / "crime_pattern_prediction"

FILES_AND_CONTENTS = {
    "README.md": "# crime_pattern_prediction\n\nProject scaffold for crime pattern analysis and prediction.\n",
    "LICENSE": "MIT License\n\nCopyright (c) YEAR\n\n[Replace with actual license text]\n",
    ".gitignore": textwrap.dedent(
        """
        __pycache__/
        *.pyc
        .env
        .venv
        venv/
        .ipynb_checkpoints/
        data/
        models/
        *.joblib
        *.parquet
        """
    ).strip() + "\n",
    "requirements.txt": "flask\npandas\nscikit-learn\njoblib\npyarrow\nfastparquet\nfolium\nplotly\npytest\n",
    "environment.yml": textwrap.dedent(
        """
        name: crime_pattern_env
        channels:
          - defaults
        dependencies:
          - python=3.10
          - pip
          - pip:
            - -r requirements.txt
        """
    ).strip() + "\n",
    "Dockerfile": textwrap.dedent(
        """
        # Simple Dockerfile for Flask app
        FROM python:3.10-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt
        COPY . .
        ENV FLASK_APP=src/app
        CMD ["flask", "run", "--host=0.0.0.0"]
        """
    ).strip() + "\n",
    "docker-compose.yml": textwrap.dedent(
        """
        version: '3.8'
        services:
          web:
            build: .
            ports:
              - "5000:5000"
            volumes:
              - .:/app
        """
    ).strip() + "\n",
    "Makefile": textwrap.dedent(
        """
        .PHONY: setup run train
        setup:
        \tpython -m pip install -r requirements.txt

        run:
        \tflask --app src.app run

        train:
        \tpython src/models/train.py
        """
    ).lstrip(),
    "data/raw/raw_data.csv": "",  # placeholder empty CSV
    "data/interim/.gitkeep": "",
    "data/processed/grid_aggregated.parquet": "",  # placeholder empty file
    "notebooks/01_Exploratory_Data_Analysis.ipynb": json.dumps({
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }, indent=2),
    "notebooks/02_Feature_Engineering_and_Vis.ipynb": json.dumps({
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }, indent=2),
    "notebooks/03_Model_Experiments.ipynb": json.dumps({
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }, indent=2),
    # src app package
    "src/app/__init__.py": textwrap.dedent(
        """
        from flask import Flask

        def create_app():
            app = Flask(__name__)
            from . import routes, api  # noqa: F401
            return app
        """
    ).lstrip(),
    "src/app/routes.py": textwrap.dedent(
        """
        from flask import render_template
        from . import create_app

        # routes will be registered via blueprint or directly in create_app
        """
    ).lstrip(),
    "src/app/api.py": textwrap.dedent(
        """
        from flask import Blueprint, jsonify, request

        bp = Blueprint('api', __name__)

        @bp.route('/predict', methods=['POST'])
        def predict():
            # placeholder predict endpoint
            return jsonify({'prediction': None})
        """
    ).lstrip(),
    "src/app/models_api.py": textwrap.dedent(
        """
        # helpers to load model and run prediction
        def load_model(path):
            raise NotImplementedError
        """
    ).lstrip(),
    "src/app/static/css/.gitkeep": "",
    "src/app/static/js/.gitkeep": "",
    "src/app/templates/index.html": "<!-- index page -->\n<html><body><h1>crime_pattern_prediction</h1></body></html>\n",
    "src/app/templates/predict.html": "<!-- predict page -->\n",
    "src/app/templates/admin.html": "<!-- admin page -->\n",
    # src data
    "src/data/load_data.py": textwrap.dedent(
        """
        def load_raw(path):
            # safe loader stub
            import pandas as pd
            return pd.read_csv(path)
        """
    ).lstrip(),
    "src/data/clean.py": textwrap.dedent(
        """
        def clean(df):
            # cleaning helpers
            return df
        """
    ).lstrip(),
    "src/data/features.py": textwrap.dedent(
        """
        # feature engineering functions (grid, time, agg)
        def make_features(df):
            return df
        """
    ).lstrip(),
    "src/data/aggregate.py": textwrap.dedent(
        """
        def aggregate_grid(df):
            return df
        """
    ).lstrip(),
    "src/data/save_load.py": textwrap.dedent(
        """
        def save_parquet(df, path):
            df.to_parquet(path)
        """
    ).lstrip(),
    # src models
    "src/models/train.py": textwrap.dedent(
        """
        def main():
            print("Training pipeline placeholder")

        if __name__ == '__main__':
            main()
        """
    ).lstrip(),
    "src/models/evaluate.py": textwrap.dedent(
        """
        def evaluate(model, X, y):
            return {}
        """
    ).lstrip(),
    "src/models/predict.py": textwrap.dedent(
        """
        def predict(pipeline, X):
            return pipeline.predict(X)
        """
    ).lstrip(),
    "src/models/persistence.py": textwrap.dedent(
        """
        def save_pipeline(pipeline, path):
            import joblib
            joblib.dump(pipeline, path)

        def load_pipeline(path):
            import joblib
            return joblib.load(path)
        """
    ).lstrip(),
    # viz
    "src/viz/maps.py": textwrap.dedent(
        """
        def create_map(df):
            # folium map stub
            return None
        """
    ).lstrip(),
    # utils
    "src/utils/io_helpers.py": textwrap.dedent(
        """
        def safe_read(path):
            with open(path, 'rb') as f:
                return f.read()
        """
    ).lstrip(),
    "src/utils/ml_helpers.py": textwrap.dedent(
        """
        def score(y_true, y_pred):
            return {}
        """
    ).lstrip(),
    "src/utils/config.py": textwrap.dedent(
        """
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        DATA_DIR = PROJECT_ROOT / 'data'
        """
    ).lstrip(),
    # tests
    "src/tests/test_features.py": textwrap.dedent(
        """
        def test_sample():
            assert True
        """
    ).lstrip(),
    "src/tests/test_train.py": textwrap.dedent(
        """
        def test_train_runs():
            assert True
        """
    ).lstrip(),
    # models placeholder (binary)
    "models/.gitkeep": "",
    "models/crime_pipeline.joblib": None,  # will be created as an empty binary file
    # reports
    "reports/figures/.gitkeep": "",
    "reports/model_report.md": "# Model Report\n\nPlaceholder\n",
    # docs
    "docs/architecture.md": "# Architecture\n\nPlaceholder\n",
    "docs/data_dictionary.md": "# Data Dictionary\n\nPlaceholder\n",
}

DIRECTORIES_TO_MAKE = [
    "data/raw",
    "data/interim",
    "data/processed",
    "notebooks",
    "src/app/static/css",
    "src/app/static/js",
    "src/app/templates",
    "src/data",
    "src/models",
    "src/viz",
    "src/utils",
    "src/tests",
    "models",
    "reports/figures",
    "docs",
]


def create_dirs(base: Path):
    for d in DIRECTORIES_TO_MAKE:
        dirpath = base.joinpath(d)
        dirpath.mkdir(parents=True, exist_ok=True)
        # create .gitkeep if directory is empty and not already present
        gitkeep = dirpath / ".gitkeep"
        if not any(dirpath.iterdir()):
            gitkeep.write_text("")


def create_files(base: Path):
    for rel_path, content in FILES_AND_CONTENTS.items():
        target = base.joinpath(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # special handling for binary placeholder
        if rel_path == "models/crime_pipeline.joblib":
            target.parent.mkdir(parents=True, exist_ok=True)
            # create an empty binary file if not exists
            if not target.exists():
                target.write_bytes(b"")
            continue

        # if content is None treat as empty
        if content is None:
            content = ""

        # Write JSON string or plain text
        mode = "w"
        text = content
        if isinstance(content, (bytes, bytearray)):
            mode = "wb"
            text = content
        if mode == "w":
            target.write_text(text, encoding="utf-8")
        else:
            target.write_bytes(text)


def main(target_dir: Path):
    print(f"Creating project at: {target_dir.resolve()}")
    target_dir.mkdir(parents=True, exist_ok=True)
    create_dirs(target_dir)
    create_files(target_dir)
    print("Done. Created folders and files. You may want to edit placeholders (README, LICENSE, etc.).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create crime_pattern_prediction scaffold")
    parser.add_argument("--target", "-t", default=str(DEFAULT_TARGET), help="Target directory")
    args = parser.parse_args()
    main(Path(args.target))
