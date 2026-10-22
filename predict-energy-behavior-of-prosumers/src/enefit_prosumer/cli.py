from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from enefit_prosumer.data import DataStorage
from enefit_prosumer.features import FeatureEngineer
from enefit_prosumer.model import fit_model


def train(args: argparse.Namespace) -> None:
    data = DataStorage(root=args.input_dir, start_date=args.start_date)
    features = FeatureEngineer(data).generate_features(data.df_data)
    features = features[features["target"].notna()]
    y = features.pop("target")

    model = fit_model(features, y, n_jobs=args.n_jobs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": features.columns.to_list()}, args.output)
    print(f"Saved model to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enefit prosumer forecasting utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a local XGBoost model")
    train_parser.add_argument("--input-dir", default="input", help="Directory with Kaggle CSV files")
    train_parser.add_argument("--output", default="models/xgb.joblib", help="Model artifact path")
    train_parser.add_argument("--start-date", default="2022-01-01", help="Training lower bound")
    train_parser.add_argument("--n-jobs", default=-1, type=int, help="XGBoost worker count")
    train_parser.set_defaults(func=train)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
