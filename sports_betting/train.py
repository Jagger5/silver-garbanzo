"""Train and evaluate the ensemble sports betting model from the CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import TrainingConfig, generate_synthetic_dataset, load_dataset, load_ncaa_basketball
from .models import EnsembleSportsModel, evaluate_model


@dataclass
class CLIArgs:
    data_path: Optional[Path]
    output_dir: Path
    generate_sample: bool
    ncaa_sample: bool
    test_size: float
    random_state: int
    sports_column: str



def parse_args() -> CLIArgs:
    parser = argparse.ArgumentParser(description="Train an ensemble sports betting model.")
    parser.add_argument("--data", type=Path, help="Path to CSV containing training data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Where to write trained artifacts and reports.",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a synthetic multi-sport dataset instead of loading one.",
    )
    parser.add_argument(
        "--ncaa-sample",
        action="store_true",
        help="Use the bundled NCAA men's basketball sample dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to reserve for validation.",
    )
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument(
        "--sports-column",
        type=str,
        default="sport",
        help="Name of the column containing the sport identifier.",
    )

    args = parser.parse_args()
    return CLIArgs(
        data_path=args.data,
        output_dir=args.output_dir,
        generate_sample=args.generate_sample,
        ncaa_sample=args.ncaa_sample,
        test_size=args.test_size,
        random_state=args.random_state,
        sports_column=args.sports_column,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.ncaa_sample:
        dataset_path = Path(__file__).resolve().parent.parent / "data" / "ncaa_2024_sample.csv"
        frame = load_ncaa_basketball(dataset_path)
    elif args.generate_sample:
        frame = generate_synthetic_dataset(random_state=args.random_state)
    elif args.data_path is not None:
        if args.sports_column == "sport":
            try:
                frame = load_ncaa_basketball(args.data_path)
            except Exception:
                frame = load_dataset(args.data_path)
        else:
            frame = load_dataset(args.data_path)
    else:
        raise SystemExit("Provide --data <path> or --generate-sample to build the dataset.")

    # Remove post-game leakage columns if present.
    for col in ["home_score", "away_score", "outcome"]:
        if col in frame.columns:
            frame = frame.drop(columns=[col])

    config = TrainingConfig(
        sports_column=args.sports_column,
        categorical_features=[
            col
            for col in [
                args.sports_column,
                "home_team",
                "away_team",
                "conference",
                "home_conference",
                "away_conference",
            ]
            if col in frame.columns
        ],
        numerical_features=[
            col
            for col in frame.columns
            if col
            not in {
                args.sports_column,
                "target",
                "home_team",
                "away_team",
                "conference",
                "home_conference",
                "away_conference",
            }
        ],
    )

    train_df, valid_df = train_test_split(
        frame,
        test_size=args.test_size,
        stratify=frame[config.target],
        random_state=args.random_state,
    )

    ensemble = EnsembleSportsModel(config)
    ensemble.fit(train_df)

    evaluation = evaluate_model(valid_df, config=config, n_splits=3)
    report_path = args.output_dir / "evaluation.md"
    report_path.write_text(evaluation.as_markdown())

    model_path = args.output_dir / "ensemble_model.joblib"
    joblib.dump(ensemble, model_path)

    summary = (
        f"Saved model to {model_path}\n"
        f"Validation ROC-AUC: {evaluation.overall_auc:.3f}\n"
        f"Per-sport AUC: {evaluation.auc_by_sport}\n"
    )
    print(summary)

    predictions = ensemble.predict_proba(valid_df.drop(columns=[config.target]))[:, 1]
    prediction_frame = pd.DataFrame(
        {
            "sport": valid_df[config.sports_column].values,
            "actual": valid_df[config.target].values,
            "predicted_prob": predictions,
        }
    )
    prediction_path = args.output_dir / "validation_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)


if __name__ == "__main__":
    main()
