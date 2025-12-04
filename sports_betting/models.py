"""Model building utilities for the ensemble sports betting model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .data import TrainingConfig


def build_preprocessing_pipeline(
    categorical_features: Iterable[str], numerical_features: Iterable[str]
) -> ColumnTransformer:
    """Create a preprocessing pipeline for mixed data types."""

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, list(categorical_features)),
            ("numeric", numeric_pipeline, list(numerical_features)),
        ]
    )


@dataclass
class EnsembleSportsModel:
    """Stacked ensemble model tailored for multi-sport betting outcomes."""

    config: TrainingConfig
    base_estimators: Optional[List[Tuple[str, Pipeline]]] = None
    stacker: Optional[Pipeline] = None
    model: Optional[Pipeline] = None

    def build_model(self, feature_columns: List[str]) -> Pipeline:
        categorical_features = self.config.resolved_categoricals(feature_columns)
        numerical_features = self.config.resolved_numericals(feature_columns)
        preprocessing = build_preprocessing_pipeline(categorical_features, numerical_features)

        base_estimators = self.base_estimators or self._default_estimators()
        stacker = self.stacker or LogisticRegression(max_iter=500, class_weight="balanced")

        model = Pipeline(
            steps=[
                ("preprocess", preprocessing),
                (
                    "ensemble",
                    StackingClassifier(
                        estimators=base_estimators,
                        final_estimator=stacker,
                        stack_method="predict_proba",
                        passthrough=False,
                    ),
                ),
            ]
        )

        self.model = model
        return model

    def fit(self, frame: pd.DataFrame) -> Pipeline:
        if self.model is None:
            feature_columns = [col for col in frame.columns if col != self.config.target]
            self.build_model(feature_columns)

        y = frame[self.config.target]
        X = frame.drop(columns=[self.config.target])

        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        assert self.model is not None
        self.model.fit(X, y, ensemble__sample_weight=sample_weight)
        return self.model

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fit; call `fit` first.")
        return self.model.predict_proba(frame)

    def _default_estimators(self) -> List[Tuple[str, Pipeline]]:
        categorical_features = self.config.categorical_features
        numerical_features = self.config.numerical_features
        preprocessing = build_preprocessing_pipeline(categorical_features, numerical_features)

        logistic = Pipeline(
            steps=[
                ("preprocess", preprocessing),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=300,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )

        forest = Pipeline(
            steps=[
                ("preprocess", preprocessing),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=17,
                    ),
                ),
            ]
        )

        gradient = Pipeline(
            steps=[
                ("preprocess", preprocessing),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        C=0.6,
                        class_weight="balanced",
                        max_iter=400,
                    ),
                ),
            ]
        )

        return [
            ("logit", logistic),
            ("forest", forest),
            ("calibrated_logit", gradient),
        ]


@dataclass
class EvaluationResult:
    auc_by_sport: Dict[str, float]
    overall_auc: float

    def as_markdown(self) -> str:
        sports_rows = "\n".join(
            f"- {sport}: {auc:.3f}" for sport, auc in sorted(self.auc_by_sport.items())
        )
        return f"**Overall ROC-AUC:** {self.overall_auc:.3f}\n\n**By sport:**\n{sports_rows}"


def evaluate_model(frame: pd.DataFrame, config: TrainingConfig, n_splits: int = 5) -> EvaluationResult:
    if config.sports_column is None or config.sports_column not in frame.columns:
        raise ValueError("Sports column missing; update TrainingConfig.sports_column to match your data.")

    feature_columns = [col for col in frame.columns if col != config.target]
    model = EnsembleSportsModel(config)
    model.build_model(feature_columns)

    y = frame[config.target]
    X = frame.drop(columns=[config.target])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(
        model.model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    overall_auc = roc_auc_score(y, y_pred)

    auc_by_sport: Dict[str, float] = {}
    for sport, subset in frame.groupby(config.sports_column):
        if subset[config.target].nunique() < 2:
            continue
        auc_by_sport[sport] = roc_auc_score(subset[config.target], y_pred[subset.index])

    return EvaluationResult(auc_by_sport=auc_by_sport, overall_auc=overall_auc)


def create_roc_plot(frame: pd.DataFrame, config: TrainingConfig, y_pred: np.ndarray) -> RocCurveDisplay:
    y = frame[config.target]
    display = RocCurveDisplay.from_predictions(y_true=y, y_pred=y_pred)
    return display
