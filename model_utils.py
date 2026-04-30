import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RUNS_TARGET_COLUMN = "target_runs_scored"
FIFTY_PLUS_TARGET_COLUMN = "target_scored_50_plus"
RANDOM_STATE = 42

# Rows that shouldnt be model input 
EXCLUDED_FEATURE_COLUMNS = {
    "match_id",
    "date",
    "batter",
    "batter_id",
}

# Columns saved in the detailed prediction CSVs
OUTPUT_COLUMNS = [
    "date",
    "match_id",
    "venue",
    "batting_team",
    "bowling_team",
    "innings_number",
    "batting_position",
    "player_career_runs_avg",
    "player_vs_opponent_runs_avg",
    "tree_predicted_runs",
    "tree_upside_predicted_runs",
    "tree_absolute_error",
    "tree_50_plus_probability",
    "nn_predicted_runs",
    "nn_absolute_error",
    "nn_50_plus_probability",
    "actual_runs",
    "actual_50_plus",
]

# Columns shown in the compact terminal report
PRINT_COLUMNS = [
    "date",
    "opponent",
    "venue",
    "bat_pos",
    "career_avg",
    "vs_opp_avg",
    "tree_pred",
    "tree_up",
    "tree_50p",
    "tree_err",
    "nn_pred",
    "nn_50p",
    "nn_err",
    "actual",
]


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def print_metrics(label, y_true, y_pred):
    # Print regression metrics for run predictions
    print(label)
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.3f}")
    print(f"RMSE: {rmse(y_true, y_pred):.3f}")
    print(f"R2: {r2_score(y_true, y_pred):.3f}")


def print_probability_metrics(label, y_true, probabilities):
    # Print classification metrics for 50-plus probabilities
    predictions = probabilities >= 0.5

    print(label)
    print(f"Accuracy: {accuracy_score(y_true, predictions):.3f}")
    print(f"ROC AUC: {roc_auc_score(y_true, probabilities):.3f}")
    print(f"Brier loss: {brier_score_loss(y_true, probabilities):.3f}")


def print_run_totals(report):
    # Compare total predicted runs against the real player total
    actual_total = report["actual_runs"].sum()
    tree_total = report["tree_predicted_runs"].sum()
    tree_upside_total = report["tree_upside_predicted_runs"].sum()
    nn_total = report["nn_predicted_runs"].sum()

    print("Run total comparison")
    print(f"Actual total: {actual_total:.0f}")
    print(f"Tree predicted total: {tree_total:.1f} ({tree_total - actual_total:+.1f})")
    print(
        "Tree upside total: "
        f"{tree_upside_total:.1f} ({tree_upside_total - actual_total:+.1f})"
    )
    print(f"Neural-network total: {nn_total:.1f} ({nn_total - actual_total:+.1f})")


def feature_columns(df):
    # Use every non-target column except row identifiers
    return [
        col
        for col in df.columns
        if not col.startswith("target_") and col not in EXCLUDED_FEATURE_COLUMNS
    ]


def shorten_text(value, max_length):
    # Keep terminal tables readable
    text = str(value)
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."


def printable_report(report):
    # Build a compact dataframe for terminal output
    compact = pd.DataFrame(
        {
            "date": report["date"],
            "opponent": report["bowling_team"].map(lambda value: shorten_text(value, 18)),
            "venue": report["venue"].map(lambda value: shorten_text(value, 28)),
            "bat_pos": report["batting_position"],
            "career_avg": report["player_career_runs_avg"].round(1),
            "vs_opp_avg": report["player_vs_opponent_runs_avg"].round(1),
            "tree_pred": report["tree_predicted_runs"].round(1),
            "tree_up": report["tree_upside_predicted_runs"].round(1),
            "tree_50p": (report["tree_50_plus_probability"] * 100).round(0).astype(int),
            "tree_err": report["tree_absolute_error"].round(1),
            "nn_pred": report["nn_predicted_runs"].round(1),
            "nn_50p": (report["nn_50_plus_probability"] * 100).round(0).astype(int),
            "nn_err": report["nn_absolute_error"].round(1),
            "actual": report["actual_runs"].astype(int),
        }
    )
    return compact[PRINT_COLUMNS]


def build_preprocessor(numeric_features, categorical_features, scale_numeric):
    # Impute missing values, scale numeric values for neural nets, and one-hot categories
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def build_tree_regressor(numeric_features, categorical_features):
    # Standard tree model for expected runs
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=False,
    )
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_tree_upside_regressor(numeric_features, categorical_features):
    # Riskier tree model for a 75th-percentile run estimate
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=False,
    )
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.75,
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_nn_regressor(numeric_features, categorical_features):
    # Neural-network model for expected runs
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=True,
    )
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_tree_classifier(numeric_features, categorical_features):
    # Tree model for 50-plus probability
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=False,
    )
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_nn_classifier(numeric_features, categorical_features):
    # Neural-network model for 50-plus probability
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=True,
    )
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
