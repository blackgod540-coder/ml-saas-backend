import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)


class MLEngine:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        self.dropped_columns = []  # Track noise/ID columns automatically removed
        self.feature_defaults = {}
        self.model_type = None

    def clean_and_preprocess(self):
        # 1. Drop rows missing the target variable
        self.df = self.df.dropna(subset=[self.target_column]).copy()

        # 2. Filter noise columns (IDs, names, hashes, unique strings > 50%)
        cols_to_drop = []
        total_rows = len(self.df)
        if total_rows > 0:
            for col in self.df.columns:
                if col == self.target_column:
                    continue

                if self.df[col].dtype == 'object' or isinstance(self.df[col].dtype, pd.CategoricalDtype):
                    unique_ratio = self.df[col].nunique() / total_rows
                    if unique_ratio > 0.5:
                        cols_to_drop.append(col)

        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self.dropped_columns = cols_to_drop

        # 3. Store defaults for metadata/dashboard fallbacks
        for col in self.df.columns:
            if col == self.target_column:
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                med = self.df[col].median()
                self.feature_defaults[col] = float(med) if not pd.isna(med) else 0.0
            else:
                mode_s = self.df[col].mode()
                self.feature_defaults[col] = str(mode_s[0]) if not mode_s.empty else "Unknown"

        # 4. Determine task type
        target_series = self.df[self.target_column]
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 10:
            self.model_type = "regression"
        else:
            self.model_type = "classification"

    def train_and_save(self, model_id: str):
        self.clean_and_preprocess()

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # Categorize columns for pipeline routing
        numeric_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        # Robust Preprocessing Transformers
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, numeric_cols),
                ('cat', cat_pipeline, categorical_cols)
            ]
        )

        # Select Gradient Boosting estimator
        if self.model_type == "classification":
            estimator = HistGradientBoostingClassifier(random_state=42, max_iter=150)
        else:
            estimator = HistGradientBoostingRegressor(random_state=42, max_iter=150)

        # Full end-to-end sklearn Pipeline
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', estimator)
        ])

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit Pipeline
        full_pipeline.fit(X_train, y_train)
        preds = full_pipeline.predict(X_test)

        # Evaluate performance
        metrics = {}
        if self.model_type == "classification":
            metrics = {
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "f1_score": round(float(f1_score(y_test, preds, average='weighted', zero_division=0)), 4)
            }
        else:
            metrics = {
                "r2_score": round(float(r2_score(y_test, preds)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 2),
                "mae": round(float(mean_absolute_error(y_test, preds)), 2)
            }

        # EXPORT THE PIPELINE DIRECTLY
        # Saving full_pipeline ensures joblib.load("model.joblib").predict(df) works natively
        file_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
        joblib.dump(full_pipeline, file_path)

        return {
            "model_id": model_id,
            "model_type": self.model_type,
            "performance_metrics": metrics,
            "features_used": list(X.columns),
            "noise_columns_dropped": self.dropped_columns,
            "feature_defaults": self.feature_defaults
        }