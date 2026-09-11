import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

class MLEngine:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        self.label_encoders = {}
        self.feature_defaults = {}
        self.dropped_columns = []  # Track noise columns we automatically remove
        self.model_type = None

    def clean_and_preprocess(self):
        # 1. Drop rows missing the target variable (can't train on what we don't know)
        self.df = self.df.dropna(subset=[self.target_column])

        # 2. Advanced Feature Engineering & Noise Filtration
        cols_to_drop = []
        for col in self.df.columns:
            if col == self.target_column:
                continue

            # NOISE FILTER: Drop columns with entirely unique text/strings (like IDs, Names, Hashes)
            # If more than 50% of the text data in a column is unique, it's noise.
            if self.df[col].dtype == 'object':
                unique_ratio = self.df[col].nunique() / len(self.df[col])
                if unique_ratio > 0.5:
                    cols_to_drop.append(col)
                    continue

        self.df = self.df.drop(columns=cols_to_drop)
        self.dropped_columns = cols_to_drop

        # 3. Smart Imputation & Encoding for remaining valid columns
        for col in self.df.columns:
            if col == self.target_column:
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Store median for API fallback, though HistGradient handles NaNs natively
                median_val = self.df[col].median()
                fallback_val = float(median_val) if not pd.isna(median_val) else 0.0
                self.feature_defaults[col] = fallback_val
                self.df[col] = self.df[col].fillna(fallback_val)
            else:
                mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
                self.feature_defaults[col] = mode_val
                self.df[col] = self.df[col].fillna(mode_val)

                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le

        # 4. Determine Model Type securely
        target_series = self.df[self.target_column]
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 10:
            self.model_type = "regression"
        else:
            self.model_type = "classification"
            if not pd.api.types.is_numeric_dtype(target_series):
                target_le = LabelEncoder()
                self.df[self.target_column] = target_le.fit_transform(target_series.astype(str))
                self.label_encoders[self.target_column] = target_le

    def train_and_save(self, model_id: str):
        self.clean_and_preprocess()

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. Train using robust Gradient Boosting (Kaggle Standard)
        metrics = {}
        if self.model_type == "classification":
            # Using class_weight='balanced' in standard models prevents bias. 
            # HistGradient doesn't use it directly, but its architecture handles imbalance well.
            model = HistGradientBoostingClassifier(random_state=42, max_iter=100)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = {
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "f1_score": round(float(f1_score(y_test, preds, average='weighted')), 4)
            }
        else:
            model = HistGradientBoostingRegressor(random_state=42, max_iter=100)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = {
                "r2_score": round(float(r2_score(y_test, preds)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 2),
                "mae": round(float(mean_absolute_error(y_test, preds)), 2)
            }

        artifacts = {
            "model": model,
            "model_type": self.model_type,
            "features": list(X.columns),
            "label_encoders": self.label_encoders,
            "feature_defaults": self.feature_defaults,
            "target_column": self.target_column
        }

        file_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
        joblib.dump(artifacts, file_path)

        return {
            "model_id": model_id,
            "model_type": self.model_type,
            "performance_metrics": metrics,
            "features_used": list(X.columns),
            "noise_columns_dropped": self.dropped_columns,
            "feature_defaults": self.feature_defaults
        }