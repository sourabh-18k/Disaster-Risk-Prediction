"""
Machine Learning Models for Landslide Risk Prediction (v1 — with SMOTE & class weights)

Implements four models following the same architecture as our flood module:
  1. Random Forest         — ensemble of decision trees with class weights
  2. XGBoost               — gradient boosting with per-sample weights
  3. LightGBM              — fast gradient boosting with class weights
  4. Gradient Boosting     — sklearn GBM with per-sample weights

All models include:
  - SMOTE oversampling for minority classes (High, Extreme risk)
  - Aggressive class weights favouring detection of landslide events
  - Optimised hyperparameters for imbalanced geospatial data
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Custom class weights for landslide risk (same philosophy as flood v2)
# =============================================================================
LANDSLIDE_CLASS_WEIGHTS = {
    0: 1.0,    # No Risk
    1: 2.0,    # Low
    2: 4.0,    # Moderate
    3: 12.0,   # High — must not miss
    4: 15.0,   # Extreme — critical
}


def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights from LANDSLIDE_CLASS_WEIGHTS (for XGBoost/GBM)."""
    return np.array([LANDSLIDE_CLASS_WEIGHTS.get(int(c), 1.0) for c in y])


class LandslideMLModels:
    """ML Models for Landslide Risk Prediction with SMOTE + class weights."""

    def __init__(self, random_state: int = 42, use_smote: bool = True):
        self.random_state = random_state
        self.use_smote = use_smote
        self.models: Dict = {}
        self.scalers: Dict = {}
        self.feature_importance: Dict = {}
        self.feature_names: list = []
        self.optimal_thresholds: Dict = {}

    # -----------------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'risk_level',
                     test_size: float = 0.2
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data with time-based split + optional SMOTE."""
        exclude_cols = [
            target_col, 'is_landslide', 'region', 'hazard_score',
            'rain_3d_sum (mm)', 'rain_7d_sum (mm)', 'rain_14d_sum (mm)',
        ]
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Time-based split
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        self.scalers['standard'] = scaler
        self.feature_names = feature_cols

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(feature_cols)}")
        print(f"\nTarget distribution (train — BEFORE resampling):")
        names = {0: 'No Risk', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Extreme'}
        for level, count in y_train.value_counts().sort_index().items():
            pct = count / len(y_train) * 100
            print(f"  {names.get(level, level)}: {count} ({pct:.1f}%)")

        # --- SMOTE ---
        if self.use_smote:
            print("\nApplying SMOTE oversampling to training data...")
            majority_count = y_train.value_counts().max()
            target_count = max(int(majority_count * 0.35), 500)
            sampling_strategy = {}
            for cls, cnt in y_train.value_counts().items():
                if cnt < target_count:
                    sampling_strategy[cls] = target_count

            if sampling_strategy:
                min_class_count = y_train.value_counts().min()
                k = min(5, max(1, min_class_count - 1))
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                    k_neighbors=k
                )
                X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
                X_train_scaled = pd.DataFrame(
                    X_resampled, columns=X_train.columns
                )
                y_train = pd.Series(y_resampled, name=target_col)
                print(f"  Resampled training set: {len(X_train_scaled)} samples")
                print(f"\n  Target distribution (train — AFTER SMOTE):")
                for level, count in y_train.value_counts().sort_index().items():
                    pct = count / len(y_train) * 100
                    print(f"    {names.get(level, level)}: {count} ({pct:.1f}%)")

        return X_train_scaled, X_test_scaled, y_train, y_test

    # -----------------------------------------------------------------
    def _get_models(self) -> Dict:
        """Return configured model instances."""
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=500, max_depth=25, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt',
                class_weight=LANDSLIDE_CLASS_WEIGHTS,
                random_state=self.random_state, n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=400, max_depth=12, learning_rate=0.05,
                min_child_weight=3, gamma=0.1, reg_alpha=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=-1,
                use_label_encoder=False, eval_metric='mlogloss', verbosity=0
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=400, max_depth=12, num_leaves=63,
                learning_rate=0.05, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                class_weight=LANDSLIDE_CLASS_WEIGHTS,
                random_state=self.random_state, n_jobs=-1, verbose=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=80, max_depth=5, learning_rate=0.1,
                min_samples_split=20, min_samples_leaf=10, subsample=0.8,
                max_features='sqrt',
                random_state=self.random_state
            ),
        }

    # -----------------------------------------------------------------
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train all four models."""
        models = self._get_models()
        sample_weights = _compute_sample_weights(y_train.values)

        print("=" * 80)
        print("TRAINING LANDSLIDE PREDICTION MODELS")
        print("=" * 80)

        for name, model in models.items():
            print(f"\nTraining {name}...")
            try:
                if name in ('xgboost', 'gradient_boosting'):
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    model.fit(X_train, y_train)
                self.models[name] = model
                print(f"  Model trained successfully")

                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(
                        zip(self.feature_names, model.feature_importances_)
                    )
            except Exception as e:
                print(f"  ERROR training {name}: {e}")

        print(f"\nAll {len(self.models)} models trained successfully!")

    # -----------------------------------------------------------------
    def get_feature_importance(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for a model."""
        if model_name not in self.feature_importance:
            return pd.DataFrame()
        fi = self.feature_importance[model_name]
        fi_df = pd.DataFrame([
            {'feature': feat, 'importance': imp} for feat, imp in fi.items()
        ]).sort_values('importance', ascending=False).head(top_n)
        return fi_df

    # -----------------------------------------------------------------
    def save_models(self, save_dir: str):
        """Save trained models and scaler."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            fp = save_path / f'{name}.pkl'
            joblib.dump(model, fp)
            print(f"Saved {name} to {fp}")
        if 'standard' in self.scalers:
            joblib.dump(self.scalers['standard'], save_path / 'scaler.pkl')
            print(f"Saved scaler to {save_path / 'scaler.pkl'}")

    def load_models(self, load_dir: str):
        """Load saved models."""
        load_path = Path(load_dir)
        for pkl in load_path.glob('*.pkl'):
            if pkl.stem == 'scaler':
                self.scalers['standard'] = joblib.load(pkl)
            else:
                self.models[pkl.stem] = joblib.load(pkl)
        print(f"Loaded {len(self.models)} models from {load_dir}")
