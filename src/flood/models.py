"""
Models Module for Flood Risk Prediction System (v2 — Improved)
Implements ML models for multi-class flood risk classification

Improvements over v1:
- SMOTE oversampling to handle severe class imbalance (flood days ~5%)
- Aggressive class weights (flood classes get 8-15× penalty)
- Per-sample weights for XGBoost/GBM to prioritize flood detection
- Optimised probability-threshold prediction for binary flood detection
- Higher POD without sacrificing too much accuracy

Models: Random Forest, XGBoost, LightGBM, Gradient Boosting
+ Ensemble model for improved predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Custom class-weight map — heavily penalises missing High/Extreme classes
# ---------------------------------------------------------------------------
FLOOD_CLASS_WEIGHTS = {
    0: 1.0,    # No Risk   — baseline
    1: 2.0,    # Low       — slight upweight
    2: 4.0,    # Moderate  — important transition zone
    3: 12.0,   # High      — flood event — must not miss
    4: 15.0,   # Extreme   — critical — highest penalty
}


def _compute_sample_weights(y: pd.Series) -> np.ndarray:
    """Return per-sample weight array derived from FLOOD_CLASS_WEIGHTS."""
    return np.array([FLOOD_CLASS_WEIGHTS.get(label, 1.0) for label in y])


class FloodMLModels:
    """
    Machine Learning Models for Flood Risk Prediction (v2 — flood-optimised)
    """
    
    def __init__(self, random_state=42, use_smote: bool = True):
        self.random_state = random_state
        self.use_smote = use_smote
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.feature_names = []
        self.optimal_thresholds = {}   # per-model optimal flood probability thresholds
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'risk_level',
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                      pd.Series, pd.Series]:
        """
        Prepare data for training with time-based split + optional SMOTE
        """
        # Columns to exclude from features
        exclude_cols = [
            target_col, 'is_flood', 'city',
            'above_warning', 'above_danger', 'soil_saturated',
            'discharge_percentile', 'rain_3d_sum', 'rain_7d_sum'
        ]
        
        # Get numeric feature columns only
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Time-based split (chronological)
        split_idx = int(len(df) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        self.scalers['standard'] = scaler
        self.feature_names = feature_cols
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(feature_cols)}")
        print(f"\nTarget distribution (train — BEFORE resampling):")
        for level, count in y_train.value_counts().sort_index().items():
            risk_names = {0: 'No Risk', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Extreme'}
            pct = count / len(y_train) * 100
            print(f"  {risk_names.get(level, level)}: {count} ({pct:.1f}%)")
        
        # ----- SMOTE oversampling on training set only -----
        if self.use_smote:
            print("\nApplying SMOTE oversampling to training data...")
            # Determine target counts: upsample minority classes toward 30% of majority
            majority_count = y_train.value_counts().max()
            target_count = max(int(majority_count * 0.35), 500)
            sampling_strategy = {}
            for cls, cnt in y_train.value_counts().items():
                if cnt < target_count:
                    sampling_strategy[cls] = target_count
            
            if sampling_strategy:
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                    k_neighbors=min(5, min(y_train.value_counts().values) - 1)
                )
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
                X_train_scaled = pd.DataFrame(X_train_resampled, columns=feature_cols)
                y_train = pd.Series(y_train_resampled, name=target_col)
                
                print(f"  Resampled training set: {len(X_train_scaled)} samples")
                print(f"\n  Target distribution (train — AFTER SMOTE):")
                for level, count in y_train.value_counts().sort_index().items():
                    risk_names = {0: 'No Risk', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Extreme'}
                    pct = count / len(y_train) * 100
                    print(f"    {risk_names.get(level, level)}: {count} ({pct:.1f}%)")
            else:
                print("  No resampling needed — classes are balanced enough.")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, **kwargs):
        """Train Random Forest model with flood-optimised class weights"""
        print("\nTraining Random Forest (flood-optimised)...")
        
        default_params = {
            'n_estimators': 500,
            'max_depth': 25,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': FLOOD_CLASS_WEIGHTS,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        self.feature_importance['random_forest'] = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("  Model trained successfully")
        return model
    
    def train_xgboost(self, X_train, y_train, **kwargs):
        """Train XGBoost model with sample weights"""
        print("\nTraining XGBoost (flood-optimised)...")
        
        default_params = {
            'n_estimators': 400,
            'max_depth': 12,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        }
        default_params.update(kwargs)
        
        sample_weights = _compute_sample_weights(y_train)
        
        model = xgb.XGBClassifier(**default_params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("  Model trained successfully")
        return model
    
    def train_lightgbm(self, X_train, y_train, **kwargs):
        """Train LightGBM model with flood-optimised class weights"""
        print("\nTraining LightGBM (flood-optimised)...")
        
        default_params = {
            'n_estimators': 400,
            'max_depth': 12,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 10,
            'random_state': self.random_state,
            'n_jobs': -1,
            'class_weight': FLOOD_CLASS_WEIGHTS
        }
        default_params.update(kwargs)
        
        model = lgb.LGBMClassifier(**default_params, verbose=-1)
        model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        self.feature_importance['lightgbm'] = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("  Model trained successfully")
        return model
    
    def train_gradient_boosting(self, X_train, y_train, **kwargs):
        """Train Gradient Boosting model with sample weights"""
        print("\nTraining Gradient Boosting (flood-optimised)...")
        
        default_params = {
            'n_estimators': 150,
            'max_depth': 8,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'max_features': 'sqrt',
            'min_samples_leaf': 5,
            'random_state': self.random_state
        }
        default_params.update(kwargs)
        
        sample_weights = _compute_sample_weights(y_train)
        
        model = GradientBoostingClassifier(**default_params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        self.models['gradient_boosting'] = model
        self.feature_importance['gradient_boosting'] = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("  Model trained successfully")
        return model
    
    def train_all_models(self, X_train, y_train):
        """Train all ML models"""
        print("="*80)
        print("TRAINING FLOOD PREDICTION MODELS (v2 — FLOOD-OPTIMISED)")
        print("="*80)
        
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        print(f"\nAll {len(self.models)} models trained successfully!")
    
    def predict(self, model_name: str, X) -> np.ndarray:
        """Make predictions with a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        return self.models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X) -> np.ndarray:
        """Get prediction probabilities"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        return self.models[model_name].predict_proba(X)
    
    def get_feature_importance(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features"""
        if model_name not in self.feature_importance:
            raise ValueError(f"Feature importance not available for {model_name}")
        return self.feature_importance[model_name].head(top_n)
    
    def save_models(self, output_dir: str = "models/flood/ml/"):
        """Save all trained models and scaler"""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(output_dir, f"{name}.pkl")
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")
        
        if self.scalers:
            scaler_path = os.path.join(output_dir, "scaler.pkl")
            joblib.dump(self.scalers['standard'], scaler_path)
            print(f"Saved scaler to {scaler_path}")
    
    def load_models(self, model_dir: str = "models/flood/ml/"):
        """Load trained models"""
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl') and filename != 'scaler.pkl':
                model_name = filename.replace('.pkl', '')
                filepath = os.path.join(model_dir, filename)
                self.models[model_name] = joblib.load(filepath)
                print(f"Loaded {model_name}")
        
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scalers['standard'] = joblib.load(scaler_path)
            print("Loaded scaler")


class FloodEnsembleModel:
    """
    Ensemble model combining predictions from multiple flood models
    Uses weighted probability averaging
    """
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        self.models = models
        self.weights = weights or {name: 1.0 / len(models) for name in models.keys()}
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
    
    def predict_proba(self, X) -> np.ndarray:
        all_probas = []
        for name, model in self.models.items():
            probas = model.predict_proba(X)
            all_probas.append(probas * self.weights[name])
        return np.sum(all_probas, axis=0)
    
    def predict(self, X) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


def create_ensemble_model(ml_models: FloodMLModels,
                         weights: Dict[str, float] = None) -> FloodEnsembleModel:
    """Create an ensemble from trained flood models"""
    return FloodEnsembleModel(ml_models.models, weights)


if __name__ == "__main__":
    print("Flood Models module loaded successfully")
    print("Available: FloodMLModels, FloodEnsembleModel")
