"""
Models Module for Heatwave Prediction System
Implements machine learning and deep learning models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')


class HeatwaveMLModels:
    """
    Machine Learning Models for Heatwave Prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'risk_level',
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                       pd.Series, pd.Series]:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame with features and target
            target_col: Name of target column
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Select features (exclude target and metadata columns)
        exclude_cols = [
            target_col, 'is_heatwave', 'city', 'is_hot_day', 'is_severe_hot_day',
            'heatwave_candidate', 'severe_heatwave_candidate'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any remaining non-numeric columns
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Time-based split (using last test_size fraction as test set)
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
        print(f"\nTarget distribution (train):")
        print(y_train.value_counts().sort_index())
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, **kwargs):
        """Train Random Forest model"""
        print("\nTraining Random Forest...")
        
        default_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'class_weight': 'balanced',
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
        
        print(f"  Model trained. OOB Score: {model.oob_score_ if hasattr(model, 'oob_score_') else 'N/A'}")
        return model
    
    def train_xgboost(self, X_train, y_train, **kwargs):
        """Train XGBoost model"""
        print("\nTraining XGBoost...")
        
        default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_jobs': -1,
            'eval_metric': 'mlogloss'
        }
        default_params.update(kwargs)
        
        model = xgb.XGBClassifier(**default_params)
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("  Model trained successfully")
        return model
    
    def train_lightgbm(self, X_train, y_train, **kwargs):
        """Train LightGBM model"""
        print("\nTraining LightGBM...")
        
        default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_jobs': -1,
            'class_weight': 'balanced'
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
        """Train Gradient Boosting model"""
        print("\nTraining Gradient Boosting...")
        
        default_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'random_state': self.random_state
        }
        default_params.update(kwargs)
        
        model = GradientBoostingClassifier(**default_params)
        model.fit(X_train, y_train)
        
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
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*80)
        
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        print("\nAll models trained successfully!")
        print(f"Total models: {len(self.models)}")
    
    def predict(self, model_name: str, X) -> np.ndarray:
        """Make predictions with a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        return self.models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X) -> np.ndarray:
        """Get prediction probabilities"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict_proba(X)
    
    def get_feature_importance(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features"""
        if model_name not in self.feature_importance:
            raise ValueError(f"Feature importance not available for {model_name}")
        
        return self.feature_importance[model_name].head(top_n)
    
    def save_models(self, output_dir: str = "models/ml/"):
        """Save all trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(output_dir, f"{name}.pkl")
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")
        
        # Save scaler
        if self.scalers:
            scaler_path = os.path.join(output_dir, "scaler.pkl")
            joblib.dump(self.scalers['standard'], scaler_path)
            print(f"Saved scaler to {scaler_path}")
    
    def load_models(self, model_dir: str = "models/ml/"):
        """Load trained models"""
        import os
        
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl') and filename != 'scaler.pkl':
                model_name = filename.replace('.pkl', '')
                filepath = os.path.join(model_dir, filename)
                self.models[model_name] = joblib.load(filepath)
                print(f"Loaded {model_name} from {filepath}")
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scalers['standard'] = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")


class EnsembleModel:
    """
    Ensemble model combining predictions from multiple models
    """
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        """
        Initialize ensemble
        
        Args:
            models: Dictionary of model_name -> model object
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights or {name: 1.0 / len(models) for name in models.keys()}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def predict_proba(self, X) -> np.ndarray:
        """Get ensemble prediction probabilities"""
        all_probas = []
        
        for name, model in self.models.items():
            probas = model.predict_proba(X)
            weighted_probas = probas * self.weights[name]
            all_probas.append(weighted_probas)
        
        # Average weighted probabilities
        ensemble_probas = np.sum(all_probas, axis=0)
        return ensemble_probas
    
    def predict(self, X) -> np.ndarray:
        """Get ensemble predictions"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


def create_ensemble_model(ml_models: HeatwaveMLModels,
                         weights: Dict[str, float] = None) -> EnsembleModel:
    """
    Create an ensemble model from trained ML models
    
    Args:
        ml_models: HeatwaveMLModels object with trained models
        weights: Optional custom weights for each model
        
    Returns:
        EnsembleModel instance
    """
    return EnsembleModel(ml_models.models, weights)


if __name__ == "__main__":
    print("Models module loaded successfully")
    print("Available model classes:")
    print("  - HeatwaveMLModels: Train and predict with ML models")
    print("  - EnsembleModel: Combine multiple models")
