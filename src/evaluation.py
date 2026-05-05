"""
Evaluation Module for Heatwave Prediction System
Provides comprehensive metrics and visualization for model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class HeatwaveEvaluator:
    """
    Comprehensive evaluation metrics for heatwave prediction models
    """
    
    def __init__(self):
        self.results = {}
        
    def calculate_basic_metrics(self, y_true, y_pred, model_name: str) -> Dict:
        """Calculate basic classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def calculate_disaster_metrics(self, y_true, y_pred, high_risk_threshold: int = 3) -> Dict:
        """
        Calculate disaster-specific metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            high_risk_threshold: Risk level threshold for high risk (3 = High Risk, 4 = Extreme)
            
        Returns:
            Dictionary of disaster metrics
        """
        # Convert to binary: high risk vs not high risk
        y_true_binary = (y_true >= high_risk_threshold).astype(int)
        y_pred_binary = (y_pred >= high_risk_threshold).astype(int)
        
        # True Positives, False Positives, False Negatives, True Negatives
        TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        # Probability of Detection (POD) = Hit Rate = Recall for high-risk class
        POD = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        # False Alarm Rate (FAR)
        FAR = FP / (TP + FP) if (TP + FP) > 0 else 0
        
        # Critical Success Index (CSI) = Threat Score
        CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        
        # Specificity = True Negative Rate
        Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        # Balanced Accuracy
        Sensitivity = POD
        Balanced_Accuracy = (Sensitivity + Specificity) / 2
        
        metrics = {
            'POD': POD,
            'FAR': FAR,
            'CSI': CSI,
            'Specificity': Specificity,
            'Balanced_Accuracy': Balanced_Accuracy,
            'True_Positives': int(TP),
            'False_Positives': int(FP),
            'False_Negatives': int(FN),
            'True_Negatives': int(TN)
        }
        
        return metrics
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """
        Comprehensive evaluation of a single model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True labels
            model_name: Name of the model
            
        Returns:
            Dictionary of all metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        
        # Basic metrics
        basic_metrics = self.calculate_basic_metrics(y_test, y_pred, model_name)
        
        # Disaster-specific metrics
        disaster_metrics = self.calculate_disaster_metrics(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class metrics
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        results = {
            'model_name': model_name,
            'basic_metrics': basic_metrics,
            'disaster_metrics': disaster_metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_pred
        }
        
        # Probability predictions if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            results['prediction_probabilities'] = y_pred_proba
        
        return results
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            results_dict: Dictionary of model_name -> evaluation results
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for model_name, results in results_dict.items():
            row = {'Model': model_name}
            row.update(results['basic_metrics'])
            row.update(results['disaster_metrics'])
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Model')
        
        return comparison_df
    
    def plot_confusion_matrix(self, cm, model_name: str, 
                             risk_labels: List[str] = None,
                             figsize: Tuple = (10, 8)):
        """Plot confusion matrix heatmap"""
        if risk_labels is None:
            risk_labels = ['No Risk', 'Low', 'Moderate', 'High', 'Extreme']
        
        # Trim labels if necessary
        n_classes = cm.shape[0]
        risk_labels = risk_labels[:n_classes]
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=risk_labels,
                   yticklabels=risk_labels,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Risk Level', fontsize=12)
        plt.xlabel('Predicted Risk Level', fontsize=12)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                            metrics: List[str] = None,
                            figsize: Tuple = (14, 6)):
        """Plot comparison of multiple models"""
        if metrics is None:
            metrics = ['accuracy', 'f1_weighted', 'POD', 'CSI']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                ax = axes[idx]
                comparison_df[metric].plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
                ax.set_ylabel('Score')
                ax.set_xlabel('Model')
                ax.set_ylim(0, 1)
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self, results_dict: Dict[str, Dict], y_test,
                       risk_level: int = 3, figsize: Tuple = (10, 8)):
        """
        Plot ROC curves for multiple models
        
        Args:
            results_dict: Dictionary of model results
            y_test: True labels
            risk_level: Risk level to evaluate (binary: >= risk_level)
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Convert to binary classification
        y_test_binary = (y_test >= risk_level).astype(int)
        
        for model_name, results in results_dict.items():
            if 'prediction_probabilities' in results:
                y_pred_proba = results['prediction_probabilities']
                
                # Get probability for high risk classes
                y_pred_proba_high = y_pred_proba[:, risk_level:].sum(axis=1)
                
                fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba_high)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - High Risk Detection (≥ Level {risk_level})', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_feature_importance(self, feature_importance_df: pd.DataFrame,
                               top_n: int = 20, figsize: Tuple = (10, 8)):
        """Plot feature importance"""
        plt.figure(figsize=figsize)
        
        top_features = feature_importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def print_evaluation_report(self, results: Dict):
        """Print formatted evaluation report"""
        print("="*80)
        print(f"EVALUATION REPORT: {results['model_name']}")
        print("="*80)
        
        print("\n--- Basic Metrics ---")
        for metric, value in results['basic_metrics'].items():
            print(f"{metric:.<40} {value:.4f}")
        
        print("\n--- Disaster-Specific Metrics ---")
        for metric, value in results['disaster_metrics'].items():
            if isinstance(value, (int, np.integer)):
                print(f"{metric:.<40} {value}")
            else:
                print(f"{metric:.<40} {value:.4f}")
        
        print("\n--- Per-Class Performance ---")
        class_report = results['classification_report']
        risk_names = {
            '0': 'No Risk',
            '1': 'Low Risk',
            '2': 'Moderate Risk',
            '3': 'High Risk',
            '4': 'Extreme Risk'
        }
        
        for class_label, metrics in class_report.items():
            if class_label in risk_names:
                print(f"\n{risk_names[class_label]}:")
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if metric != 'support':
                            print(f"  {metric:.<35} {value:.4f}")
                        else:
                            print(f"  {metric:.<35} {int(value)}")


def evaluate_all_models(ml_models, X_test, y_test) -> Dict[str, Dict]:
    """
    Evaluate all ML models
    
    Args:
        ml_models: HeatwaveMLModels object with trained models
        X_test: Test features
        y_test: True labels
        
    Returns:
        Dictionary of model_name -> evaluation results
    """
    evaluator = HeatwaveEvaluator()
    results = {}
    
    print("="*80)
    print("EVALUATING MODELS")
    print("="*80)
    
    for model_name in ml_models.models.keys():
        print(f"\nEvaluating {model_name}...")
        model = ml_models.models[model_name]
        results[model_name] = evaluator.evaluate_model(model, X_test, y_test, model_name)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    print("Evaluation module loaded successfully")
    print("Use HeatwaveEvaluator class for comprehensive model evaluation")
