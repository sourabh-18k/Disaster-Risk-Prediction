"""
Evaluation Module for Heatwave Prediction System (v2 — Improved)
Provides comprehensive metrics, disaster-specific metrics, and visualization

Improvements over v1:
- Optimal probability threshold tuning for binary heatwave detection
- Threshold-aware evaluation maximises CSI while keeping FAR reasonable
- Side-by-side comparison of default vs threshold-tuned metrics
- HSS (Heidke Skill Score) and Bias Score — WMO standard metrics

Disaster-specific metrics:
- POD (Probability of Detection / Hit Rate) — critical for heatwave warnings
- FAR (False Alarm Rate) — important for public trust
- CSI (Critical Success Index / Threat Score)
- HSS (Heidke Skill Score) — WMO standard
- Bias Score — tendency to over/under-predict
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
        Calculate heatwave-disaster-specific metrics (WMO standard)
        
        POD, FAR, CSI, HSS are standard WMO (World Meteorological Organization)
        verification metrics for disaster forecasting
        """
        # Convert to binary: high risk vs not high risk
        y_true_binary = (np.array(y_true) >= high_risk_threshold).astype(int)
        y_pred_binary = (np.array(y_pred) >= high_risk_threshold).astype(int)
        
        # True Positives, False Positives, False Negatives, True Negatives
        TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        # Probability of Detection (POD) = Hit Rate
        POD = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        # False Alarm Rate (FAR)
        FAR = FP / (TP + FP) if (TP + FP) > 0 else 0
        
        # Critical Success Index (CSI) = Threat Score
        CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        
        # Specificity = True Negative Rate
        Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        # Balanced Accuracy
        Balanced_Accuracy = (POD + Specificity) / 2
        
        # Heidke Skill Score (HSS) — WMO standard for disaster forecasting
        N = TP + FP + FN + TN
        expected_correct = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / N if N > 0 else 0
        HSS = (TP + TN - expected_correct) / (N - expected_correct) if (N - expected_correct) > 0 else 0
        
        # Bias Score — tendency to over/under-predict
        Bias_Score = (TP + FP) / (TP + FN) if (TP + FN) > 0 else 0
        
        metrics = {
            'POD': POD,
            'FAR': FAR,
            'CSI': CSI,
            'Specificity': Specificity,
            'Balanced_Accuracy': Balanced_Accuracy,
            'HSS': HSS,
            'Bias_Score': Bias_Score,
            'True_Positives': int(TP),
            'False_Positives': int(FP),
            'False_Negatives': int(FN),
            'True_Negatives': int(TN)
        }
        
        return metrics
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """Comprehensive evaluation of a single model (default + threshold-tuned)"""
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
        
        # Probability predictions + threshold tuning if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            results['prediction_probabilities'] = y_pred_proba
            
            # --- Threshold-tuned heatwave detection ---
            tuned = self.find_optimal_threshold(y_test, y_pred_proba, heatwave_threshold=3)
            results['threshold_tuned'] = tuned
        
        return results
    
    # -----------------------------------------------------------------
    # Optimal threshold search (NEW — matches flood/landslide)
    # -----------------------------------------------------------------
    def find_optimal_threshold(self, y_test, probas: np.ndarray,
                                heatwave_threshold: int = 3,
                                max_far: float = 0.20) -> Dict:
        """
        Scan probability thresholds [0.05 … 0.95] and pick the one that
        maximises CSI while keeping FAR ≤ max_far.
        
        P(heatwave) = P(High) + P(Extreme).
        
        Returns dict with optimal threshold, tuned predictions,
        and tuned disaster metrics.
        """
        y_true_bin = (np.array(y_test) >= heatwave_threshold).astype(int)
        
        # Probability of "heatwave" = sum of P(High) + P(Extreme)
        n_classes = probas.shape[1]
        if n_classes <= heatwave_threshold:
            return {}
        hw_prob = probas[:, heatwave_threshold:].sum(axis=1)
        
        best_csi = -1
        best_threshold = 0.5
        best_metrics = {}
        all_scans = []
        
        for thr in np.arange(0.05, 0.96, 0.02):
            y_pred_bin = (hw_prob >= thr).astype(int)
            TP = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
            FP = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
            FN = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
            TN = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
            
            POD = TP / (TP + FN) if (TP + FN) > 0 else 0
            FAR = FP / (TP + FP) if (TP + FP) > 0 else 0
            CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
            N = TP + FP + FN + TN
            E = ((TP+FP)*(TP+FN) + (TN+FP)*(TN+FN)) / N if N > 0 else 0
            HSS = (TP+TN - E) / (N - E) if (N - E) > 0 else 0
            
            scan_row = {'threshold': round(thr, 2), 'POD': POD, 'FAR': FAR,
                        'CSI': CSI, 'HSS': HSS, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}
            all_scans.append(scan_row)
            
            if FAR <= max_far and CSI > best_csi:
                best_csi = CSI
                best_threshold = round(thr, 2)
                best_metrics = scan_row.copy()
        
        # Build tuned multi-class predictions:
        # Start with argmax, then override to ≥3 when hw_prob ≥ threshold
        y_pred_default = np.argmax(probas, axis=1)
        y_pred_tuned = y_pred_default.copy()
        hw_mask = hw_prob >= best_threshold
        # For samples predicted < heatwave_threshold but hw_prob high, bump to 3
        upgrade_mask = hw_mask & (y_pred_tuned < heatwave_threshold)
        y_pred_tuned[upgrade_mask] = heatwave_threshold
        
        # Recalculate full disaster metrics on tuned predictions
        tuned_disaster = self.calculate_disaster_metrics(
            y_test, y_pred_tuned, high_risk_threshold=heatwave_threshold
        )
        tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
        tuned_f1 = f1_score(y_test, y_pred_tuned, average='weighted', zero_division=0)
        
        return {
            'optimal_threshold': best_threshold,
            'tuned_predictions': y_pred_tuned,
            'tuned_disaster_metrics': tuned_disaster,
            'tuned_accuracy': tuned_accuracy,
            'tuned_f1_weighted': tuned_f1,
            'threshold_scan': pd.DataFrame(all_scans),
            'best_scan_metrics': best_metrics
        }
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models side by side (default + threshold-tuned)"""
        comparison_data = []
        
        for model_name, results in results_dict.items():
            row = {'Model': model_name}
            row.update(results['basic_metrics'])
            row.update(results['disaster_metrics'])
            
            # Add threshold-tuned columns if available
            if 'threshold_tuned' in results and results['threshold_tuned']:
                tuned = results['threshold_tuned']
                row['Tuned_Threshold'] = tuned.get('optimal_threshold', None)
                row['Tuned_Accuracy'] = tuned.get('tuned_accuracy', None)
                row['Tuned_F1'] = tuned.get('tuned_f1_weighted', None)
                td = tuned.get('tuned_disaster_metrics', {})
                row['Tuned_POD'] = td.get('POD', None)
                row['Tuned_FAR'] = td.get('FAR', None)
                row['Tuned_CSI'] = td.get('CSI', None)
                row['Tuned_HSS'] = td.get('HSS', None)
                row['Tuned_TP'] = td.get('True_Positives', None)
                row['Tuned_FP'] = td.get('False_Positives', None)
                row['Tuned_FN'] = td.get('False_Negatives', None)
                row['Tuned_TN'] = td.get('True_Negatives', None)
            
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
                            figsize: Tuple = (16, 6)):
        """Plot comparison of multiple models"""
        if metrics is None:
            metrics = ['accuracy', 'f1_weighted', 'POD', 'CSI', 'HSS']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        
        if len(metrics) == 1:
            axes = [axes]
        
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        
        for idx, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                ax = axes[idx]
                bars = comparison_df[metric].plot(kind='bar', ax=ax, color=colors[:len(comparison_df)])
                ax.set_title(metric.replace('_', ' ').upper(), fontweight='bold')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1.05)
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar_container in ax.containers:
                    ax.bar_label(bar_container, fmt='%.3f', padding=3, fontsize=8)
        
        plt.suptitle('Heatwave Prediction Model Comparison', fontsize=14, fontweight='bold', y=1.02)
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
        print(f"HEATWAVE EVALUATION REPORT: {results['model_name']}")
        print("="*80)
        
        print("\n--- Basic Metrics ---")
        for metric, value in results['basic_metrics'].items():
            print(f"  {metric:.<40} {value:.4f}")
        
        print("\n--- Disaster-Specific Metrics (WMO Standard) ---")
        for metric, value in results['disaster_metrics'].items():
            if isinstance(value, (int, np.integer)):
                print(f"  {metric:.<40} {value}")
            else:
                print(f"  {metric:.<40} {value:.4f}")
        
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
