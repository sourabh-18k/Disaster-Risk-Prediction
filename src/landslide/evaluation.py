"""
Evaluation Module for Landslide Risk Prediction

Implements WMO-style disaster verification metrics + probability threshold tuning.
Same architecture as the flood evaluation module (v2).

Metrics:
  - Standard ML: Accuracy, F1 (weighted), Precision, Recall, Balanced Accuracy
  - Disaster verification: POD, FAR, CSI, HSS, Bias Score
  - Threshold-tuned binary detection: P(landslide) vs optimal threshold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')


class LandslideEvaluator:
    """Evaluation engine for landslide risk prediction models."""

    def __init__(self, high_risk_threshold: int = 3):
        """
        Args:
            high_risk_threshold: Risk level at or above which we consider
                                 a day as a "landslide event" for binary POD/FAR.
                                 3 = High, 4 = Extreme
        """
        self.high_risk_threshold = high_risk_threshold

    # -----------------------------------------------------------------
    # Core metrics
    # -----------------------------------------------------------------
    def calculate_basic_metrics(self, y_true, y_pred, model_name: str = '') -> Dict:
        return {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        }

    def calculate_disaster_metrics(self, y_true, y_pred,
                                    high_risk_threshold: Optional[int] = None) -> Dict:
        """Binary disaster metrics: POD, FAR, CSI, HSS."""
        thr = high_risk_threshold or self.high_risk_threshold
        y_true_bin = (np.array(y_true) >= thr).astype(int)
        y_pred_bin = (np.array(y_pred) >= thr).astype(int)

        TP = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        FP = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
        FN = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
        TN = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))

        POD = TP / (TP + FN) if (TP + FN) > 0 else 0
        FAR = FP / (TP + FP) if (TP + FP) > 0 else 0
        CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        N = TP + FP + FN + TN
        E = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / N if N > 0 else 0
        HSS = (TP + TN - E) / (N - E) if (N - E) > 0 else 0
        BIAS = (TP + FP) / (TP + FN) if (TP + FN) > 0 else 0

        return {
            'POD': round(POD, 4), 'FAR': round(FAR, 4),
            'CSI': round(CSI, 4), 'HSS': round(HSS, 4),
            'Bias_Score': round(BIAS, 4),
            'True_Positives': TP, 'False_Positives': FP,
            'False_Negatives': FN, 'True_Negatives': TN,
        }

    # -----------------------------------------------------------------
    # Model evaluation (default + threshold-tuned)
    # -----------------------------------------------------------------
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        y_pred = model.predict(X_test)

        basic = self.calculate_basic_metrics(y_test, y_pred, model_name)
        disaster = self.calculate_disaster_metrics(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results = {
            'model_name': model_name,
            'basic_metrics': basic,
            'disaster_metrics': disaster,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
        }

        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_test)
            results['prediction_probabilities'] = probas
            tuned = self.find_optimal_threshold(y_test, probas, flood_threshold=3)
            results['threshold_tuned'] = tuned

        return results

    # -----------------------------------------------------------------
    # Threshold tuning
    # -----------------------------------------------------------------
    def find_optimal_threshold(self, y_test, probas: np.ndarray,
                                flood_threshold: int = 3,
                                max_far: float = 0.20) -> Dict:
        """
        Scan P(landslide) thresholds to maximise CSI with FAR ≤ max_far.
        P(landslide) = P(High) + P(Extreme).
        """
        y_true_bin = (np.array(y_test) >= flood_threshold).astype(int)
        n_classes = probas.shape[1]
        if n_classes <= flood_threshold:
            return {}
        ls_prob = probas[:, flood_threshold:].sum(axis=1)

        best_csi, best_thr, best_metrics = -1, 0.5, {}
        all_scans = []

        for thr in np.arange(0.05, 0.96, 0.02):
            yp = (ls_prob >= thr).astype(int)
            TP = int(((y_true_bin == 1) & (yp == 1)).sum())
            FP = int(((y_true_bin == 0) & (yp == 1)).sum())
            FN = int(((y_true_bin == 1) & (yp == 0)).sum())
            TN = int(((y_true_bin == 0) & (yp == 0)).sum())

            POD = TP / (TP + FN) if (TP + FN) else 0
            FAR = FP / (TP + FP) if (TP + FP) else 0
            CSI = TP / (TP + FP + FN) if (TP + FP + FN) else 0
            N = TP + FP + FN + TN
            E = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / N if N else 0
            HSS = (TP + TN - E) / (N - E) if (N - E) else 0

            row = {'threshold': round(thr, 2), 'POD': POD, 'FAR': FAR,
                   'CSI': CSI, 'HSS': HSS, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}
            all_scans.append(row)

            if FAR <= max_far and CSI > best_csi:
                best_csi = CSI
                best_thr = round(thr, 2)
                best_metrics = row.copy()

        # Build tuned multi-class predictions
        y_pred_default = np.argmax(probas, axis=1)
        y_pred_tuned = y_pred_default.copy()
        upgrade_mask = (ls_prob >= best_thr) & (y_pred_tuned < flood_threshold)
        y_pred_tuned[upgrade_mask] = flood_threshold

        tuned_disaster = self.calculate_disaster_metrics(y_test, y_pred_tuned)
        tuned_acc = accuracy_score(y_test, y_pred_tuned)
        tuned_f1 = f1_score(y_test, y_pred_tuned, average='weighted', zero_division=0)

        return {
            'optimal_threshold': best_thr,
            'tuned_predictions': y_pred_tuned,
            'tuned_disaster_metrics': tuned_disaster,
            'tuned_accuracy': tuned_acc,
            'tuned_f1_weighted': tuned_f1,
            'threshold_scan': pd.DataFrame(all_scans),
            'best_scan_metrics': best_metrics,
        }

    # -----------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        rows = []
        for model_name, res in results_dict.items():
            row = {'Model': model_name}
            row.update(res['basic_metrics'])
            row.update(res['disaster_metrics'])
            if 'threshold_tuned' in res and res['threshold_tuned']:
                t = res['threshold_tuned']
                row['Tuned_Threshold'] = t.get('optimal_threshold')
                row['Tuned_Accuracy'] = t.get('tuned_accuracy')
                row['Tuned_F1'] = t.get('tuned_f1_weighted')
                td = t.get('tuned_disaster_metrics', {})
                row['Tuned_POD'] = td.get('POD')
                row['Tuned_FAR'] = td.get('FAR')
                row['Tuned_CSI'] = td.get('CSI')
                row['Tuned_HSS'] = td.get('HSS')
                row['Tuned_TP'] = td.get('True_Positives')
                row['Tuned_FP'] = td.get('False_Positives')
                row['Tuned_FN'] = td.get('False_Negatives')
                row['Tuned_TN'] = td.get('True_Negatives')
            rows.append(row)
        return pd.DataFrame(rows).set_index('Model')

    # -----------------------------------------------------------------
    # Visualisation helpers
    # -----------------------------------------------------------------
    def plot_model_comparison(self, comparison_df, metrics=None, figsize=(16, 6)):
        if metrics is None:
            metrics = ['accuracy', 'f1_weighted', 'POD', 'CSI', 'HSS']
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        for idx, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                comparison_df[metric].plot(kind='bar', ax=axes[idx],
                                           color=colors[:len(comparison_df)])
                axes[idx].set_title(metric.replace('_', ' ').upper(), fontweight='bold')
                axes[idx].set_ylim(0, 1.1)
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(axis='y', alpha=0.3)
        plt.suptitle('Model Comparison — Landslide Risk Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_feature_importance(self, fi_df, top_n=20, figsize=(12, 8)):
        fi_plot = fi_df.head(top_n).sort_values('importance')
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(fi_plot['feature'], fi_plot['importance'], color='#FF5722')
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_roc_curves(self, results_dict, y_test, risk_level=3, figsize=(10, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        y_bin = (np.array(y_test) >= risk_level).astype(int)
        for idx, (name, res) in enumerate(results_dict.items()):
            if 'prediction_probabilities' in res:
                probas = res['prediction_probabilities']
                if probas.shape[1] > risk_level:
                    ls_prob = probas[:, risk_level:].sum(axis=1)
                    fpr, tpr, _ = roc_curve(y_bin, ls_prob)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=colors[idx % 4], linewidth=2,
                           label=f'{name} (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves — Landslide Detection', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def print_evaluation_report(self, result: Dict):
        name = result['model_name']
        bm = result['basic_metrics']
        dm = result['disaster_metrics']
        print(f"\n{'=' * 60}")
        print(f"  {name.upper()}")
        print(f"{'=' * 60}")
        print(f"  Accuracy:  {bm['accuracy']:.4f}   F1: {bm['f1_weighted']:.4f}")
        print(f"  POD: {dm['POD']:.4f}  FAR: {dm['FAR']:.4f}  "
              f"CSI: {dm['CSI']:.4f}  HSS: {dm['HSS']:.4f}")
        print(f"  TP={dm['True_Positives']}  FP={dm['False_Positives']}  "
              f"FN={dm['False_Negatives']}  TN={dm['True_Negatives']}")


def evaluate_all_models(ml_models, X_test, y_test) -> Dict:
    """Evaluate all trained models."""
    evaluator = LandslideEvaluator()
    results = {}
    for name, model in ml_models.models.items():
        results[name] = evaluator.evaluate_model(model, X_test, y_test, name)
    return results
