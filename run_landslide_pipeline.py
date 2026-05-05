"""
Landslide Risk Prediction Pipeline
End-to-end: Data Loading → Preprocessing → Feature Engineering → Model Training → Evaluation

Regions:  Shimla (HP), Munnar (Kerala), Darjeeling (WB), Uttarkashi (UK), Kohima (Nagaland)
Data:     15 years of daily geo-hydrometeorological data (2010-2024)
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║      LANDSLIDE DISASTER RISK PREDICTION SYSTEM                               ║
║      Multi-Region Landslide Forecasting for India (2010-2024)                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

sys.path.append('src/landslide')
sys.path.append('src')
from landslide.data_preprocessing import load_and_preprocess_data
from landslide.feature_engineering import create_all_features
from landslide.models import LandslideMLModels
from landslide.evaluation import LandslideEvaluator, evaluate_all_models
from landslide.utils import setup_directories, analyze_landslide_events


def main():
    """Run the complete landslide prediction pipeline."""

    # =========================================================================
    # Step 1: Setup
    # =========================================================================
    print("\n[1/7] Setting up project directories...")
    setup_directories()

    # =========================================================================
    # Step 2: Load and preprocess data
    # =========================================================================
    print("\n[2/7] Loading and preprocessing landslide data...")
    region_data, combined_data = load_and_preprocess_data("landslide/")

    # =========================================================================
    # Step 3: Analyze landslide events
    # =========================================================================
    print("\n[3/7] Analyzing landslide events...")
    for region in combined_data['region'].unique():
        analyze_landslide_events(combined_data, region)

    # =========================================================================
    # Step 4: Feature engineering
    # =========================================================================
    print("\n[4/7] Engineering features...")
    featured_data = create_all_features(combined_data)
    featured_data.to_csv('data/landslide/features/featured_landslide_data_all_regions.csv')
    print(f"  Saved featured data: {featured_data.shape}")

    # =========================================================================
    # Step 5: Train models
    # =========================================================================
    print("\n[5/7] Training ML models (SMOTE + custom class weights)...")
    ml_models = LandslideMLModels(random_state=42, use_smote=True)
    X_train, X_test, y_train, y_test = ml_models.prepare_data(
        featured_data, target_col='risk_level', test_size=0.2
    )
    ml_models.train_all_models(X_train, y_train)
    ml_models.save_models('models/landslide/ml/')

    # =========================================================================
    # Step 6: Evaluate models
    # =========================================================================
    print("\n[6/7] Evaluating models (default + threshold-tuned)...")
    results = evaluate_all_models(ml_models, X_test, y_test)
    evaluator = LandslideEvaluator()

    for name, res in results.items():
        evaluator.print_evaluation_report(res)
        if 'threshold_tuned' in res and res['threshold_tuned']:
            t = res['threshold_tuned']
            print(f"  >> TUNED: threshold={t['optimal_threshold']}, "
                  f"POD={t['best_scan_metrics'].get('POD', 0):.4f}, "
                  f"FAR={t['best_scan_metrics'].get('FAR', 0):.4f}, "
                  f"CSI={t['best_scan_metrics'].get('CSI', 0):.4f}")

    comparison_df = evaluator.compare_models(results)
    comparison_df.to_csv('results/landslide/model_comparison.csv')
    print(f"\nModel comparison saved to results/landslide/model_comparison.csv")

    # =========================================================================
    # Step 7: Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("LANDSLIDE PREDICTION — RESULTS SUMMARY")
    print("=" * 80)

    # Default (argmax) results
    print("\nDEFAULT (argmax) PREDICTIONS:")
    default_cols = ['accuracy', 'f1_weighted', 'POD', 'FAR', 'CSI', 'HSS']
    available = [c for c in default_cols if c in comparison_df.columns]
    print(comparison_df[available].round(4).to_string())

    # Threshold-tuned results
    tuned_cols = ['Tuned_Threshold', 'Tuned_Accuracy', 'Tuned_F1',
                  'Tuned_POD', 'Tuned_FAR', 'Tuned_CSI', 'Tuned_HSS']
    available_t = [c for c in tuned_cols if c in comparison_df.columns]
    if available_t:
        print("\nTHRESHOLD-TUNED LANDSLIDE DETECTION:")
        print(comparison_df[available_t].round(4).to_string())

        if 'Tuned_CSI' in comparison_df.columns:
            best = comparison_df['Tuned_CSI'].idxmax()
            print(f"\n*** BEST MODEL: {best} (threshold-tuned) ***")
            print(f"  Threshold:  {comparison_df.loc[best, 'Tuned_Threshold']}")
            print(f"  Accuracy:   {comparison_df.loc[best, 'Tuned_Accuracy']:.4f}")
            print(f"  POD:        {comparison_df.loc[best, 'Tuned_POD']:.4f}")
            print(f"  FAR:        {comparison_df.loc[best, 'Tuned_FAR']:.4f}")
            print(f"  CSI:        {comparison_df.loc[best, 'Tuned_CSI']:.4f}")
            print(f"  HSS:        {comparison_df.loc[best, 'Tuned_HSS']:.4f}")
            if 'Tuned_TP' in comparison_df.columns:
                print(f"  TP={int(comparison_df.loc[best, 'Tuned_TP'])}, "
                      f"FP={int(comparison_df.loc[best, 'Tuned_FP'])}, "
                      f"FN={int(comparison_df.loc[best, 'Tuned_FN'])}, "
                      f"TN={int(comparison_df.loc[best, 'Tuned_TN'])}")

    # Feature importance
    best_default = comparison_df['f1_weighted'].idxmax()
    top_features = ml_models.get_feature_importance(best_default, top_n=10)
    if len(top_features) > 0:
        print(f"\nTop 10 Features ({best_default}):")
        for i, (_, row) in enumerate(top_features.iterrows()):
            print(f"  {i + 1:2d}. {row['feature']:.<50} {row['importance']:.4f}")

    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
