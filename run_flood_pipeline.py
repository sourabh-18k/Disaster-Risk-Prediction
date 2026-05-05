"""

Flood Risk Prediction Pipeline
End-to-end pipeline : Data Loading → Preprocessing → Feature Engineering → Model Training → Evaluation

Regions : Guwahati (Assam), Patna (Bihar), Kochi (Kerala), Mumbai (Maharashtra), Dehradun (Uttarakhand)
Data : 15 years of daily hydrometeorological data (2010-2024)

"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║      FLOOD DISASTER RISK PREDICTION SYSTEM                                   ║
║      Multi-Region Flood Forecasting for India (2010-2024)                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# Import flood modules
sys.path.append('src/flood')
sys.path.append('src')
from flood.data_preprocessing import load_and_preprocess_data
from flood.feature_engineering import create_all_features
from flood.models import FloodMLModels, create_ensemble_model
from flood.evaluation import FloodEvaluator, evaluate_all_models
from flood.utils import setup_directories, analyze_flood_events


def main():
    """Run the complete flood prediction pipeline"""
    
    # =========================================================================
    # Step 1: Setup
    # =========================================================================
    print("\n[1/7] Setting up project directories...")
    setup_directories()
    
    # =========================================================================
    # Step 2: Load and preprocess data
    # =========================================================================
    print("\n[2/7] Loading and preprocessing flood data...")
    city_data, combined_data = load_and_preprocess_data("flood/")
    
    # =========================================================================
    # Step 3: Analyze flood events
    # =========================================================================
    print("\n[3/7] Analyzing flood events...")
    for city in combined_data['city'].unique():
        analyze_flood_events(combined_data, city)
    
    # =========================================================================
    # Step 4: Feature engineering
    # =========================================================================
    print("\n[4/7] Engineering features...")
    featured_data = create_all_features(combined_data)
    
    # Save featured data
    featured_data.to_csv('data/flood/features/featured_flood_data_all_regions.csv')
    print("  Featured data saved")
    
    # =========================================================================
    # Step 5: Train models
    # =========================================================================
    print("\n[5/7] Training machine learning models...")
    ml_models = FloodMLModels(random_state=42)
    X_train, X_test, y_train, y_test = ml_models.prepare_data(
        featured_data,
        target_col='risk_level',
        test_size=0.2
    )
    
    ml_models.train_all_models(X_train, y_train)
    ml_models.save_models('models/flood/ml/')
    print("  Models trained and saved")
    
    # =========================================================================
    # Step 6: Evaluate models
    # =========================================================================
    print("\n[6/7] Evaluating models...")
    results = evaluate_all_models(ml_models, X_test, y_test)
    
    evaluator = FloodEvaluator()
    comparison_df = evaluator.compare_models(results)
    
    # Save results
    comparison_df.to_csv('results/flood/model_comparison.csv')
    print("  Evaluation complete, results saved")
    
    # =========================================================================
    # Step 7: Summary
    # =========================================================================
    print("\n[7/7] Generating summary...")
    print("\n" + "="*80)
    print("FLOOD PREDICTION RESULTS SUMMARY (v2 — IMPROVED)")
    print("="*80)
    
    print(f"\nDataset:")
    print(f"  Total records: {len(featured_data):,}")
    print(f"  Features: {len(featured_data.columns)}")
    print(f"  Regions: {', '.join(combined_data['city'].unique())}")
    print(f"  Date range: {combined_data.index.min().date()} to {combined_data.index.max().date()}")
    
    flood_days = combined_data['is_flood'].sum()
    print(f"  Flood risk days: {flood_days} ({flood_days/len(combined_data)*100:.2f}%)")
    
    print(f"\nModels Trained:")
    for model_name in ml_models.models.keys():
        print(f"  - {model_name}")
    
    print(f"\n--- DEFAULT (argmax) PREDICTIONS ---")
    best_model = comparison_df['f1_weighted'].idxmax()
    print(f"Best Model (by F1): {best_model}")
    print(f"  Accuracy:  {comparison_df.loc[best_model, 'accuracy']:.4f}")
    print(f"  F1-Score:  {comparison_df.loc[best_model, 'f1_weighted']:.4f}")
    print(f"  POD:       {comparison_df.loc[best_model, 'POD']:.4f}")
    print(f"  FAR:       {comparison_df.loc[best_model, 'FAR']:.4f}")
    print(f"  CSI:       {comparison_df.loc[best_model, 'CSI']:.4f}")
    print(f"  HSS:       {comparison_df.loc[best_model, 'HSS']:.4f}")
    
    # Threshold-tuned summary
    if 'Tuned_POD' in comparison_df.columns:
        print(f"\n--- THRESHOLD-TUNED FLOOD DETECTION ---")
        # Pick best tuned model by Tuned_CSI
        tuned_cols = comparison_df[['Tuned_Threshold', 'Tuned_Accuracy', 'Tuned_F1',
                                     'Tuned_POD', 'Tuned_FAR', 'Tuned_CSI', 'Tuned_HSS']].dropna()
        if len(tuned_cols) > 0:
            best_tuned = tuned_cols['Tuned_CSI'].idxmax()
            print(f"Best Tuned Model (by CSI): {best_tuned}")
            print(f"  Threshold: {comparison_df.loc[best_tuned, 'Tuned_Threshold']:.2f}")
            print(f"  Accuracy:  {comparison_df.loc[best_tuned, 'Tuned_Accuracy']:.4f}")
            print(f"  F1-Score:  {comparison_df.loc[best_tuned, 'Tuned_F1']:.4f}")
            print(f"  POD:       {comparison_df.loc[best_tuned, 'Tuned_POD']:.4f}")
            print(f"  FAR:       {comparison_df.loc[best_tuned, 'Tuned_FAR']:.4f}")
            print(f"  CSI:       {comparison_df.loc[best_tuned, 'Tuned_CSI']:.4f}")
            print(f"  HSS:       {comparison_df.loc[best_tuned, 'Tuned_HSS']:.4f}")
            print(f"  TP: {int(comparison_df.loc[best_tuned, 'Tuned_TP'])}  "
                  f"FP: {int(comparison_df.loc[best_tuned, 'Tuned_FP'])}  "
                  f"FN: {int(comparison_df.loc[best_tuned, 'Tuned_FN'])}  "
                  f"TN: {int(comparison_df.loc[best_tuned, 'Tuned_TN'])}")
    
    print("\n" + "="*80)
    print("Top 10 Most Important Features:")
    print("="*80)
    top_features = ml_models.get_feature_importance(best_model, top_n=10)
    for idx, (_, row) in enumerate(top_features.iterrows()):
        print(f"  {idx+1:2d}. {row['feature']:.<55} {row['importance']:.4f}")
    
    print("\n" + "="*80)
    print("All Model Comparisons (Default):")
    print("="*80)
    print(comparison_df[['accuracy', 'f1_weighted', 'POD', 'FAR', 'CSI', 'HSS']].round(4))
    
    if 'Tuned_POD' in comparison_df.columns:
        print("\n" + "="*80)
        print("All Model Comparisons (Threshold-Tuned):")
        print("="*80)
        tuned_display = ['Tuned_Threshold', 'Tuned_Accuracy', 'Tuned_F1',
                         'Tuned_POD', 'Tuned_FAR', 'Tuned_CSI', 'Tuned_HSS']
        available = [c for c in tuned_display if c in comparison_df.columns]
        if available:
            print(comparison_df[available].round(4))
    
    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print("  data/flood/features/   - Feature-engineered data")
    print("  models/flood/ml/       - Trained models & scaler")
    print("  results/flood/         - Model comparison results")
    
    print("\n" + "="*80)
    print("FLOOD PIPELINE COMPLETE!")
    print("="*80)
    
    return ml_models, featured_data, results, comparison_df


if __name__ == "__main__":
    try:
        ml_models, featured_data, results, comparison_df = main()
        print("\nSuccessfully completed!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
