"""
Quick Start Script for Heatwave Disaster Risk Prediction System
Run this script to execute the entire pipeline end-to-end
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║      HEATWAVE DISASTER RISK PREDICTION SYSTEM                                ║
║      Research-Level Multi-City Prediction for India                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# Import modules
sys.path.append('src')
from data_preprocessing import load_and_preprocess_data
from feature_engineering import create_all_features
from models import HeatwaveMLModels, create_ensemble_model
from evaluation import HeatwaveEvaluator, evaluate_all_models
from utils import setup_directories, analyze_heatwave_events

def main():
    """Run the complete pipeline"""
    
    # Setup directories
    print("\n[1/7] Setting up project directories...")
    setup_directories()
    
    # Load and preprocess data
    print("\n[2/7] Loading and preprocessing data...")
    city_data, combined_data = load_and_preprocess_data("heat/")
    
    # Analyze heatwave events
    print("\n[3/7] Analyzing heatwave events...")
    for city in combined_data['city'].unique():
        analyze_heatwave_events(combined_data, city)
    
    # Feature engineering
    print("\n[4/7] Engineering features...")
    featured_data = create_all_features(combined_data)
    
    # Save featured data
    featured_data.to_csv('data/features/featured_data_all_cities.csv')
    print("✓ Featured data saved")
    
    # Prepare models
    print("\n[5/7] Training machine learning models...")
    ml_models = HeatwaveMLModels(random_state=42)
    X_train, X_test, y_train, y_test = ml_models.prepare_data(
        featured_data, 
        target_col='risk_level', 
        test_size=0.2
    )
    
    # Train models
    ml_models.train_all_models(X_train, y_train)
    ml_models.save_models('models/ml/')
    print("✓ Models trained and saved")
    
    # Evaluate models
    print("\n[6/7] Evaluating models...")
    results = evaluate_all_models(ml_models, X_test, y_test)
    
    # Create evaluator and get comparison
    evaluator = HeatwaveEvaluator()
    comparison_df = evaluator.compare_models(results)
    
    # Save comparison
    comparison_df.to_csv('results/model_comparison.csv')
    print("✓ Evaluation complete, results saved")
    
    # Print summary
    print("\n[7/7] Generating summary...")
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nDataset:")
    print(f"  Total records: {len(featured_data):,}")
    print(f"  Features: {len(featured_data.columns)}")
    print(f"  Cities: {', '.join(combined_data['city'].unique())}")
    
    print(f"\nModels Trained:")
    for model_name in ml_models.models.keys():
        print(f"  ✓ {model_name}")
    
    print(f"\nBest Model: {comparison_df['f1_weighted'].idxmax()}")
    best_model = comparison_df['f1_weighted'].idxmax()
    print(f"  Accuracy: {comparison_df.loc[best_model, 'accuracy']:.4f}")
    print(f"  F1-Score: {comparison_df.loc[best_model, 'f1_weighted']:.4f}")
    print(f"  POD (Probability of Detection): {comparison_df.loc[best_model, 'POD']:.4f}")
    print(f"  CSI (Critical Success Index): {comparison_df.loc[best_model, 'CSI']:.4f}")
    print(f"  FAR (False Alarm Rate): {comparison_df.loc[best_model, 'FAR']:.4f}")
    
    print("\n" + "="*80)
    print("Top 10 Most Important Features:")
    print("="*80)
    top_features = ml_models.get_feature_importance(best_model, top_n=10)
    for idx, row in top_features.iterrows():
        print(f"  {idx+1:2d}. {row['feature']:.<60} {row['importance']:.4f}")
    
    print("\n" + "="*80)
    print("All Model Comparisons:")
    print("="*80)
    print(comparison_df[['accuracy', 'f1_weighted', 'POD', 'FAR', 'CSI']].round(4))
    
    print("\n" + "="*80)
    print("OUTPUT FILES CREATED")
    print("="*80)
    print("  📁 data/processed/ - Preprocessed city data")
    print("  📁 data/features/ - Feature-engineered data")
    print("  📁 models/ml/ - Trained models")
    print("  📁 results/ - Evaluation results")
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Open 'main_analysis.ipynb' for detailed analysis")
    print("  2. Review 'results/model_comparison.csv' for metrics")
    print("  3. Run visualizations in the notebook")
    print("  4. Use trained models for predictions")
    
    return ml_models, featured_data, results, comparison_df


if __name__ == "__main__":
    try:
        ml_models, featured_data, results, comparison_df = main()
        print("\n✓ Successfully completed!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
