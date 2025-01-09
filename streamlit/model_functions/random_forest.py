from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

def train_random_forest(X_train, y_train, X_test, y_test, X_val=None, y_val=None, **params):
    """
    Train a random forest classifier and return predictions and model details.
    """
    # Initialize and train model
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Generate predictions for test set
    test_pred = model.predict(X_test)
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Generate predictions for validation set if provided
    val_pred = None
    val_pred_proba = None
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Feature importances
    feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_importances.sort_values(ascending=False, inplace=True)
    
    return {
        'model': model,
        'test_predictions': test_pred,
        'test_predictions_proba': test_pred_proba,
        'val_predictions': val_pred,
        'val_predictions_proba': val_pred_proba,
        'cv_scores': cv_scores,
        'feature_importances': feat_importances
    }
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd
from tqdm import tqdm

def auto_tune_random_forest(X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    """
    Automatically tune Random Forest parameters and return top 5 models.
    Focuses on the most important parameters:
    - n_estimators: number of trees in the forest
    - max_depth: maximum depth of the trees
    - min_samples_split: minimum samples required to split an internal node
    - min_samples_leaf: minimum samples required at a leaf node
    - max_features: number of features to consider when looking for the best split
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 150, 200, 300], 
        'max_depth': [None, 4, 8, 10, 15, 20],  
        'min_samples_split': [2, 4, 6],   
        'min_samples_leaf': [1, 2, 4],    
        'max_features': ['sqrt', 'log2']   
    }
    
    print("Starting auto-tuning process with parameter grid:")
    for param, values in param_grid.items():
        print(f"- {param}: {values}")
    
    # Generate all combinations
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Total parameter combinations to try: {len(param_combinations)}")
    
    all_results = []
    failed_combinations = []
    
    # Create progress bar
    progress_bar = tqdm(total=len(param_combinations), desc="Auto-tuning Random Forest")
    
    for params in param_combinations:
        try:
            # Train model with current parameters
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Generate predictions
            test_pred = model.predict(X_test)
            test_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, test_pred, average='binary', zero_division=0
            )
            accuracy = accuracy_score(y_test, test_pred)
            roc_auc = roc_auc_score(y_test, test_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Store results
            model_results = {
                'model': model,
                'test_predictions': test_pred,
                'test_predictions_proba': test_pred_proba,
                'cv_scores': cv_scores,
                'roc_auc': roc_auc
            }
            
            # Calculate metrics
            test_metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'ROC_AUC': roc_auc,
                'CV_Mean': cv_scores.mean(),
                'CV_Std': cv_scores.std()
            }
            
            # Calculate overall score
            overall_score = (
                accuracy * 0.3 +
                f1 * 0.3 +
                roc_auc * 0.4
            )
            
            all_results.append({
                'params': params,
                'overall_score': overall_score,
                'metrics': test_metrics,
                'model_results': model_results
            })
            
        except Exception as e:
            error_msg = f"Error with parameters {params}: {str(e)}"
            print(error_msg)
            failed_combinations.append({
                'params': params,
                'error': str(e)
            })
            continue
        
        finally:
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Print summary of failures if any occurred
    if failed_combinations:
        print("\nFailed parameter combinations:")
        for fail in failed_combinations:
            print(f"Parameters: {fail['params']}")
            print(f"Error: {fail['error']}")
            print("---")
    
    # Sort results by overall score and return top 5
    all_results.sort(key=lambda x: x['overall_score'], reverse=True)
    
    print(f"\nAuto-tuning complete. Found {len(all_results)} successful models.")
    print(f"Best score achieved: {all_results[0]['overall_score'] if all_results else 'No successful models'}")

    # Print top 5 models' metrics
    for i, result in enumerate(all_results[:5], 1):
        print(f"Model {i}:")
        print(f"Parameters: {result['params']}")
        print(f"Overall Score: {result['overall_score']:.4f}")
        print("Metrics:", result['metrics'])
        print("---")
    
    return all_results[:5]
