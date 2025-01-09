from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid, cross_val_score
import pandas as pd
from tqdm import tqdm

def train_decision_tree(X_train, y_train, X_test, y_test, X_val=None, y_val=None, **params):

    # Initialize and train model
    model = DecisionTreeClassifier(**params, random_state=42)
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
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, test_pred_proba)
    
    return {
        'model': model,
        'test_predictions': test_pred,
        'test_predictions_proba': test_pred_proba,
        'val_predictions': val_pred,
        'val_predictions_proba': val_pred_proba,
        'cv_scores': cv_scores,
        'feature_importances': feat_importances,
        'roc_auc': roc_auc
    }
    

def auto_tune_decision_tree(X_train, y_train, X_test, y_test, X_val=None, y_val=None):

    # Define focused parameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],  # Measure of split quality
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],  # Control tree complexity
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],   # Minimum samples for splitting
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]      # Minimum samples at leaves
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
    progress_bar = tqdm(total=len(param_combinations), desc="Auto-tuning Decision Tree")
    
    for params in param_combinations:
        try:
            # Train model with current parameters
            model = DecisionTreeClassifier(**params, random_state=42)
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
            
            # Feature importances
            feat_importances = pd.Series(
                model.feature_importances_, 
                index=X_train.columns
            ).sort_values(ascending=False)
            
            # Validation predictions if provided
            val_pred = None
            val_pred_proba = None
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                val_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Store results
            model_results = {
                'model': model,
                'test_predictions': test_pred,
                'test_predictions_proba': test_pred_proba,
                'val_predictions': val_pred,
                'val_predictions_proba': val_pred_proba,
                'cv_scores': cv_scores,
                'feature_importances': feat_importances,
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

    #printing top 5 models accuracy, precision, recall, f1, roc_auc, cv_mean, cv_std
    for i, result in enumerate(all_results[:5], 1):
        print(f"Model {i}:")
        print(f"Accuracy: {result['metrics']['Accuracy']:.4f}")
        print(f"Precision: {result['metrics']['Precision']:.4f}")
        print(f"Recall: {result['metrics']['Recall']:.4f}")
        print(f"F1: {result['metrics']['F1']:.4f}")
        print(f"ROC AUC: {result['metrics']['ROC_AUC']:.4f}")
        print(f"CV Mean: {result['metrics']['CV_Mean']:.4f}")
        print(f"CV Std: {result['metrics']['CV_Std']:.4f}")
        print("---")
    
    return all_results[:5]