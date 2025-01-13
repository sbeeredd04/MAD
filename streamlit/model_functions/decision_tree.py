from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid, cross_val_score
import pandas as pd
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
import pandas as pd

def train_decision_tree(X_train, y_train, 
                              X_test, y_test, 
                              X_val=None, y_val=None, 
                              **params):

    # 1. SMOTE on the training data
    sm = SMOTE(random_state=42, sampling_strategy=0.50)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    # 2. Initialize and train model on SMOTE-enhanced data
    model = DecisionTreeClassifier(random_state=42, **params)
    model.fit(X_train_sm, y_train_sm)

    # 3. Predictions on the (untouched) test set
    test_pred = model.predict(X_test)
    test_pred_proba = model.predict_proba(X_test)[:, 1]

    # 4. Predictions on the (untouched) validation set if provided
    val_pred = None
    val_pred_proba = None
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_pred_proba = model.predict_proba(X_val)[:, 1]

    # 5. Cross-validation on the SMOTE-enhanced training data
    cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=5)

    # 6. Feature importances
    feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_importances.sort_values(ascending=False, inplace=True)

    # 7. Calculate ROC AUC on the test set
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

     # Initialize SMOTE
    sm = SMOTE(random_state=42, sampling_strategy=0.50)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    
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
            model.fit(X_train_sm, y_train_sm)
            
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
            cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=5)
            
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
                accuracy * 0.2 +    # Reduced from 0.3
                f1 * 0.3 +         # Reduced from 0.3
                roc_auc * 0.2 +    # Reduced from 0.4
                recall * 0.3       # Added recall with 0.3 weight
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

    for i, result in enumerate(all_results[:5], 1):
        print(f"\nModel {i}:")
        print("Performance Metrics:")
        print(f"{'Metric':<15} {'Score':<10}")
        print("-" * 25)
        print(f"{'Accuracy':<15} {result['metrics']['Accuracy']:.4f}")
        print(f"{'Precision':<15} {result['metrics']['Precision']:.4f}")
        print(f"{'Recall':<15} {result['metrics']['Recall']:.4f} *")  # Highlight recall
        print(f"{'F1':<15} {result['metrics']['F1']:.4f}")
        print(f"{'ROC AUC':<15} {result['metrics']['ROC_AUC']:.4f}")
        print(f"{'CV Mean':<15} {result['metrics']['CV_Mean']:.4f}")
        print(f"{'CV Std':<15} {result['metrics']['CV_Std']:.4f}")
        print(f"\nOverall Score: {result['overall_score']:.4f}")
        print("---")
    
    return all_results[:5]