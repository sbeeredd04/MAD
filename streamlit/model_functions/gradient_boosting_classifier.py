from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', message='Parameters: { "use_label_encoder" } are not used.')

def train_gradient_boosting(X_train, y_train, X_test, y_test, X_val=None, y_val=None, **params):
    """
    Train a gradient boosting model using specified framework with SMOTE resampling.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        X_val, y_val: Optional validation data
        **params: Model parameters including framework
    """
    framework = params.pop('framework', 'sklearn')
    
    # Apply SMOTE to training data
    sm = SMOTE(random_state=42, sampling_strategy=0.50)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    
    # Initialize model based on selected framework
    if framework == 'sklearn':
        model = GradientBoostingClassifier(random_state=42, **params)
    elif framework == 'xgboost':
        model = XGBClassifier(random_state=42, **params)
    elif framework == 'lightgbm':
        model = LGBMClassifier(random_state=42, **params)
    elif framework == 'catboost':
        model = CatBoostClassifier(random_state=42, verbose=False, **params)
    else:
        raise ValueError(f"Unknown framework: {framework}")
    
    # Train model on SMOTE-enhanced data
    model.fit(X_train_sm, y_train_sm)
    
    # Generate predictions for test set
    test_pred = model.predict(X_test)
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Generate predictions for validation set if provided
    val_pred = None
    val_pred_proba = None
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Cross-validation on SMOTE-enhanced data
    cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=5)
    
    # Feature importances (handle different attribute names)
    if framework == 'sklearn':
        importances = model.feature_importances_
    elif framework == 'xgboost':
        importances = model.feature_importances_
    elif framework == 'lightgbm':
        importances = model.feature_importances_
    elif framework == 'catboost':
        importances = model.feature_importances_
    
    feat_importances = pd.Series(importances, index=X_train.columns)
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
    

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score
)
from sklearn.model_selection import ParameterGrid, cross_val_score

# Gradient Boosting Frameworks
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def auto_tune_gradient_boosting(
    X_train, y_train,
    X_test,  y_test,
    X_val=None, y_val=None,
    framework='sklearn'
):

    #adding smote to the training data
    sm = SMOTE(random_state=42, sampling_strategy=0.50)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    # Define parameter grids for each framework
    param_grids = {
        'sklearn': {
            'learning_rate':    [0.01, 0.1],
            'n_estimators':     [100, 200],
            'max_depth':        [3, 5, 7],
            'subsample':        [1.0, 0.8],
            'min_samples_split': [2, 5],
            'min_samples_leaf':  [1, 2],
        },
        'xgboost': {
            'learning_rate':   [0.01, 0.1],
            'n_estimators':    [100, 200],
            'max_depth':       [3, 5, 7],
            'subsample':       [1.0, 0.8],
            'colsample_bytree':[1.0, 0.8],
        },
        'lightgbm': {
            'learning_rate':   [0.01, 0.1],
            'n_estimators':    [100, 200],
            'max_depth':       [3, 5, 7, -1],  # -1 means no limit in LightGBM
            'subsample':       [1.0, 0.8],
            'num_leaves':      [31, 63],
        },
        'catboost': {
            'learning_rate':   [0.01, 0.1],
            'n_estimators':    [100, 200],
            'depth':           [3, 5, 7],
            'subsample':       [1.0, 0.8],
        }
    }

    # Select the appropriate parameter grid
    if framework not in param_grids:
        raise ValueError(f"Unknown framework '{framework}'. "
                         f"Must be one of {list(param_grids.keys())}.")

    param_grid = param_grids[framework]

    print(f"Starting auto-tuning process for {framework} with parameter grid:")
    for param, values in param_grid.items():
        print(f"- {param}: {values}")

    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Total parameter combinations to try: {len(param_combinations)}")

    all_results = []
    failed_combinations = []

    # Create a progress bar
    progress_bar = tqdm(total=len(param_combinations), desc=f"Auto-tuning {framework}")

    for params in param_combinations:
        try:
            # 1. Initialize model
            if framework == 'sklearn':
                model = GradientBoostingClassifier(random_state=42, **params)
            elif framework == 'xgboost':
                # Some XGBoost versions require disabling user_label_encoder or specifying eval_metric
                model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **params)
            elif framework == 'lightgbm':
                model = LGBMClassifier(random_state=42, **params)
            elif framework == 'catboost':
                model = CatBoostClassifier(random_state=42, verbose=False, **params)
            else:
                raise ValueError(f"Unknown framework: {framework}")
            
            # 2. Train model on training data
            model.fit(X_train_sm, y_train_sm)

            # 3. Predict on test data
            test_pred = model.predict(X_test)
            # Many GBDT frameworks have predict_proba; if not, adjust accordingly
            test_pred_proba = model.predict_proba(X_test)[:, 1]

            # 4. Calculate test metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, test_pred, average='binary', zero_division=0
            )
            accuracy = accuracy_score(y_test, test_pred)
            roc_auc = roc_auc_score(y_test, test_pred_proba)

            # 5. Cross-validation on the training set
            cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=5)

            # 6. Feature importances
            # Different frameworks store importances differently,
            # but all used here expose 'feature_importances_' or 'feature_importance()'.
            if hasattr(model, 'feature_importances_'):
                feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
            else:
                # For CatBoost, model.get_feature_importance() can be used
                feat_importances = pd.Series(model.get_feature_importance(), index=X_train.columns)
            
            feat_importances.sort_values(ascending=False, inplace=True)

            # 7. If validation set is provided, get validation predictions
            val_pred = None
            val_pred_proba = None
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                val_pred_proba = model.predict_proba(X_val)[:, 1]

            # 8. Store model outputs
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

            # 9. Summarize metrics
            test_metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'ROC_AUC': roc_auc,
                'CV_Mean': cv_scores.mean(),
                'CV_Std': cv_scores.std()
            }

            # 10. Calculate an "overall_score" to compare models
            #    Weighted combination of metrics, modify as desired
            overall_score = (
                    accuracy * 0.2 +    # Reduced from 0.3
                    f1 * 0.3 +         # Reduced from 0.3
                    roc_auc * 0.2 +    # Reduced from 0.4
                    recall * 0.3       # Added recall with 0.3 weight
                )
    
            # Collect all results
            all_results.append({
                'params': params,
                'overall_score': overall_score,
                'metrics': test_metrics,
                'model_results': model_results
            })

        except Exception as e:
            # If a combination fails (e.g., invalid hyperparams), save the error
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

    # Print summary of failures (if any)
    if failed_combinations:
        print("\nFailed parameter combinations:")
        for fail in failed_combinations:
            print(f"Parameters: {fail['params']}")
            print(f"Error: {fail['error']}")
            print("---")

    # Sort the successful results by overall_score, descending
    all_results.sort(key=lambda x: x['overall_score'], reverse=True)

    print(f"\nAuto-tuning complete. Found {len(all_results)} successful models.")
    if all_results:
        print(f"Best score achieved: {all_results[0]['overall_score']:.4f}")
    else:
        print("No successful models found.")

    # Print top 5 models (or fewer if fewer than 5)
    top_k = min(5, len(all_results))
    print(f"\nShowing top {top_k} models:")
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

    return all_results[:top_k]
