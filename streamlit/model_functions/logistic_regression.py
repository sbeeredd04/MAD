from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
from imblearn.over_sampling import SMOTE

def train_logistic_regression(X_train, y_train, X_test, y_test, X_val=None, y_val=None, **params):
    """
    Train a logistic regression model and return predictions and model details.
    """
    # Initialize and train model

    # 1. SMOTE on the training data
    sm = SMOTE(random_state=42, sampling_strategy=0.50)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    
    model = LogisticRegression(**params, random_state=42)
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
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=5)
    
    # Feature importances (coefficients for logistic regression)
    feat_importances = pd.Series(abs(model.coef_[0]), index=X_train.columns)
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
    
    
#auto_tune_logistic_regression for auto tuning the logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

def auto_tune_logistic_regression(X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    """
    Automatically tune Logistic Regression parameters and return top 5 models.
    Focuses on the most important parameters:
    - C: inverse of regularization strength
    - penalty: regularization penalty
    - solver: optimization algorithm
    """
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    # Generate all combinations of parameters
    grid = ParameterGrid(param_grid)
    
    # Initialize best models
    best_models = []
    
    # Loop over all parameter combinations
    for params in tqdm(grid):
        # Train model with given parameters
        model = train_logistic_regression(X_train, y_train, X_test, y_test, X_val, y_val, **params)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, model['test_predictions'])
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, model['test_predictions'], average='binary')
        roc_auc = roc_auc_score(y_test, model['test_predictions_proba'])
        
        # Save model and metrics
        best_models.append({
            'model': model['model'],
            'params': params,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'roc_auc': roc_auc
        })
    
    # Sort models by ROC AUC
    best_models = sorted(best_models, key=lambda x: x['roc_auc'], reverse=True)
    
    return best_models[:5]