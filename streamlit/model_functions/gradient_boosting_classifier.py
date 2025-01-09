from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

def train_gradient_boosting(X_train, y_train, X_test, y_test, X_val=None, y_val=None, **params):
    """
    Train a gradient boosting model using specified framework and return predictions and model details.
    """
    framework = params.pop('framework', 'sklearn')
    
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
    
    # Train model
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