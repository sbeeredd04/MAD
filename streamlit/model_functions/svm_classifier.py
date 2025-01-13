from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

def train_svm(X_train, y_train, X_test, y_test, X_val=None, y_val=None, **params):
    """
    Train an SVM classifier with SMOTE resampling.
    """
    # Apply SMOTE to training data
    sm = SMOTE(random_state=42, sampling_strategy=0.50)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    
    # Initialize and train model on SMOTE-enhanced data
    model = SVC(probability=True, random_state=42, **params)
    model.fit(X_train_sm, y_train_sm)
    
    # Generate predictions for test set
    test_pred = model.predict(X_test)
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Generate validation predictions if provided
    val_pred = None
    val_pred_proba = None
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Cross-validation on SMOTE-enhanced data
    cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=5)
    
    # ...existing feature importance calculation code...
    
    return {
        'model': model,
        'test_predictions': test_pred,
        'test_predictions_proba': test_pred_proba,
        'val_predictions': val_pred,
        'val_predictions_proba': val_pred_proba,
        'cv_scores': cv_scores,
        'feature_importances': feat_importances
    }