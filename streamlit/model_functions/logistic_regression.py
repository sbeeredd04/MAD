from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

def train_logistic_regression(X_train, y_train, X_test, X_val=None, y_val=None, **params):
    """
    Train a logistic regression model and return predictions and model details.
    """
    # Initialize and train model
    model = LogisticRegression(**params, random_state=42)
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