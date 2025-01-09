from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

def train_svm(X_train, y_train, X_test, y_test, X_val=None, y_val=None, **params):
    """
    Train an SVM classifier and return predictions and model details.
    
    Parameters:
    -----------
    X_train: array-like
        Training features
    y_train: array-like
        Training labels
    X_test: array-like
        Test features
    y_test: array-like
        Test labels
    X_val: array-like, optional
        Validation features
    y_val: array-like, optional
        Validation labels
    kernel: str
        The kernel type ('linear', 'rbf', 'poly', 'sigmoid')
    C: float
        Regularization parameter
    gamma: str or float
        Kernel coefficient ('scale', 'auto' or float value)
    degree: int
        Degree of polynomial kernel (only for poly kernel)
    """
    # Initialize and train model
    model = SVC(probability=True, random_state=42, **params)
    model.fit(X_train, y_train)
    
    # Generate predictions for test set
    test_pred = model.predict(X_test)
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Generate validation predictions if provided
    val_pred = None
    val_pred_proba = None
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Feature importances through permutation
    importances = np.zeros(X_train.shape[1])
    baseline_score = model.score(X_test, y_test)
    
    # Convert DataFrame to numpy array for permutation
    X_test_np = X_test.to_numpy()
    
    for i in range(X_train.shape[1]):
        X_test_permuted = X_test_np.copy()
        X_test_permuted[:, i] = np.random.permutation(X_test_permuted[:, i])
        # Convert back to DataFrame for model.score
        X_test_permuted_df = pd.DataFrame(X_test_permuted, columns=X_test.columns, index=X_test.index)
        score = model.score(X_test_permuted_df, y_test)
        importances[i] = baseline_score - score
    
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