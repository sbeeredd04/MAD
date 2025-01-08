import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import os
from pathlib import Path
import model_functions
from model_functions.decision_tree import train_decision_tree
from model_functions.random_forest import train_random_forest
from model_functions.gradient_boosting_classifier import train_gradient_boosting
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.inspection import permutation_importance



def get_available_csvs(root_dir='../data'):

    csv_files = []
    root_path = Path(root_dir)
    
    for path in root_path.rglob('*.csv'):
        # Create a display name that shows the relative path from root_dir
        relative_path = path.relative_to(root_path)
        display_name = str(relative_path)
        csv_files.append((display_name, str(path.absolute())))
    
    return sorted(csv_files)

def save_dataframe_to_csv(df, file_path='../data', name=None):
    df.to_csv(file_path, index=False)

def load_and_analyze_csv(file_path):

    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Analyze columns
    analysis = {
        'date_columns': [],
        'numeric_columns': [],
        'categorical_columns': [],
        'target_columns': [],
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_data': {}
    }
    
    for column in df.columns:
        # Check for missing values
        missing = df[column].isnull().sum()
        if missing > 0:
            analysis['missing_data'][column] = missing
        
        # Analyze column type
        if df[column].dtype in ['int64', 'float64']:
            # Check if binary (potential target)
            if df[column].nunique() == 2:
                analysis['target_columns'].append(column)
            else:
                analysis['numeric_columns'].append(column)
        elif df[column].dtype == 'object':
            # Try to parse as date
            try:
                pd.to_datetime(df[column])
                analysis['date_columns'].append(column)
            except:
                analysis['categorical_columns'].append(column)
    
    return df, analysis

def get_feature_engineering_options():
    return {
        'Raw Features': {
            'description': 'Use original features without transformation',
            'params': {}
        },
        'Rolling Features': {
            'description': 'Apply rolling window statistics to features',
            'params': {
                'window_size': {
                    'type': 'int',
                    'min': 2,
                    'max': 50,
                    'default': 3,
                    'description': 'Size of rolling window'
                },
                'operations': {
                    'type': 'multiselect',
                    'options': ['mean', 'std', 'min', 'max', 'median', 'skew', 'zscore'],
                    'default': ['mean'],
                    'description': 'Rolling window operations to apply'
                }
            }
        },
        'Lagging Features': {
            'description': 'Add lagged versions of features',
            'params': {
                'lag_periods': {
                    'type': 'multiselect',
                    'options': [1, 2, 3, 4, 5, 6, 7, 14, 21, 28],
                    'default': [1],
                    'description': 'Number of periods to lag'
                }
            }
        }
    }
    
def add_lagging_features(df, feature_cols, lag_periods=None):

    df = df.copy()
    
    # Default lag periods if none specified
    if lag_periods is None:
        lag_periods = [1]
    
    # Store non-feature columns
    non_feature_cols = [col for col in df.columns if col not in feature_cols]
    new_df = df.copy()
    
    # Add lag features
    for col in feature_cols:
        for lag in lag_periods:
            new_col_name = f"{col}_lag_{lag}"
            new_df[new_col_name] = df[col].shift(lag)
    
    return new_df.dropna()

def apply_feature_engineering(df, feature_cols, target_col, method='Raw Features', params=None):
    df = df.copy()
    corr_method = params.get('correlation_method', 'pearson')
    
    if method == 'Raw Features':
        transformed_df = df
        
    elif method == 'Rolling Features':
        if params is None:
            params = {
                'window_size': 3,
                'operations': ['mean']
            }
            
        transformed_df = add_rolling_features(
            df, 
            feature_cols, 
            rolling_window=params['window_size'],
            rolling_ops=params['operations']
        )
        
    elif method == 'Lagging Features':
        if params is None:
            params = {
                'lag_periods': [1]
            }
            
        transformed_df = add_lagging_features(
            df,
            feature_cols,
            lag_periods=params['lag_periods']
        )
    
    # Calculate feature summary
    feature_summary = {
        'correlation_with_target': transformed_df[feature_cols].corrwith(
            transformed_df[target_col], 
            method=corr_method
        ),
        'correlation_matrix': transformed_df[feature_cols].corr(method=corr_method),
        'feature_stats': transformed_df[feature_cols].describe(),
        'missing_values': transformed_df[feature_cols].isnull().sum()
    }
    
    return transformed_df, feature_summary

def add_rolling_features(df, feature_cols, rolling_window=3, rolling_ops=None):

    df = df.copy()
        
    # Default operations if none specified
    if rolling_ops is None:
        rolling_ops = ['mean']
    
    # Store non-feature columns
    non_feature_cols = [col for col in df.columns if col not in feature_cols]
    new_df = df.copy()
    
    # Calculate rolling features
    for col in feature_cols:
        roll_obj = df[col].rolling(window=rolling_window)
        
        # Apply each requested operation
        for op in rolling_ops:
            if op == 'mean':
                new_df[f"{col}_roll_mean_{rolling_window}"] = roll_obj.mean()
            elif op == 'std':
                new_df[f"{col}_roll_std_{rolling_window}"] = roll_obj.std()
            elif op == 'min':
                new_df[f"{col}_roll_min_{rolling_window}"] = roll_obj.min()
            elif op == 'max':
                new_df[f"{col}_roll_max_{rolling_window}"] = roll_obj.max()
            elif op == 'median':
                new_df[f"{col}_roll_median_{rolling_window}"] = roll_obj.median()
            elif op == 'skew':
                new_df[f"{col}_roll_skew_{rolling_window}"] = roll_obj.skew()
            elif op == 'zscore':
                # Calculate rolling z-score
                roll_mean = roll_obj.mean()
                roll_std = roll_obj.std()
                new_df[f"{col}_roll_zscore_{rolling_window}"] = (df[col] - roll_mean) / (roll_std + 1e-9)
                
    return new_df.dropna()


def compute_correlations(df, target_col='Y', threshold=0.05):

    # Exclude target and 'Data' from features
    feature_cols = [c for c in df.columns if c not in [target_col, 'Data']]
    
    # Compute Pearson correlation
    correlations = df[feature_cols].corrwith(df[target_col], method='pearson')
    correlations = correlations.dropna()  # drop any NaN results

    # Sort by absolute correlation
    correlations = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )

    # Filter based on threshold
    high_corr_features = correlations[abs(correlations) >= threshold]
    
    correlation_dict = {
        'selected_features': high_corr_features.index.tolist(),
        'correlation_scores': high_corr_features.values
    }
    return correlations, correlation_dict


def create_time_split(df, target_col='Y', selected_features=None, split_date='2018-01-01'):
    if selected_features is None:
        selected_features = [col for col in df.columns if col not in [target_col, 'Data']]

    # Convert split_date to datetime if it's not already
    if isinstance(split_date, str):
        split_date = pd.to_datetime(split_date)
    
    # Ensure split_date is datetime64[ns]
    if not isinstance(split_date, pd.Timestamp):
        split_date = pd.Timestamp(split_date)

    # Train if Data < split_date, test if Data >= split_date
    train_mask = df['Data'] < split_date
    test_mask = df['Data'] >= split_date

    X_train = df.loc[train_mask, selected_features]
    X_test = df.loc[test_mask, selected_features]
    y_train = df.loc[train_mask, target_col]
    y_test = df.loc[test_mask, target_col]

    return X_train, X_test, y_train, y_test

def create_time_split_with_validation(df, target_col, selected_features, test_split_date, val_split_date):
    # Convert dates to datetime if they're not already
    if isinstance(test_split_date, str):
        test_split_date = pd.to_datetime(test_split_date)
    if isinstance(val_split_date, str):
        val_split_date = pd.to_datetime(val_split_date)
    
    # Ensure dates are datetime64[ns]
    if not isinstance(test_split_date, pd.Timestamp):
        test_split_date = pd.Timestamp(test_split_date)
    if not isinstance(val_split_date, pd.Timestamp):
        val_split_date = pd.Timestamp(val_split_date)

    # Create masks for splitting
    train_mask = df['Data'] < test_split_date
    test_mask = (df['Data'] >= test_split_date) & (df['Data'] < val_split_date)
    val_mask = df['Data'] >= val_split_date
    
    # Split features
    X_train = df[train_mask][selected_features]
    X_test = df[test_mask][selected_features]
    X_val = df[val_mask][selected_features]
    
    # Split target
    y_train = df[train_mask][target_col]
    y_test = df[test_mask][target_col]
    y_val = df[val_mask][target_col]
    
    return X_train, X_test, X_val, y_train, y_test, y_val


def get_performance_metrics(y_true, y_pred):
    """Calculate performance metrics for model evaluation."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    }
    return metrics

def train_model(X_train, y_train, X_test, y_test, X_val=None, y_val=None, model_type='decision_tree', **params):

    # Select appropriate training function
    if model_type == 'decision_tree':
        model_results = train_decision_tree(X_train, y_train, X_test, X_val, y_val, **params)
    elif model_type == 'random_forest':
        model_results = train_random_forest(X_train, y_train, X_test, X_val, y_val, **params)
    elif model_type == 'gradient_boost':
        model_results = train_gradient_boosting(X_train, y_train, X_test, X_val, y_val, **params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate metrics for test set
    test_metrics = get_performance_metrics(y_test, model_results['test_predictions'])
    test_metrics['CV Score Mean'] = model_results['cv_scores'].mean()
    test_metrics['CV Score Std'] = model_results['cv_scores'].std()
    
    # Calculate metrics for validation set if provided
    val_metrics = None
    if X_val is not None and y_val is not None and model_results['val_predictions'] is not None:
        val_metrics = get_performance_metrics(y_val, model_results['val_predictions'])
    
    # Add metrics to results
    model_results['test_metrics'] = test_metrics
    model_results['val_metrics'] = val_metrics
    
    return model_results

def compute_permutation_importance(model, X_test, y_test, n_repeats=5):

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42
    )
    
    # Create DataFrame with importance scores
    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    })
    
    # Sort by importance and set index to Feature
    importance_df.sort_values('Importance', ascending=False, inplace=True)
    importance_df.set_index('Feature', inplace=True)
    
    return importance_df

def create_feature_importance_summary(model_feature_importance, perm_importance, feature_names):

    summary_df = pd.DataFrame({
        'Feature': feature_names,
        'Native_Importance': model_feature_importance,
        'Permutation_Importance': perm_importance['Importance'],  # Keep these column names consistent
        'Permutation_Std': perm_importance['Std']
    })
    
    # Sort by permutation importance
    summary_df = summary_df.sort_values('Permutation_Importance', ascending=False)
    summary_df.set_index('Feature', inplace=True)
    
    return summary_df

def plot_permutation_importance(importance_df):

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create error bars plot using the correct column names
    importance_df.plot(
        y='Permutation_Importance',  # Changed from 'Importance' to match the column name
        yerr='Permutation_Std',      # Changed from 'Std' to match the column name
        kind='barh',
        ax=ax
    )
    
    ax.set_title('Feature Importance (Permutation)')
    ax.set_xlabel('Mean Importance')
    ax.set_ylabel('Features')
    
    plt.tight_layout()
    return fig