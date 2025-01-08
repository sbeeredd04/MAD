import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import  classification_report, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_correlations(df, target_col='Y'):
    
    # Get all columns except target and Data
    feature_cols = df.columns.drop([target_col, 'Data'])
    
    # Compute Pearson correlations
    correlations = df[feature_cols].corrwith(df[target_col], method='pearson')
    
    # Sort by absolute correlation values
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    
    # Select features with absolute correlation above threshold (0.1)
    high_corr_features = correlations[abs(correlations) >= 0.05]
    
    return correlations, {
        'selected_features': high_corr_features.index.tolist(),
        'correlation_scores': high_corr_features.values
    }

def create_train_test_split(df, selected_features, target_col='Y', split_date='2018-01-01'):
    
    # Split based on date
    train_mask = df['Data'] < split_date
    test_mask = df['Data'] >= split_date
    
    # Create train/test sets
    X_train = df.loc[train_mask, selected_features]
    X_test = df.loc[test_mask, selected_features]
    y_train = df.loc[train_mask, target_col]
    y_test = df.loc[test_mask, target_col]
    
    return X_train, X_test, y_train, y_test


def get_performance_metrics(y_true, y_pred, y_prob=None):

    metrics = {}
    
    # Basic metrics
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1'] = f1
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['True Negatives'] = tn
    metrics['False Positives'] = fp
    metrics['False Negatives'] = fn
    metrics['True Positives'] = tp
    
    # Additional derived metrics
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['False Positive Rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['False Negative Rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return metrics

def compute_tree_based_importance(X_train, y_train, X_test, y_test, selected_features, model_type='decision_tree', **model_params):

    # Initialize models with their specific parameters
    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=model_params.get('max_depth', 5),
            min_samples_split=model_params.get('min_samples_split', 2),
            random_state=42
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            min_samples_split=model_params.get('min_samples_split', 2),
            random_state=42
        )
    elif model_type == 'gradient_boost':
        model = GradientBoostingClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            learning_rate=model_params.get('learning_rate', 0.1),
            max_depth=model_params.get('max_depth', 3),
            random_state=42
        )
    else:
        raise ValueError(f"Invalid model_type. Choose from ['decision_tree', 'random_forest', 'gradient_boost']")
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Get feature importance
    importance = pd.Series(model.feature_importances_, index=selected_features)
    importance = importance.sort_values(ascending=False)
    
    # Get cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Calculate performance metrics
    performance_metrics = get_performance_metrics(y_test, y_pred, y_prob)
    
    return {
        'model': model,
        'importance_scores': importance,
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'performance_metrics': performance_metrics,
        'predictions': y_pred,
        'probabilities': y_prob
    }

def evaluate_models(df, target_col='Y', model_params=None):

    if model_params is None:
        model_params = {
            'decision_tree': {'max_depth': 5, 'min_samples_split': 5},
            'random_forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
            'gradient_boost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
        }
    
    # Compute correlations and select features
    correlations, corr_results = compute_correlations(df, target_col)
    selected_features = corr_results['selected_features']
    
    # Split the data
    X_train, X_test, y_train, y_test = create_train_test_split(df, selected_features, target_col=target_col)
    
    model_results = {}
    performance_metrics = []
    
    for model_name, params in model_params.items():
        # Compute feature importance and model performance
        results = compute_tree_based_importance(
            X_train, 
            y_train,
            X_test,
            y_test,
            selected_features, 
            model_type=model_name, 
            **params
        )
        
        # Add train/test data to results
        results.update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        })
        
        model_results[model_name] = results
        
        # Collect performance metrics
        metrics_summary = {
            'model': model_name,
            'cv_score': results['cv_score'],
            'cv_std': results['cv_std'],
            'test_accuracy': results['performance_metrics']['Accuracy'],
            'precision': results['performance_metrics']['Precision'],
            'recall': results['performance_metrics']['Recall'],
            'f1_score': results['performance_metrics']['F1']
        }
        performance_metrics.append(metrics_summary)
    
    performance_df = pd.DataFrame(performance_metrics)
    
    return model_results, performance_df

def plot_model_metrics(model, X_train, X_test, y_train, y_test, feature_importance):

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature Importance Plot
    importance_df = feature_importance.head(10)
    sns.barplot(x=importance_df.values, y=importance_df.index, ax=axes[0,0])
    axes[0,0].set_title('Top 10 Feature Importance')
    
    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,1])
    axes[0,1].set_title('Confusion Matrix')
    
    # ROC Curve
    try:
        y_pred_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[1,0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1,0].plot([0, 1], [0, 1], 'k--')
        axes[1,0].set_title('ROC Curve')
        axes[1,0].legend()
    except:
        axes[1,0].text(0.5, 0.5, 'ROC curve not available\nfor this model type', 
                      ha='center', va='center')
    
    # Classification Report
    report = classification_report(y_test, y_pred)
    axes[1,1].text(0.1, 0.1, report, fontsize=10, family='monospace')
    axes[1,1].axis('off')
    axes[1,1].set_title('Classification Report')
    
    plt.tight_layout()
    return fig


def get_relevant_features(df, target_col='Y', n_top_features=10, model_type='decision_tree', model_params={}, plot_metrics=True, split_date='2018-01-01'):

    # Step 1: Compute correlations
    correlations, corr_results = compute_correlations(df, target_col)
    selected_features = corr_results['selected_features']
    
    print(f"Number of features after correlation filtering: {len(selected_features)}")
    
    # Step 2: Split the data
    X_train, X_test, y_train, y_test = create_train_test_split(
        df, 
        selected_features, 
        target_col=target_col,
        split_date=split_date
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Step 3: Compute tree-based importance with specified model
    tree_results = compute_tree_based_importance(
        X_train, y_train, selected_features, 
        model_type=model_type, 
        **model_params
    )
    
    # Get top N features based on importance scores
    final_importance = tree_results['importance_scores'].head(n_top_features)
    final_features = final_importance.index.tolist()
    
    # Generate plots if requested
    plots = None
    if plot_metrics:
        plots = plot_model_metrics(
            tree_results['model'],
            X_train, X_test,
            y_train, y_test,
            tree_results['importance_scores']
        )
    
    return {
        'selected_features': final_features,
        'importance_scores': final_importance,
        'correlation_scores': correlations[final_features],
        'cv_score': tree_results['cv_score'],
        'cv_std': tree_results['cv_std'],
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'model': tree_results['model'],
        'plots': plots,
        'X_train': X_train[final_features],  # Added for convenience
        'X_test': X_test[final_features],    # Added for convenience
        'y_train': y_train,
        'y_test': y_test
    }
    
