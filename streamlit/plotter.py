import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred):
    """Plot and return confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    return fig

def plot_feature_importances(feature_importances):
    """Plot and return feature importances figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if feature_importances is not None:
        top_n = feature_importances.head(10)
        sns.barplot(x=top_n.values, y=top_n.index, ax=ax)
        ax.set_title('Top 10 Feature Importances')
    else:
        ax.text(0.5, 0.5, 'No Feature Importances Available',
                ha='center', va='center')
        ax.set_title('Feature Importances')
    
    return fig

def plot_roc_curve(y_true, y_pred_proba):
    """Plot and return ROC curve figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'r--')
        ax.legend(loc='lower right')
        ax.set_title('ROC Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
    else:
        ax.text(0.5, 0.5, 'No Probability Predictions Available',
                ha='center', va='center')
        ax.set_title('ROC Curve')
    
    return fig

def plot_model_results(y_true, y_pred, y_pred_proba, feature_importances):
    """Create combined plot with all metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    sns.set_style('whitegrid')
    
    plot_confusion_matrix(y_true, y_pred, ax=axes[0])
    plot_feature_importances(feature_importances, ax=axes[1])
    plot_roc_curve(y_true, y_pred_proba, ax=axes[2])
    
    plt.tight_layout()
    return fig