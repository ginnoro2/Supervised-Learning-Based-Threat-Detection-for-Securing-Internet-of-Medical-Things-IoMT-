import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from pathlib import Path
from sklearn.preprocessing import label_binarize

def plot_model_comparison(results, output_dir):
    """
    Create bar plots comparing model performance metrics
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    model_names = list(results.keys())
    
    # Prepare data for plotting
    plot_data = []
    for model_name, result in results.items():
        metrics_dict = {
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision': result['weighted_precision'],
            'Recall': result['weighted_recall'],
            'F1-score': result['weighted_f1']
        }
        plot_data.append(metrics_dict)
    
    df = pd.DataFrame(plot_data)
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, df[metric], width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*1.5, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png')
    plt.close()

def plot_class_wise_performance(results, output_dir):
    """
    Create heatmaps showing class-wise performance for each model
    """
    for model_name, result in results.items():
        plt.figure(figsize=(10, 8))
        
        # Create matrix of class-wise metrics
        metrics_df = pd.DataFrame({
            'Precision': result['class_precision'],
            'Recall': result['class_recall'],
            'F1-score': result['class_f1']
        })
        
        # Plot heatmap
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title(f'Class-wise Performance Metrics - {model_name}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/class_performance_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

def plot_roc_curves(models, X_test, y_test, class_names, output_dir):
    """
    Plot ROC curves for all models and classes
    """
    plt.figure(figsize=(12, 8))
    
    # Binarize the labels for ROC curve
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    
    for model_name, model in models.items():
        # Get predictions
        y_score = model.predict_proba(X_test)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curves
            plt.plot(fpr[i], tpr[i],
                    label=f'{model_name} - {class_names[i]} (AUC = {roc_auc[i]:.2f})',
                    alpha=0.6)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models and Classes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', bbox_inches='tight')
    plt.close()

def plot_feature_importance_comparison(results, feature_names, output_dir):
    """
    Create comparison plots for feature importance across models
    """
    # Get top 15 features from each model
    top_features = {}
    for model_name, result in results.items():
        if 'feature_importance' in result:
            importance = result['feature_importance']
            top_features[model_name] = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).nlargest(15, 'importance')
    
    if top_features:
        plt.figure(figsize=(15, 10))
        
        # Plot feature importance for each model
        n_models = len(top_features)
        for idx, (model_name, feat_imp) in enumerate(top_features.items(), 1):
            plt.subplot(n_models, 1, idx)
            sns.barplot(x='importance', y='feature', data=feat_imp)
            plt.title(f'Top 15 Important Features - {model_name}')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance_comparison.png')
        plt.close()

def plot_confusion_matrices(results, class_names, output_dir):
    """
    Plot normalized confusion matrices for all models
    """
    for model_name, result in results.items():
        plt.figure(figsize=(10, 8))
        cm = result['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Normalized Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_normalized_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

def plot_training_times(results, output_dir):
    """
    Create bar plot comparing training times
    """
    model_names = list(results.keys())
    training_times = [result['training_time'] for result in results.values()]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=training_times)
    plt.title('Training Time Comparison')
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_times.png')
    plt.close()

def evaluate_and_visualize(models, X_test, y_test, feature_names, class_names, output_dir):
    """
    Main function to evaluate models and create visualizations
    """
    Path(output_dir).mkdir(exist_ok=True)
    results = {}
    
    for model_name, model in models.items():
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Store results
        results[model_name] = {
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'weighted_precision': report['weighted avg']['precision'],
            'weighted_recall': report['weighted avg']['recall'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'class_precision': [report[c]['precision'] for c in class_names],
            'class_recall': [report[c]['recall'] for c in class_names],
            'class_f1': [report[c]['f1-score'] for c in class_names],
            'training_time': model.training_time if hasattr(model, 'training_time') else 0
        }
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            results[model_name]['feature_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            results[model_name]['feature_importance'] = np.abs(model.coef_).mean(axis=0)
    
    # Create visualizations
    print("Creating model evaluation visualizations...")
    
    print("1. Plotting model comparison...")
    plot_model_comparison(results, output_dir)
    
    print("2. Plotting class-wise performance...")
    plot_class_wise_performance(results, output_dir)
    
    print("3. Plotting ROC curves...")
    plot_roc_curves(models, X_test, y_test, class_names, output_dir)
    
    print("4. Plotting feature importance comparison...")
    plot_feature_importance_comparison(results, feature_names, output_dir)
    
    print("5. Plotting confusion matrices...")
    plot_confusion_matrices(results, class_names, output_dir)
    
    print("6. Plotting training times...")
    plot_training_times(results, output_dir)
    
    print(f"\nEvaluation visualizations complete! Results saved in '{output_dir}' directory.")
    
    return results 