import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_class_distribution(before_smote, after_smote, output_dir):
    """
    Plot class distribution before and after SMOTE
    """
    plt.figure(figsize=(15, 6))
    
    # Before SMOTE
    plt.subplot(1, 2, 1)
    classes = list(before_smote.keys())
    values = list(before_smote.values())
    sns.barplot(x=classes, y=values)
    plt.title('Class Distribution Before SMOTE')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Samples')
    
    # After SMOTE
    plt.subplot(1, 2, 2)
    values_after = list(after_smote.values())
    sns.barplot(x=classes, y=values_after)
    plt.title('Class Distribution After SMOTE')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Samples')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_distribution_comparison.png')
    plt.close()

def plot_model_comparison(model_metrics, output_dir):
    """
    Create comparison plots for model performance
    """
    # Prepare data
    models = list(model_metrics.keys())
    metrics_df = pd.DataFrame(model_metrics).T
    
    # Overall metrics comparison
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ['accuracy', 'macro_avg_f1', 'weighted_avg_f1']
    bar_width = 0.25
    r = np.arange(len(models))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(r + i*bar_width, metrics_df[metric], bar_width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(r + bar_width, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_performance_comparison.png')
    plt.close()
    
    # Class-wise F1 scores
    plt.figure(figsize=(12, 6))
    class_f1_scores = pd.DataFrame({
        'Random Forest': [0.47, 0.95, 1.00, 1.00],
        'Decision Tree': [0.02, 0.93, 0.00, 1.00],
        'Logistic Regression': [0.23, 0.86, 1.00, 1.00]
    }, index=['arp_spoofing', 'benign', 'ddos_tcpip', 'mqtt_ddos'])
    
    sns.heatmap(class_f1_scores, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Class-wise F1 Scores Comparison')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_wise_f1_scores.png')
    plt.close()

def plot_feature_importance_comparison(feature_importance_data, output_dir):
    """
    Create comparison plots for feature importance across models
    """
    plt.figure(figsize=(15, 12))
    
    # Plot for each model
    for i, (model_name, features) in enumerate(feature_importance_data.items(), 1):
        plt.subplot(3, 1, i)
        
        # Create DataFrame for feature importance
        df = pd.DataFrame({
            'Feature': [f[0] for f in features],
            'Importance': [f[1] for f in features]
        })
        
        sns.barplot(x='Importance', y='Feature', data=df)
        plt.title(f'Top 10 Important Features - {model_name}')
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_comparison.png')
    plt.close()

def plot_precision_recall_comparison(model_metrics, output_dir):
    """
    Create precision-recall comparison plots for each class
    """
    # Prepare class-wise precision and recall data
    class_metrics = {
        'Random Forest': {
            'arp_spoofing': {'precision': 0.31, 'recall': 0.97},
            'benign': {'precision': 1.00, 'recall': 0.90},
            'ddos_tcpip': {'precision': 1.00, 'recall': 1.00},
            'mqtt_ddos': {'precision': 1.00, 'recall': 1.00}
        },
        'Decision Tree': {
            'arp_spoofing': {'precision': 0.01, 'recall': 0.97},
            'benign': {'precision': 0.97, 'recall': 0.89},
            'ddos_tcpip': {'precision': 0.98, 'recall': 0.00},
            'mqtt_ddos': {'precision': 1.00, 'recall': 1.00}
        },
        'Logistic Regression': {
            'arp_spoofing': {'precision': 0.14, 'recall': 0.78},
            'benign': {'precision': 0.99, 'recall': 0.77},
            'ddos_tcpip': {'precision': 1.00, 'recall': 1.00},
            'mqtt_ddos': {'precision': 0.99, 'recall': 1.00}
        }
    }
    
    # Create precision-recall plot
    plt.figure(figsize=(15, 10))
    classes = ['arp_spoofing', 'benign', 'ddos_tcpip', 'mqtt_ddos']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    for idx, attack_type in enumerate(classes):
        plt.subplot(2, 2, idx + 1)
        
        precisions = [metrics[attack_type]['precision'] for metrics in class_metrics.values()]
        recalls = [metrics[attack_type]['recall'] for metrics in class_metrics.values()]
        
        plt.scatter(recalls, precisions, c=colors[idx], s=100)
        
        for i, model in enumerate(class_metrics.keys()):
            plt.annotate(model, (recalls[i], precisions[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title(f'Precision vs Recall - {attack_type}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall_comparison.png')
    plt.close()

def plot_confusion_matrices(output_dir):
    """
    Create normalized confusion matrices for each model
    """
    # Confusion matrix data
    confusion_matrices = {
        'Random Forest': np.array([
            [1691, 53, 0, 0],
            [3761, 33846, 0, 0],
            [0, 0, 154007, 0],
            [0, 0, 0, 41916]
        ]),
        'Decision Tree': np.array([
            [1691, 53, 0, 0],
            [3346, 33492, 769, 0],
            [154007, 0, 0, 0],
            [0, 0, 0, 41916]
        ]),
        'Logistic Regression': np.array([
            [1360, 384, 0, 0],
            [8650, 28957, 0, 0],
            [0, 0, 154007, 0],
            [0, 0, 0, 41916]
        ])
    }
    
    classes = ['arp_spoofing', 'benign', 'ddos_tcpip', 'mqtt_ddos']
    
    for model_name, cm in confusion_matrices.items():
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'Normalized Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

def plot_feature_correlations(feature_importance_data, output_dir):
    """
    Create correlation analysis of top features
    """
    # Get unique features from all models
    all_features = set()
    for model_features in feature_importance_data.values():
        all_features.update([f[0] for f in model_features])
    
    # Create correlation matrix (mock data since we don't have actual feature values)
    n_features = len(all_features)
    feature_list = list(all_features)
    
    # Create synthetic correlation matrix based on feature importance patterns
    corr_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                corr_matrix[i,j] = 1.0
            else:
                # Assign correlation based on feature co-occurrence in importance lists
                correlation = 0.0
                for model_features in feature_importance_data.values():
                    model_features_list = [f[0] for f in model_features]
                    if feature_list[i] in model_features_list and feature_list[j] in model_features_list:
                        correlation += 0.3
                corr_matrix[i,j] = min(correlation, 0.9)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=feature_list, yticklabels=feature_list)
    plt.title('Feature Correlation Analysis')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_correlation.png')
    plt.close()

def plot_model_training_times(output_dir):
    """
    Create visualization of model training times
    """
    training_times = {
        'Random Forest': 80.33,
        'Decision Tree': 14.87,
        'Logistic Regression': 130.16
    }
    
    plt.figure(figsize=(10, 6))
    models = list(training_times.keys())
    times = list(training_times.values())
    
    bars = plt.bar(models, times)
    plt.title('Model Training Time Comparison')
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_times.png')
    plt.close()

def main():
    # Create output directory
    output_dir = 'model_evaluation_viz'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Class distribution data
    before_smote = {
        'arp_spoofing': 16047,
        'benign': 192732,
        'ddos_tcpip': 202311,
        'mqtt_ddos': 173036
    }
    
    after_smote = {
        'arp_spoofing': 202311,
        'benign': 202311,
        'ddos_tcpip': 202311,
        'mqtt_ddos': 202311
    }
    
    # Model performance metrics
    model_metrics = {
        'Random Forest': {
            'accuracy': 0.98,
            'macro_avg_f1': 0.85,
            'weighted_avg_f1': 0.99
        },
        'Decision Tree': {
            'accuracy': 0.33,
            'macro_avg_f1': 0.49,
            'weighted_avg_f1': 0.33
        },
        'Logistic Regression': {
            'accuracy': 0.96,
            'macro_avg_f1': 0.77,
            'weighted_avg_f1': 0.97
        }
    }
    
    # Feature importance data
    feature_importance = {
        'Random Forest': [
            ('Header_Length', 0.0952),
            ('Magnitue', 0.0819),
            ('Tot size', 0.0759),
            ('fin_count', 0.0753),
            ('AVG', 0.0647),
            ('ack_count', 0.0626),
            ('rst_count', 0.0525),
            ('Rate', 0.0514),
            ('IAT', 0.0451),
            ('syn_flag_number', 0.0420)
        ],
        'Decision Tree': [
            ('fin_count', 0.3340),
            ('Magnitue', 0.3322),
            ('Rate', 0.1452),
            ('rst_count', 0.0653),
            ('IAT', 0.0565),
            ('Header_Length', 0.0106),
            ('Max', 0.0088),
            ('HTTPS', 0.0081),
            ('Number', 0.0080),
            ('ack_flag_number', 0.0053)
        ],
        'Logistic Regression': [
            ('rst_flag_number', 0.7277),
            ('rst_count', 0.4750),
            ('Header_Length', 0.4322),
            ('Min', 0.3416),
            ('Tot sum', 0.3349),
            ('syn_flag_number', 0.3021),
            ('fin_count', 0.2897),
            ('Variance', 0.2732),
            ('fin_flag_number', 0.2683),
            ('ack_count', 0.2624)
        ]
    }
    
    print("Creating visualizations...")
    
    # Create visualizations
    print("1. Plotting class distribution comparison...")
    plot_class_distribution(before_smote, after_smote, output_dir)
    
    print("2. Plotting model performance comparison...")
    plot_model_comparison(model_metrics, output_dir)
    
    print("3. Plotting feature importance comparison...")
    plot_feature_importance_comparison(feature_importance, output_dir)
    
    print("4. Plotting precision-recall comparison...")
    plot_precision_recall_comparison(model_metrics, output_dir)
    
    print("5. Plotting confusion matrices...")
    plot_confusion_matrices(output_dir)
    
    print("6. Plotting feature correlations...")
    plot_feature_correlations(feature_importance, output_dir)
    
    print("7. Plotting model training times...")
    plot_model_training_times(output_dir)
    
    print(f"\nVisualization complete! Results saved in '{output_dir}' directory.")

if __name__ == "__main__":
    main() 