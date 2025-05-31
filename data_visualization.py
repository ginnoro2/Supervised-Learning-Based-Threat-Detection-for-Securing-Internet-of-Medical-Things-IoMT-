import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(file_paths):
    """
    Load and combine multiple CSV files into a single DataFrame
    """
    dataframes = []
    labels = []
    
    for label, path in file_paths.items():
        print(f"Loading {path}...")
        df = pd.read_csv(path)
        df['label'] = label
        dataframes.append(df)
        labels.append(label)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df, labels

def create_distribution_plots(train_df, test_df, output_dir):
    """
    Create distribution plots for class distribution in training and testing sets
    """
    plt.figure(figsize=(15, 6))
    
    # Training set distribution
    plt.subplot(1, 2, 1)
    train_dist = train_df['label'].value_counts()
    sns.barplot(x=train_dist.index, y=train_dist.values)
    plt.title('Training Set Class Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Samples')
    
    # Testing set distribution
    plt.subplot(1, 2, 2)
    test_dist = test_df['label'].value_counts()
    sns.barplot(x=test_dist.index, y=test_dist.values)
    plt.title('Testing Set Class Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Samples')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_distribution.png')
    plt.close()

def create_feature_boxplots(train_df, test_df, output_dir):
    """
    Create boxplots for numerical features by class
    """
    numerical_features = train_df.select_dtypes(include=[np.number]).columns
    numerical_features = [f for f in numerical_features if f != 'label']
    
    # Select top 10 features based on variance
    feature_variance = train_df[numerical_features].var()
    top_features = feature_variance.nlargest(10).index
    
    for feature in top_features:
        plt.figure(figsize=(15, 6))
        
        # Training set boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(x='label', y=feature, data=train_df)
        plt.title(f'{feature} Distribution (Training)')
        plt.xticks(rotation=45)
        
        # Testing set boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x='label', y=feature, data=test_df)
        plt.title(f'{feature} Distribution (Testing)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/boxplot_{feature}.png')
        plt.close()

def create_correlation_heatmap(train_df, test_df, output_dir):
    """
    Create correlation heatmaps for numerical features
    """
    numerical_features = train_df.select_dtypes(include=[np.number]).columns
    
    # Training set correlation
    plt.figure(figsize=(12, 10))
    sns.heatmap(train_df[numerical_features].corr(), 
                cmap='coolwarm', 
                center=0,
                annot=False)
    plt.title('Feature Correlation (Training Set)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap_train.png')
    plt.close()
    
    # Testing set correlation
    plt.figure(figsize=(12, 10))
    sns.heatmap(test_df[numerical_features].corr(), 
                cmap='coolwarm', 
                center=0,
                annot=False)
    plt.title('Feature Correlation (Testing Set)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap_test.png')
    plt.close()

def create_feature_density_plots(train_df, test_df, output_dir):
    """
    Create density plots for top features by class
    """
    numerical_features = train_df.select_dtypes(include=[np.number]).columns
    numerical_features = [f for f in numerical_features if f != 'label']
    
    # Select top 6 features based on variance
    feature_variance = train_df[numerical_features].var()
    top_features = feature_variance.nlargest(6).index
    
    plt.figure(figsize=(15, 10))
    for idx, feature in enumerate(top_features, 1):
        plt.subplot(2, 3, idx)
        
        # Plot training set densities
        for label in train_df['label'].unique():
            sns.kdeplot(data=train_df[train_df['label'] == label][feature],
                       label=f'Train-{label}',
                       linestyle='-',
                       alpha=0.5)
        
        # Plot testing set densities
        for label in test_df['label'].unique():
            sns.kdeplot(data=test_df[test_df['label'] == label][feature],
                       label=f'Test-{label}',
                       linestyle='--',
                       alpha=0.5)
        
        plt.title(f'{feature} Density')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_densities.png', bbox_inches='tight')
    plt.close()

def main():
    # Create output directory for plots
    output_dir = 'visualization_results'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Define file paths
    train_files = {
        'benign': 'Benign_train.pcap.csv',
        'arp_spoofing': 'ARP_Spoofing_train.pcap.csv',
        'ddos_tcpip': 'TCP_IP-DDoS-TCP1_train.pcap.csv',
        'mqtt_ddos': 'MQTT-DDoS-Connect_Flood_train.pcap.csv'
    }
    
    test_files = {
        'benign': 'Benign_test.pcap.csv',
        'arp_spoofing': 'ARP_Spoofing_test.pcap.csv',
        'ddos_tcpip': 'TCP_IP-DDoS-ICMP1_test.pcap.csv',
        'mqtt_ddos': 'MQTT-DDoS-Connect_Flood_test.pcap.csv'
    }
    
    # Load data
    print("Loading training data...")
    train_df, train_labels = load_data(train_files)
    
    print("\nLoading testing data...")
    test_df, test_labels = load_data(test_files)
    
    print("\nCreating visualizations...")
    
    # Create various plots
    print("1. Creating class distribution plots...")
    create_distribution_plots(train_df, test_df, output_dir)
    
    print("2. Creating feature boxplots...")
    create_feature_boxplots(train_df, test_df, output_dir)
    
    print("3. Creating correlation heatmaps...")
    create_correlation_heatmap(train_df, test_df, output_dir)
    
    print("4. Creating feature density plots...")
    create_feature_density_plots(train_df, test_df, output_dir)
    
    print(f"\nVisualization complete! Results saved in '{output_dir}' directory.")

if __name__ == "__main__":
    main() 