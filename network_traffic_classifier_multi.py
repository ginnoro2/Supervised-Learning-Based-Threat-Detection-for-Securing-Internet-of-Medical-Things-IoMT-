import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import time
from imblearn.over_sampling import SMOTE
from model_evaluation_viz import evaluate_and_visualize

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
        print(f"Shape for {label}: {df.shape}")
        dataframes.append(df)
        labels.append(label)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nCombined shape: {combined_df.shape}")
    return combined_df, labels

def train_model(model, X_train, y_train, model_name):
    """
    Train a model with SMOTE balancing
    """
    # First apply RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Apply SMOTE
    print(f"\nApplying SMOTE for {model_name}...")
    print("Class distribution before SMOTE:")
    for label, count in zip(np.unique(y_train), np.bincount(y_train)):
        print(f"{label}: {count}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print("\nClass distribution after SMOTE:")
    for label, count in zip(np.unique(y_train_resampled), np.bincount(y_train_resampled)):
        print(f"{label}: {count}")
    
    # Train the model
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    model.fit(X_train_resampled, y_train_resampled)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Store training time as attribute
    model.training_time = training_time
    model.scaler = scaler
    
    return model, scaler

def main():
    start_time = time.time()
    
    # Create output directory for evaluation results
    output_dir = 'model_evaluation_results'
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
    
    # Prepare features and labels
    X = train_df.drop('label', axis=1)
    feature_names = X.columns
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_df['label'])
    class_names = label_encoder.classes_
    
    # Print initial class distribution
    print("\nInitial class distribution:")
    for label, count in zip(class_names, np.bincount(y)):
        print(f"{label}: {count}")
    
    # Load and preprocess test data
    print("\nLoading testing data...")
    test_df, _ = load_data(test_files)
    X_test = test_df.drop('label', axis=1)
    y_test = label_encoder.transform(test_df['label'])
    
    # Print test set distribution
    print("\nTest set class distribution:")
    for label, count in zip(class_names, np.bincount(y_test)):
        print(f"{label}: {count}")
    
    # Define models to evaluate
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            multi_class='multinomial',
            solver='lbfgs',
            n_jobs=-1,
            random_state=42
        )
    }
    
    # Train models and store results
    trained_models = {}
    scalers = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training and evaluating {model_name}")
        print('='*50)
        
        trained_model, scaler = train_model(model, X, y, model_name)
        trained_models[model_name] = trained_model
        scalers[model_name] = scaler
    
    # Scale test data for each model
    X_test_scaled = {}
    for model_name, scaler in scalers.items():
        X_test_scaled[model_name] = scaler.transform(X_test)
    
    # Evaluate models and create visualizations
    results = evaluate_and_visualize(
        trained_models,
        {name: X_test_scaled[name] for name in trained_models.keys()},
        y_test,
        feature_names,
        class_names,
        output_dir
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 