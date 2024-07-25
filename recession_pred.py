import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Read the CSV file
csv_file_path = 'data.csv'
df = pd.read_csv(csv_file_path)

# Split the data into training and testing sets
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Define features and targets
features = [
    'Inflation_rate',
    'Intreast_rate',
    'Unemployement_rate',
    's&p_index_rate',
    '(10Y_Treasury_Rate)-(5Y_Treasury_Rate)',
    '(10Y_Treasury_Rate)-(3_Month_T-Bill_Rate)'
]
targets = ['Recession_in_6mo', 'Recession_in_12mo', 'Recession_in_24mo']

# Scale the features
scaler = StandardScaler()
df_train[features] = scaler.fit_transform(df_train[features])
df_test[features] = scaler.transform(df_test[features])

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Compare Accuracies
results = {}
for model_name, model in models.items():
    for target in targets:
        model.fit(df_train[features], df_train[target])
        
        predictions = model.predict(df_test[features])
        accuracy = accuracy_score(df_test[target], predictions)
        conf_matrix = confusion_matrix(df_test[target], predictions)
        class_report = classification_report(df_test[target], predictions)
        
        if model_name not in results:
            results[model_name] = {}
        results[model_name][target] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
}   

# Display results
for model_name, result in results.items():
    print(f"Model: {model_name}")
    for target, metrics in result.items():
        print(f"  Target: {target}")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Confusion Matrix:\n{metrics['confusion_matrix']}")
        print(f"    Classification Report:\n{metrics['classification_report']}")
