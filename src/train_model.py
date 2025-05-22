import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load features
script_dir = os.path.dirname(os.path.abspath(__file__))
features_path = os.path.join(script_dir, '..', 'features', 'voice_features.csv')
print(f"Loading data from: {features_path}")
df = pd.read_csv(features_path)

# Split into features and labels
X = df.drop(columns=['label'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model
models_dir = os.path.join(script_dir, '..', 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'voice_gender_rf_model.joblib')
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")
print(df['label'].value_counts())

