# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Step 2: Load the dataset
file_path = "C:/Users/sreej/Downloads/Crop_recommendation.csv"  # Update path to your dataset
df = pd.read_csv(file_path)

# Show the first few rows of the dataset
df.head()

# Step 3: Basic Exploration & Data Preprocessing
# Check for missing values and basic info
print("Data Info:")
df.info()

# Check for missing values
print("Missing Values in Dataset:")
print(df.isnull().sum())

# Drop missing values for simplicity (you can apply different strategies for imputation)
df = df.dropna()

# Step 4: Feature Selection using Correlation Matrix
# Exclude non-numeric columns for correlation matrix calculation
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Step 5: Feature Selection using Random Forest
# Using Random Forest to rank feature importance
X = df.drop(columns=["label"])  # All features except the target
y = df["label"]  # The target variable (crop label)

# Train a random forest classifier to find feature importances
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importances
feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                   index=X.columns,
                                   columns=["importance"]).sort_values("importance", ascending=False)

# Display feature importance
print("Feature Importance:")
print(feature_importances)

# Step 6: Selecting Important Features
# Let's select features with an importance score greater than 0.05 (this can be adjusted)
selected_features = feature_importances[feature_importances["importance"] > 0.05].index.tolist()
print(f"Selected Features: {selected_features}")

# Step 7: Train/Test Split
X_selected = X[selected_features]  # Selecting only important features
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Step 8: Feature Scaling (optional but recommended for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Model Training (Logistic Regression)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 10: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 11: Model Evaluation

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 12: Conclusion
# Displaying the results of feature selection, training, and evaluation.
print("Feature Selection and Model Evaluation complete.")
