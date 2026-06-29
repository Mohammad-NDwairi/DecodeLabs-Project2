# Import required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# 1. Load and understand the dataset
print("Loading Iris Dataset...")
iris = load_iris()
X = iris.data  # Features: Sepal Length, Sepal Width, Petal Length, Petal Width
y = iris.target # Target classes: Setosa, Versicolor, Virginica

# 2. Split data into training (80%) and testing (20%) sets with shuffle
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)

# 3. Feature Scaling (StandardScaler: Mean=0, Variance=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Instantiate and train the KNN Classifier (Optimal K=5)
print("Training the KNN Model...")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# 5. Make predictions on the test set
predictions = model.predict(X_test_scaled)

# 6. Output Validation & Evaluation
print("\n=== PROJECT EVALUATION ===")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(target_names=iris.target_names, y_true=y_test, y_pred=predictions))

# Calculate and display the F1 Score (Macro average for multi-class)
f1 = f1_score(y_test, predictions, average='macro')
print(f"\nFinal F1 Score (Macro Average): {f1:.4f}")