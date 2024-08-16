# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset from a CSV file
df = pd.read_csv('churn_data.csv')

# Convert categorical variables into dummy/indicator variables
# This is done to handle categorical data for the machine learning model
df = pd.get_dummies(df, drop_first=True)

# Define features (X) and target (y)
# X includes all columns except the 'churn' column
# y is the 'churn' column which we want to predict
X = df.drop('churn', axis=1)
y = df['churn']

# Split the data into training and testing sets
# 80% of the data is used for training and 20% for testing
# random_state is set for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier model
# n_estimators is set to 100, meaning 100 trees in the forest
# random_state is set for reproducibility
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the RandomForestClassifier model on the training data
rf_model.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print confusion matrix to see the performance of the classification model
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Print classification report to see precision, recall, f1-score for each class
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Calculate feature importance from the trained model
# Feature importance helps to understand which features are most influential
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Print the feature importance sorted by importance values in descending order
print(feature_importance_df.sort_values(by='Importance', ascending=False))
# Save the trained model to a file using joblib
# This allows the model to be loaded later for predictions without retraining
joblib.dump(rf_model, 'churn_model.pkl')
