import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import category_encoders as ce
from sklearn.preprocessing import label_binarize

# Load the data
data = pd.read_csv('car_evaluation.csv')

# Data Preprocessing
# Encode categorical variables
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
data_encoded = encoder.fit_transform(data)

# Separate features and target variable
X = data_encoded.drop('decision', axis=1)
y = data_encoded['decision']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Building the Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Model Evaluation
# Confusion Matrix
y_pred = rf_model.predict(X_test)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

# Classification Report
classification_report_result = classification_report(y_test, y_pred)

# AUC Score for Multi-Class
# Binarize the output classes for AUC calculation
y_test_binarized = label_binarize(y_test, classes=np.unique(y))
y_pred_proba = rf_model.predict_proba(X_test)

# Compute AUC score using One-vs-Rest approach
auc_score = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr')

# Display Results
print("Confusion Matrix:\n", confusion_matrix_result)
print("\nClassification Report:\n", classification_report_result)
print("\nAUC Score:", auc_score)
