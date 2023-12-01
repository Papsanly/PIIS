import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import category_encoders as ce
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize


# Функція для побудови матриці помилок
def plot_confusion_matrix(cm, title, ax):
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix - ' + title)


# Load the data
data = pd.read_csv('car_evaluation.csv')

# Data Preprocessing
# Encode categorical variables
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'decision'])
data_encoded = encoder.fit_transform(data)

# Separate features and target variable
X = data_encoded.drop('decision', axis=1)
y = data_encoded['decision']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

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

_, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_confusion_matrix(confusion_matrix(y_test, y_pred), 'Random Forest', ax[0])

# Тренування моделі XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Adjust the labels in your target variable to start from 0
y_train = y_train - 1
y_test = y_test - 1

xgb_model.fit(X_train, y_train)

# Оцінка моделі
y_pred_xgb = xgb_model.predict(X_test)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
class_report_xgb = classification_report(y_test, y_pred_xgb)

# AUC Score для багатокласової класифікації
y_test_binarized = label_binarize(y_test, classes=np.unique(y))
y_pred_proba_xgb = xgb_model.predict_proba(X_test)

# Відображення результатів
print("XGBoost Confusion Matrix:\n", conf_matrix_xgb)
print("\nXGBoost Classification Report:\n", class_report_xgb)


# Функція для побудови кривих ROC
def plot_roc_curve(y_test, y_pred_proba, model_name, ax):
    # Binarize y_test for multi-class ROC curve
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                      ''.format(i, roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - ' + model_name)
    ax.legend(loc="lower right")

plot_confusion_matrix(confusion_matrix(y_test, y_pred_xgb), 'XGBoost', ax[1])

_, ax = plt.subplots(figsize=(8, 6))
plot_roc_curve(y_test, y_pred_proba, 'Random Forest', ax)

plt.show()
