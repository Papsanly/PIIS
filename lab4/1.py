import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load and prepare the data
data = pd.read_csv('data.csv')

# Data Cleaning
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Feature Scaling
scaler = StandardScaler()
X = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']
X_scaled = scaler.fit_transform(X)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models to tune and compare
models = {
    'LogisticRegression': LogisticRegression(),
    'GaussianNB': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

# Parameters for GridSearch
parameters = {
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    },
    'GaussianNB': {},  # GaussianNB doesn't have hyperparameters that are typically tuned
    'DecisionTree': {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'RandomForest': {
        'n_estimators': [10, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
}

# Hyperparameter tuning and evaluation
for model_name, model in models.items():
    grid_search = GridSearchCV(model, parameters[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} - Best Parameters: {grid_search.best_params_}, Accuracy: {accuracy}")

# Comparison with Default Parameters
print("Comparison with Default Parameters:\n")
for model_name, model in models.items():
    default_model = model
    default_model.fit(X_train, y_train)
    y_pred_default = default_model.predict(X_test)
    accuracy_default = accuracy_score(y_test, y_pred_default)
    print(f"{model_name} - Default Parameters Accuracy: {accuracy_default}")
