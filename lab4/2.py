import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import category_encoders as ce


class AverageEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, model, weight=None):
        self.models_ = None
        self.model = model
        if weight is None:
            self.weight = len(model) * [1 / len(model)]
        else:
            self.weight = weight

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.model]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model, data] * weight for model, weight in zip(range(pred.shape[0]), self.weight)]
            w.append(np.round(np.sum(single)))
        return w


# Load and prepare the data
data = pd.read_csv('data.csv')

# Data Cleaning
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Data encoding
encoder = ce.OrdinalEncoder(cols=['diagnosis'])
data_encoded = encoder.fit_transform(data)

# Feature Scaling
scaler = StandardScaler()
X = data_encoded.drop(['diagnosis'], axis=1)
y = data_encoded['diagnosis']
X_scaled = scaler.fit_transform(X)

# Data Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models for ensemble techniques
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

# 2.1 Max Voting / Voting Classifier
voting_clf = VotingClassifier(estimators=[('LR', lr), ('DT', dt), ('RF', rf)], voting='hard')
voting_clf.fit(X_train, y_train)
voting_accuracy = accuracy_score(y_test, voting_clf.predict(X_test))

# 2.2 Averaging
avg = AverageEstimator(
    model=[LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()],
).fit(X_train, y_train).predict(X_test)

averaging_accuracy = accuracy_score(y_test, avg)

# 2.3 Weighted Averaging
weight_avg = AverageEstimator(
    model=[LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()],
    weight=[0.8, 0.1, 0.1]
).fit(X_train, y_train).predict(X_test)

weighted_averaging_accuracy = accuracy_score(y_test, weight_avg)

# 2.4 Stacking
base_learners = [
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier()),
    ('rf', RandomForestClassifier())
]

stack_clf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
stack_clf.fit(X_train, y_train)
stacking_accuracy = accuracy_score(y_test, stack_clf.predict(X_test))

# 2.5 Blending
X_train_blend, X_val, y_train_blend, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model1 = LogisticRegression().fit(X_train_blend, y_train_blend)
model2 = DecisionTreeClassifier().fit(X_train_blend, y_train_blend)

preds_val1 = model1.predict(X_val)
preds_val2 = model2.predict(X_val)
blend_preds_val = np.column_stack((preds_val1, preds_val2))

blend_meta_model = LogisticRegression().fit(blend_preds_val, y_val)

# Using trained meta-model on test data
preds_test1 = model1.predict(X_test)
preds_test2 = model2.predict(X_test)
blend_preds_test = np.column_stack((preds_test1, preds_test2))

blending_accuracy = accuracy_score(y_test, blend_meta_model.predict(blend_preds_test))

# 2.6 Bagging
bagging_clf = BaggingClassifier(estimator=dt, n_estimators=10, random_state=42)
bagging_clf.fit(X_train, y_train)
bagging_accuracy = accuracy_score(y_test, bagging_clf.predict(X_test))

# 2.7 Boosting - Using AdaBoost as an example
ada_boost = AdaBoostClassifier()
ada_boost.fit(X_train, y_train)
ada_accuracy = accuracy_score(y_test, ada_boost.predict(X_test))

# Hyperparameter Tuning
param_grid = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(dt, param_grid, cv=5)
grid_search.fit(X_train, y_train)
tuned_accuracy = accuracy_score(y_test, grid_search.predict(X_test))

# Results Table
results = pd.DataFrame({
    'Technique': ['Voting', 'Averaging', 'Weighted Averaging', 'Stacking', 'Blending', 'Bagging', 'Boosting',
                  'Tuned Decision Tree'],
    'Accuracy': [voting_accuracy, averaging_accuracy, weighted_averaging_accuracy, stacking_accuracy, blending_accuracy,
                 bagging_accuracy, ada_accuracy, tuned_accuracy]
})

print(results)
