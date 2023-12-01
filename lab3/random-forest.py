from fastai.imports import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv(path/'train.csv')
tst_df = pd.read_csv(path/'test.csv')

# Data preprocessing
def proc_data(df):
    df['Fare'] = df.Fare.fillna(0)
    df.fillna(modes, inplace=True)
    df['LogFare'] = np.log1p(df['Fare'])
    df['Embarked'] = pd.Categorical(df.Embarked)
    df['Sex'] = pd.Categorical(df.Sex)

proc_data(df)
proc_data(tst_df)

# Define variables
cats = ["Sex", "Embarked"]
conts = ['Age', 'SibSp', 'Parch', 'LogFare', "Pclass"]
dep = "Survived"

# Split data into training and validation sets
random.seed(42)
trn_df, val_df = train_test_split(df, test_size=0.25)
trn_df[cats] = trn_df[cats].apply(lambda x: x.cat.codes)
val_df[cats] = val_df[cats].apply(lambda x: x.cat.codes)

# Create independent and dependent variables
def xs_y(df):
    xs = df[cats + conts].copy()
    return xs, df[dep] if dep in df else None

trn_xs, trn_y = xs_y(trn_df)
val_xs, val_y = xs_y(val_df)

# Create and train random forest model
rf = RandomForestClassifier(100, min_samples_leaf=5)
rf.fit(trn_xs, trn_y)

# Make predictions on the validation set
predictions = rf.predict(val_xs)

# Evaluate the model
mean_absolute_error(val_y, predictions)
