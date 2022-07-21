import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')

df = pd.read_csv('winequality.csv')
df.head()

# statistical info
df.describe()

# datatype info
df.info()

# check for null values
df.isnull().sum()

# fill the missing values
for col, value in df.items():
    if col != 'type':
        df[col] = df[col].fillna(df[col].mean())
        
df.isnull().sum()

Exploratory Data Analysis
# create box plots
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    if col != 'type':
        sns.boxplot(y=col, data=df, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

# create dist plot
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    if col != 'type':
        sns.distplot(value, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

# log transformation
df['free sulfur dioxide'] = np.log(1 + df['free sulfur dioxide'])
sns.distplot(df['free sulfur dioxide'])

sns.countplot(df['type'])

sns.countplot(df['quality'])

corr = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')

#Input Split
X = df.drop(columns=['type', 'quality'])
y = df['quality']

y.value_counts()

from imblearn.over_sampling import SMOTE  #synthetic Minority over sample for performing over sampling
oversample = SMOTE(k_neighbors=4)
# transform the dataset
X, y = oversample.fit_resample(X, y)

y.value_counts()

#Model Training
# classify function
from sklearn.model_selection import cross_val_score, train_test_split
def classify(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # train the model
    model.fit(x_train, y_train)
    print("Accuracy:", model.score(x_test, y_test) * 100)
    
    # cross-validation
    score = cross_val_score(model, X, y, cv=5)
    print("CV Score:", np.mean(score)*100)
    
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model, X, y)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model, X, y)

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
classify(model, X, y)

import xgboost as xgb
model = xgb.XGBClassifier()
classify(model, X, y)

import lightgbm 
model = lightgbm.LGBMClassifier()
classify(model, X, y)
