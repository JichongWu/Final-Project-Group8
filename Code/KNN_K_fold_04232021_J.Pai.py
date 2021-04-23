import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import warnings

project = pd.read_csv('project.csv', encoding='ISO-8859-1')

# Data preparation for ML algorithms

# Convert string labels of WEATHER_GROUP into numbers
le = preprocessing.LabelEncoder()
weather_encoded = le.fit_transform(project['WEATHER_GROUP'])

# Convert string labels of LIGHT_GROUP into numbers
le = preprocessing.LabelEncoder()
light_encoded = le.fit_transform(project['LIGHT_GROUP'])

# Convert string labels of SURFACE_GROUP into numbers
le = preprocessing.LabelEncoder()
surface_encoded = le.fit_transform(project['SURFACE_GROUP']) 

# Convert string labels of GRADE_GROUP into numbers
le = preprocessing.LabelEncoder()
grade_encoded = le.fit_transform(project['GRADE_GROUP'])

# Convert string labels of VEHICLE_TYPE into numbers
le = preprocessing.LabelEncoder()
vehicle_encoded = le.fit_transform(project['VEHICLE_TYPE'])

# Convert string labels of ALIGN_GROUP into numbers
le = preprocessing.LabelEncoder()
align_encoded = le.fit_transform(project['ALIGN_GROUP'])

# Convert string labels of MY_GROUP into numbers
le = preprocessing.LabelEncoder()
model_year_encoded = le.fit_transform(project['MY_GROUP'])

# Convert string labels of GENDER into numbers
le = preprocessing.LabelEncoder()
gender_encoded = le.fit_transform(project['GENDER'])

# Convert string labels of AGE_GROUP into numbers
le = preprocessing.LabelEncoder()
age_encoded = le.fit_transform(project['AGE_GROUP'])

# Combine all converted features into a list of tuples
X=list(zip(weather_encoded, light_encoded, surface_encoded, grade_encoded, vehicle_encoded, align_encoded, model_year_encoded,
          gender_encoded, age_encoded))

# Convert string labels of ROLL into numbers
le = preprocessing.LabelEncoder()
y = le.fit_transform(project['ROLL']) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 10-fold cross validation:KNN algorithm
from sklearn.model_selection import cross_val_score

scores = []

# Number of neighborhood needs to be a odd value
# Suggested N is sqrt(n)
# n=61841

KNN_Algorithm = KNeighborsClassifier(n_neighbors = 249)
k_Fold = KFold(n_splits=10, random_state=8, shuffle=True)
cross_val_score(KNN_Algorithm, X_train, y_train, cv=k_Fold, n_jobs=-1)
