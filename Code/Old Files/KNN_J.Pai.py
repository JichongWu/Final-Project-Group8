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

# Data preparation for KNN algorithms

# Convert string labels of WEATHER_GROUP into numbers: 0:CLEAR/NORMAL, 1:FOG/CLOUDY, 2:RAIN/SLEET, 3:SNOW, 4:WINDY
le = preprocessing.LabelEncoder()
weather_encoded = le.fit_transform(project_KNN['WEATHER_GROUP'])

# Convert string labels of LIGHT_GROUP into numbers: 0:DARK, 1:DAWN/DUSK, 2:LIGHT
le = preprocessing.LabelEncoder()
light_encoded = le.fit_transform(project_KNN['LIGHT_GROUP'])

# Convert string labels of SURFACE_GROUP into numbers: 0:DRY, 1:OIL, 2:WET
le = preprocessing.LabelEncoder()
surface_encoded = le.fit_transform(project_KNN['SURFACE_GROUP']) 

# Convert string labels of GRADE_GROUP into numbers: 0:GRADE, 1:LEVEL
le = preprocessing.LabelEncoder()
grade_encoded = le.fit_transform(project_KNN['GRADE_GROUP'])

# Convert string labels of VEHICLE_TYPE into numbers: 0:CAR, 1:PICKUP, 2:SUV/CUV, 3:VAN
le = preprocessing.LabelEncoder()
vehicle_encoded = le.fit_transform(project_KNN['VEHICLE_TYPE'])

# Convert string labels of ALIGN_GROUP into numbers: 0:CURVE, 1:STRAIGHT
le = preprocessing.LabelEncoder()
align_encoded = le.fit_transform(project_KNN['ALIGN_GROUP'])

# Convert string labels of MY_GROUP into numbers: 0:1989-2007, 1:2008-2010, 2:2011-2019
le = preprocessing.LabelEncoder()
model_year_encoded = le.fit_transform(project_KNN['MY_GROUP'])

# Convert string labels of GENDER into numbers: 0:FEMALE, 1:MALE
le = preprocessing.LabelEncoder()
gender_encoded = le.fit_transform(project_KNN['GENDER'])

# Standarize AGE
column = 'AGE'
age_normal = (project[column] - project[column].min()) / (project[column].max() - project[column].min())

# Combine all converted features into a list of tuples
X=list(zip(weather_encoded, light_encoded, surface_encoded, grade_encoded, vehicle_encoded, align_encoded, model_year_encoded,
          gender_encoded, age_normal))

# Convert string labels of ROLL into numbers: 0:ROLLOVER, 1:NON-ROLLOVER
le = preprocessing.LabelEncoder()
y = le.fit_transform(project['ROLL']) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8, test_size = 0.3)

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
# Use sqrt(n), where n=43288

KNN_Algorithm = KNeighborsClassifier(n_neighbors = 209)
k_Fold = KFold(n_splits=10, random_state=8, shuffle=True)
scores = cross_val_score(KNN_Algorithm, X_train, y_train, cv=k_Fold, n_jobs=-1)

print(scores)
