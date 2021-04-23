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

import statsmodels.api as sm


model = sm.GLM.from_formula("ROLL ~ WEATHER+LGT_COND + SURFACE_GROUP + GRADE_GROUP + ALIGN_GROUP + VEHICLE_TYPE + MY_GROUP + GENDER + AGE_GROUP", 
                            family = sm.families.Binomial(), data=project)
result = model.fit()
result.summary()

project_logistic = project.copy()

# Re-group vehicle type
vehicle_type_map = {1:'CAR/VAN', 2:'CAR/VAN', 3:'CAR/VAN', 4:'CAR/VAN', 5:'CAR/VAN', 6:'CAR/VAN', 7:'CAR/VAN', 8:'CAR/VAN', 
                    9:'CAR/VAN', 10:'NA', 12:'CAR/VAN', 14:'SUV/CUV', 15:'SUV/CUV', 16:'SUV/CUV', 17:'CAR/VAN', 19:'NA', 
                    20:'CAR/VAN', 21:'CAR/VAN', 22:'CAR/VAN', 28:'CAR/VAN', 29:'CAR/VAN', 30:'PICKUP', 31:'PICKUP', 
                    32:'PICKUP', 33:'PICKUP', 34:'PICKUP', 39:'PICKUP', 40:'PICKUP'}
project_logistic['NEW_VEHICLE_TYPE'] = project_logistic['BODY_TYP'].map(vehicle_type_map)

# Re-group driver's age
condition = [
    (project_logistic['AGE'] >= 16) & (project_logistic['AGE'] <= 24),
    (project_logistic['AGE'] >= 25) & (project_logistic['AGE'] <= 34),
    (project_logistic['AGE'] >= 35) & (project_logistic['AGE'] <= 44),
    (project_logistic['AGE'] >= 45) & (project_logistic['AGE'] <= 54),
    (project_logistic['AGE'] >= 55) & (project_logistic['AGE'] <= 64),
    (project_logistic['AGE'] >= 65) & (project_logistic['AGE'] <= 74),
    (project_logistic['AGE'] >= 75) & (project_logistic['AGE'] <= 103),
    ]
age_group_value = ['16-24', '25-34', '35-44', '45-54', '55-64', '65-74', '> 74']
project_logistic['NEW_AGE_GROUP'] = np.select(condition, age_group_value)

model = sm.GLM.from_formula("ROLL ~ LGT_COND + SURFACE_GROUP + GRADE_GROUP + ALIGN_GROUP + NEW_VEHICLE_TYPE + MY_GROUP + GENDER + NEW_AGE_GROUP", 
                            family = sm.families.Binomial(), data=project_logistic)
result = model.fit()
result.summary()

# Data preparation for ML algorithms

# Convert string labels of LIGHT_GROUP into numbers
le = preprocessing.LabelEncoder()
light_encoded = le.fit_transform(project_logistic['LIGHT_GROUP'])

# Convert string labels of SURFACE_GROUP into numbers
le = preprocessing.LabelEncoder()
surface_encoded = le.fit_transform(project_logistic['SURFACE_GROUP']) 

# Convert string labels of GRADE_GROUP into numbers
le = preprocessing.LabelEncoder()
grade_encoded = le.fit_transform(project_logistic['GRADE_GROUP'])

# Convert string labels of VEHICLE_TYPE into numbers
le = preprocessing.LabelEncoder()
new_vehicle_encoded = le.fit_transform(project_logistic['NEW_VEHICLE_TYPE'])

# Convert string labels of ALIGN_GROUP into numbers
le = preprocessing.LabelEncoder()
align_encoded = le.fit_transform(project_logistic['ALIGN_GROUP'])

# Convert string labels of MY_GROUP into numbers
le = preprocessing.LabelEncoder()
model_year_encoded = le.fit_transform(project_logistic['MY_GROUP'])

# Convert string labels of GENDER into numbers
le = preprocessing.LabelEncoder()
gender_encoded = le.fit_transform(project_logistic['GENDER'])

# Convert string labels of AGE_GROUP into numbers
le = preprocessing.LabelEncoder()
new_age_encoded = le.fit_transform(project_logistic['NEW_AGE_GROUP'])

# Combine all converted features into a list of tuples
NEW_X=list(zip(light_encoded, surface_encoded, grade_encoded, new_vehicle_encoded, align_encoded, model_year_encoded,
          gender_encoded, new_age_encoded))

# Convert string labels of ROLL into numbers
le = preprocessing.LabelEncoder()
y = le.fit_transform(project_logistic['ROLL']) 

from sklearn.model_selection import train_test_split
NEW_X_train, NEW_X_test, y_train, y_test = train_test_split(NEW_X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
scaler.fit(NEW_X_train)
X_train = scaler.transform(NEW_X_train)
X_test = scaler.transform(NEW_X_test)

# 10-fold cross validation:logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

scores = []

Logistic_Algorithm = LogisticRegression()
K_Fold = KFold(n_splits=10, random_state=8, shuffle=True)
cross_val_score(Logistic_Algorithm, NEW_X_train, y_train, cv=K_Fold, n_jobs=-1)
