import pandas as pd
import numpy as np
# import researchpy as rp
from scipy import stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px
#import plotly.figure_factory as ff
import warnings

project = pd.read_csv('https://raw.githubusercontent.com/JichongWu/Final-Project-Group8/main/Data/project_04142021_J.Pai.csv')

# KNN models with N=5, 10, and 15

from sklearn import preprocessing

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
gender_encoded = le.fit_transform(project['GENDER_GROUP'])

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# KNN model with number of neighborhood at 5
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
result1 = accuracy_score(y_test,y_pred)
print('Accuracy of KNN model with the number of number of neighborhood at 5:', result1)

# KNN model with number of neighborhood at 10
classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
result2 = accuracy_score(y_test,y_pred)
print('Accuracy of KNN model with the number of number of neighborhood at 10:', result2)

# KNN model with number of neighborhood at 15
classifier = KNeighborsClassifier(n_neighbors = 15)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
result3 = accuracy_score(y_test,y_pred)
print('Accuracy of KNN model with the number of number of neighborhood at 15:', result3)

######################################################################################

# Logistic regression model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)

print('Accuracy of logistic regression model:',logistic_regression.score(X_test, y_test))
