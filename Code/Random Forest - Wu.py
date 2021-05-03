import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# roc_auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve

project = pd.read_csv('https://raw.githubusercontent.com/JichongWu/Final-Project-Group8/main/Data/project%20dataset%20FINAL.csv', sep=',',encoding='ISO-8859-1')

cat_vars=['ROLL','WEATHER_GROUP','LIGHT_Condition','ROAD_SURFACE','ROADWAY_GRADE','ROADWAY_ALIGNMENT','VEHICLE_TYPE','VEHICLE_YEAR','GENDER']

for var in cat_vars:
    cat_list ='var'+'_'+var
    cat_list = pd.get_dummies(project[var], prefix=var)
    data1 = project.join(cat_list)
    project = data1

data_vars = project.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
data_final = project[to_keep]

# Combine variables into a list of tuples

AGE = data_final['AGE']
WEATHER_CLEAR_NORMAL = data_final['WEATHER_GROUP_CLEAR/NORMAL']
WEATHER_FOG_CLOUDY = data_final['WEATHER_GROUP_FOG/CLOUDY']
WEATHER_RAIN_SLEET = data_final['WEATHER_GROUP_RAIN/SLEET']
WEATHER_SNOW = data_final['WEATHER_GROUP_SNOW']
LIGHT_DAWN_DUSK = data_final['LIGHT_Condition_DAWN/DUSK']
LIGHT_LIGHT = data_final['LIGHT_Condition_LIGHT']
SURFACE_DRY = data_final['ROAD_SURFACE_DRY']
SURFACE_WET = data_final['ROAD_SURFACE_WET']
GRADE_LEVEL = data_final['ROADWAY_GRADE_LEVEL']
ALIGN_STRAIGHT = data_final['ROADWAY_ALIGNMENT_STRAIGHT']
VEHICLE_CAR = data_final['VEHICLE_TYPE_CAR']
VEHICLE_PICKUP = data_final['VEHICLE_TYPE_PICKUP']
VEHICLE_VAN = data_final['VEHICLE_TYPE_VAN']
MY_2008_2019 = data_final['VEHICLE_YEAR_2008-2019']
MY_2011_2019 = data_final['VEHICLE_YEAR_2011-2019']
GENDER_FEMALE = data_final['GENDER_FEMALE']

# Features
X=list(zip(AGE,WEATHER_CLEAR_NORMAL,WEATHER_FOG_CLOUDY,WEATHER_RAIN_SLEET,WEATHER_SNOW,LIGHT_DAWN_DUSK,LIGHT_LIGHT,
           SURFACE_DRY,SURFACE_WET,GRADE_LEVEL,ALIGN_STRAIGHT,VEHICLE_CAR,VEHICLE_PICKUP,VEHICLE_VAN,MY_2008_2019,
           MY_2011_2019,GENDER_FEMALE))

# Target
Le = LabelEncoder()
y = Le.fit_transform(project['ROLL'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_rep = classification_report(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred) * 100
print(accuracy_score)

# confusion matrix plot
class_names = ['Rollover', 'Non-Rollover']
plt.matshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(conf_matrix[i][j]))


# feature selection plot
importances = clf.feature_importances_

new_feature_list_17 = ['AGE', 'WEATHER_GROUP_CLEAR/NORMAL','WEATHER_GROUP_FOG/CLOUDY','WEATHER_GROUP_RAIN/SLEET','WEATHER_GROUP_SNOW',
                        'LIGHT_Condition_DAWN/DUSK','LIGHT_Condition_LIGHT','ROAD_SURFACE_DRY','ROAD_SURFACE_WET','ROADWAY_GRADE_LEVEL',
                        'ROADWAY_ALIGNMENT_STRAIGHT','VEHICLE_TYPE_CAR','VEHICLE_TYPE_PICKUP','VEHICLE_TYPE_VAN','VEHICLE_YEAR_2008-2019',
                        'VEHICLE_YEAR_2011-2019','GENDER_FEMALE']

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, new_feature_list_17)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=True, inplace=True)

X_Features = f_importances.index
y_Importance = list(f_importances)
plt.barh(X_Features, y_Importance )