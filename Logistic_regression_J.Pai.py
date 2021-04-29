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
project_logit = project.copy()

cat_vars=['ROLL','WEATHER_GROUP','LIGHT_GROUP','SURFACE_GROUP','GRADE_GROUP','ALIGN_GROUP','VEHICLE_TYPE','MY_GROUP','GENDER']

for var in cat_vars:
    cat_list ='var'+'_'+var
    cat_list = pd.get_dummies(project_logit[var], prefix=var)
    data1 = project_logit.join(cat_list)
    project_logit = data1

data_vars = project_logit.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
data_final = project_logit[to_keep]

# Combine variables into a list of tuples

# NON_ROLLOVER = data_final['ROLL_NON-ROLLOVER'] # reference group for ROLL
ROLLOVER = data_final['ROLL_ROLLOVER']

AGE = data_final['AGE']

WEATHER_CLEAR_NORMAL = data_final['WEATHER_GROUP_CLEAR/NORMAL']
WEATHER_FOG_CLOUDY = data_final['WEATHER_GROUP_FOG/CLOUDY']
WEATHER_RAIN_SLEET = data_final['WEATHER_GROUP_RAIN/SLEET']
WEATHER_SNOW = data_final['WEATHER_GROUP_SNOW']
# WEATHER_WINDY = data_final['WEATHER_GROUP_WINDY'] # reference group for WEATHER_GROUP

# LIGHT_DARK = data_final['LIGHT_GROUP_DARK'] # reference group for LIGHT_GROUP
LIGHT_DAWN_DUSK = data_final['LIGHT_GROUP_DAWN/DUSK']
LIGHT_LIGHT = data_final['LIGHT_GROUP_LIGHT']

SURFACE_DRY = data_final['SURFACE_GROUP_DRY']
# SURFACE_OIL = data_final['SURFACE_GROUP_OIL'] # reference group for SURFACE_GROUP
SURFACE_WET = data_final['SURFACE_GROUP_WET']

# GRADE_GRADE = data_final['GRADE_GROUP_GRADE'] # reference group for GRADE_GROUP
GRADE_LEVEL = data_final['GRADE_GROUP_LEVEL']

# ALIGN_CURVE = data_final['ALIGN_GROUP_CURVE'] # reference group for ALIGN_GROUP
ALIGN_STRAIGHT = data_final['ALIGN_GROUP_STRAIGHT']

VEHICLE_CAR = data_final['VEHICLE_TYPE_CAR']
VEHICLE_PICKUP = data_final['VEHICLE_TYPE_PICKUP']
# VEHICLE_SUV_CUV = data_final['VEHICLE_TYPE_SUV/CUV'] # reference group for VEHICLE_TYPE
VEHICLE_VAN = data_final['VEHICLE_TYPE_VAN']

# MY_1989_2007 = data_final['MY_GROUP_1989-2007'] # # reference group for MY_GROUP
MY_2008_2019 = data_final['MY_GROUP_2008-2019']
MY_2011_2019 = data_final['MY_GROUP_2011-2019']

GENDER_FEMALE = data_final['GENDER_FEMALE'] 
# GENDER_MALE = data_final['GENDER_MALE'] # reference group for GENDER

# Features
X=list(zip(AGE,WEATHER_CLEAR_NORMAL,WEATHER_FOG_CLOUDY,WEATHER_RAIN_SLEET,WEATHER_SNOW,LIGHT_DAWN_DUSK,LIGHT_LIGHT,
           SURFACE_DRY,SURFACE_WET,GRADE_LEVEL,ALIGN_STRAIGHT,VEHICLE_CAR,VEHICLE_PICKUP,VEHICLE_VAN,MY_2008_2019,
           MY_2011_2019,GENDER_FEMALE))

# Target
y = list(ROLLOVER) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8, test_size = 0.3)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

logit_model = sm.Logit(y_train, X_train).fit()
print('X1: AGE')
print('X2: WEATHER_GROUP_CLEAR/NORMAL')
print('X3: WEATHER_GROUP_FOG/CLOUDY')
print('X4: WEATHER_GROUP_RAIN/SLEET')
print('X5: WEATHER_GROUP_SNOW')
print('X6: LIGHT_GROUP_DAWN/DUSK')
print('X7: LIGHT_GROUP_LIGHT')
print('X8: SURFACE_GROUP_DRY')
print('X9: SURFACE_GROUP_WET')
print('X10: GRADE_GROUP_LEVEL')
print('X11: ALIGN_GROUP_STRAIGHT')
print('X12: VEHICLE_TYPE_CAR')
print('X13: VEHICLE_TYPE_PICKUP')
print('X14: VEHICLE_TYPE_VAN')
print('X15: MY_GROUP_2008-2019')
print('X16: MY_GROUP_2011-2019')
print('X17: GENDER_FEMALE') 
print(logit_model.summary())

project_logit_2 = project.copy()

# Group roadway surface condition
vsurcond_map = {0:'NA', 1:'DRY', 2:'WET/OIL/SLUSH/MUD', 3:'WET/OIL/SLUSH/MUD', 4:'WET/OIL/SLUSH/MUD', 5:'DRY', 
                6:'WET/OIL/SLUSH/MUD', 7:'WET/OIL/SLUSH/MUD', 8:'NA', 10:'WET/OIL/SLUSH/MUD', 11:'WET/OIL/SLUSH/MUD', 
                98:'NA', 99:'NA'}
project_logit_2['SURFACE_GROUP'] = project_logit_2['VSURCOND'].map(vsurcond_map)

cat_vars_2=['ROLL', 'WEATHER_GROUP','LIGHT_GROUP','SURFACE_GROUP','GRADE_GROUP','ALIGN_GROUP','VEHICLE_TYPE','MY_GROUP']

for var in cat_vars_2:
    cat_list_2='var'+'_'+var
    cat_list_2 = pd.get_dummies(project_logit_2[var], prefix=var)
    data2 = project_logit_2.join(cat_list_2)
    project_logit_2 = data2

data_vars_2 = project_logit_2.columns.values.tolist()
to_keep = [i for i in data_vars_2 if i not in cat_vars_2]
data_final_2 = project_logit_2[to_keep]

# Combine variables into a list of tuples

# NON_ROLLOVER = data_final['ROLL_NON-ROLLOVER'] # reference group for ROLL
ROLLOVER = data_final_2['ROLL_ROLLOVER']

AGE = data_final_2['AGE']

WEATHER_CLEAR_NORMAL = data_final_2['WEATHER_GROUP_CLEAR/NORMAL']
WEATHER_FOG_CLOUDY = data_final_2['WEATHER_GROUP_FOG/CLOUDY']
WEATHER_RAIN_SLEET = data_final_2['WEATHER_GROUP_RAIN/SLEET']
WEATHER_SNOW = data_final_2['WEATHER_GROUP_SNOW']
# WEATHER_WINDY = data_final_2['WEATHER_GROUP_WINDY'] # reference group for WEATHER_GROUP

# LIGHT_DARK = data_final_2['LIGHT_GROUP_DARK'] # reference group for LIGHT_GROUP
LIGHT_DAWN_DUSK = data_final_2['LIGHT_GROUP_DAWN/DUSK']
LIGHT_LIGHT = data_final_2['LIGHT_GROUP_LIGHT']

# SURFACE_DRY = data_final_2['SURFACE_GROUP_DRY'] # reference group for SURFACE_GROUP
SURFACE_WET_OIL = data_final_2['SURFACE_GROUP_WET/OIL/SLUSH/MUD']

# GRADE_GRADE = data_final_2['GRADE_GROUP_GRADE'] # reference group for GRADE_GROUP
GRADE_LEVEL = data_final_2['GRADE_GROUP_LEVEL']

# ALIGN_CURVE = data_final_2['ALIGN_GROUP_CURVE'] # reference group for ALIGN_GROUP
ALIGN_STRAIGHT = data_final_2['ALIGN_GROUP_STRAIGHT']

VEHICLE_CAR = data_final_2['VEHICLE_TYPE_CAR']
VEHICLE_PICKUP = data_final_2['VEHICLE_TYPE_PICKUP']
# VEHICLE_SUV_CUV = data_final_2['VEHICLE_TYPE_SUV/CUV'] # reference group for VEHICLE_TYPE
VEHICLE_VAN = data_final_2['VEHICLE_TYPE_VAN']

# MY_1989_2007 = data_final_2['MY_GROUP_1989-2007'] # # reference group for MY_GROUP
MY_2008_2019 = data_final_2['MY_GROUP_2008-2019']
MY_2011_2019 = data_final_2['MY_GROUP_2011-2019']


# Features
X_2 = list(zip(AGE, WEATHER_CLEAR_NORMAL, WEATHER_FOG_CLOUDY, WEATHER_RAIN_SLEET, WEATHER_SNOW, LIGHT_DAWN_DUSK, LIGHT_LIGHT, 
               SURFACE_WET_OIL, GRADE_LEVEL, ALIGN_STRAIGHT, VEHICLE_CAR, VEHICLE_PICKUP, VEHICLE_VAN, MY_2008_2019,
               MY_2011_2019))

# Target 
y_2 = list(ROLLOVER) 

from sklearn.model_selection import train_test_split
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, random_state = 8, test_size = 0.3)

log_reg = sm.Logit(y_train_2, X_train_2).fit()
print('X1: AGE')
print('X2: WEATHER_GROUP_CLEAR/NORMAL')
print('X3: WEATHER_GROUP_FOG/CLOUDY')
print('X4: WEATHER_GROUP_RAIN/SLEET')
print('X5: WEATHER_GROUP_SNOW')
print('X6: LIGHT_GROUP_DAWN/DUSK')
print('X7: LIGHT_GROUP_LIGHT')
print('X8: SURFACE_GROUP_WET/OIL/SLUSH/MUD')
print('X9: GRADE_GROUP_LEVEL')
print('X10: ALIGN_GROUP_STRAIGHT')
print('X11: VEHICLE_TYPE_CAR')
print('X12: VEHICLE_TYPE_PICKUP')
print('X13: VEHICLE_TYPE_VAN')
print('X14: MY_GROUP_2008-2019')
print('X15: MY_GROUP_2011-2019')
print(log_reg.summary())

# 10-fold cross validation:logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

scores = []

Logistic_Algorithm = LogisticRegression()
K_Fold = KFold(n_splits=10, random_state=8, shuffle=True)
scores = cross_val_score(Logistic_Algorithm, X_train_2, y_train_2, cv=K_Fold, n_jobs=-1)

print(scores)
