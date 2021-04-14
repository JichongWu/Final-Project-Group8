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
import plotly.express as px
import plotly.figure_factory as ff
import warnings

# Import ACCIDENT data sets from 2014 to 2018
acc_2014 = pd.read_csv('ACCIDENT_2014.CSV', encoding='ISO-8859-1')
acc_2015 = pd.read_csv('ACCIDENT_2015.CSV', encoding='ISO-8859-1')
acc_2016 = pd.read_csv('ACCIDENT_2016.CSV', encoding='ISO-8859-1')
acc_2017 = pd.read_csv('ACCIDENT_2017.CSV', encoding='ISO-8859-1')
acc_2018 = pd.read_csv('ACCIDENT_2018.CSV', encoding='ISO-8859-1')

# key variables to merge data:YEAR, ST_CASE
# VE_TOTAL:number of vehicles in a fatal crash; WEATHER:atmospheric Conditions in a fatal crash;
# LGT_COND:light Conditions in a fatal crash
acc_2014 = acc_2014[['YEAR', 'ST_CASE', 'VE_TOTAL', 'WEATHER', 'LGT_COND']]
acc_2015 = acc_2015[['YEAR', 'ST_CASE', 'VE_TOTAL', 'WEATHER', 'LGT_COND']]
acc_2016 = acc_2016[['YEAR', 'ST_CASE', 'VE_TOTAL', 'WEATHER', 'LGT_COND']]
acc_2017 = acc_2017[['YEAR', 'ST_CASE', 'VE_TOTAL', 'WEATHER', 'LGT_COND']]
acc_2018 = acc_2018[['YEAR', 'ST_CASE', 'VE_TOTAL', 'WEATHER', 'LGT_COND']]

# Combine ACCIDENT data sets 2014-2018
acc = pd.concat([acc_2014, acc_2015, acc_2016, acc_2017, acc_2018],axis=0)

# Import VEHICLE data sets from 2014 to 2018
veh_2014 = pd.read_csv('VEHICLE_2014.CSV', encoding='ISO-8859-1')
veh_2015 = pd.read_csv('VEHICLE_2015.CSV', encoding='ISO-8859-1')
veh_2016 = pd.read_csv('VEHICLE_2016.CSV', encoding='ISO-8859-1')
veh_2017 = pd.read_csv('VEHICLE_2017.CSV', encoding='ISO-8859-1')
veh_2018 = pd.read_csv('VEHICLE_2018.CSV', encoding='ISO-8859-1')

# key variables to merge data:YEAR, ST_CASE, VEH_NO
# VSURCOND:roadway surface condition; VPROFILE:roadway Grade; VALIGN:roadway alignment; BODY_TYP:vehicle type;
# MOD_YEAR:vehicle model year; ROLLOVER:status of rollover;
veh_2014 = veh_2014[['ST_CASE', 'VEH_NO', 'VSURCOND', 'VPROFILE', 'VALIGN', 'BODY_TYP', 'MOD_YEAR', 'ROLLOVER']]
veh_2014['YEAR'] = 2014 # Create a new column 'YEAR'; YEAR=2014 for all rows in veh_2014
veh_2015 = veh_2015[['ST_CASE', 'VEH_NO', 'VSURCOND', 'VPROFILE', 'VALIGN', 'BODY_TYP', 'MOD_YEAR', 'ROLLOVER']]
veh_2015['YEAR'] = 2015 # Create a new column 'YEAR'; YEAR=2015 for all rows in veh_2015
veh_2016 = veh_2016[['ST_CASE', 'VEH_NO', 'VSURCOND', 'VPROFILE', 'VALIGN', 'BODY_TYP', 'MOD_YEAR', 'ROLLOVER']]
veh_2016['YEAR'] = 2016 # Create a new column 'YEAR'; YEAR=2016 for all rows in veh_2016
veh_2017 = veh_2017[['ST_CASE', 'VEH_NO', 'VSURCOND', 'VPROFILE', 'VALIGN', 'BODY_TYP', 'MOD_YEAR', 'ROLLOVER']]
veh_2017['YEAR'] = 2017 # Create a new column 'YEAR'; YEAR=2017 for all rows in veh_2017
veh_2018 = veh_2018[['ST_CASE', 'VEH_NO', 'VSURCOND', 'VPROFILE', 'VALIGN', 'BODY_TYP', 'MOD_YEAR', 'ROLLOVER']]
veh_2018['YEAR'] = 2018 # Create a new column 'YEAR'; YEAR=2018 for all rows in veh_2018

# Combine VEHICLE data sets 2014-2018
veh = pd.concat([veh_2014, veh_2015, veh_2016, veh_2017, veh_2018],axis=0)

# Import PERSON data sets from 2014 to 2018
per_2014 = pd.read_csv('PERSON_2014.CSV', encoding='ISO-8859-1')
per_2015 = pd.read_csv('PERSON_2015.CSV', encoding='ISO-8859-1')
per_2016 = pd.read_csv('PERSON_2016.CSV', encoding='ISO-8859-1')
per_2017 = pd.read_csv('PERSON_2017.CSV', encoding='ISO-8859-1')
per_2018 = pd.read_csv('PERSON_2018.CSV', encoding='ISO-8859-1')

# key variables to merge data:YEAR, ST_CASE, VEH_NO
# SEAT_POS:an occupant's seat position; SEX:an occupant's gender; AGE:an occupant's age;
per_2014 = per_2014[['ST_CASE', 'VEH_NO', 'SEAT_POS', 'SEX', 'AGE']]
per_2014['YEAR'] = 2014 # Create a new column 'YEAR'; YEAR=2014 for all rows in per_2014
per_2015 = per_2015[['ST_CASE', 'VEH_NO', 'SEAT_POS', 'SEX', 'AGE']]
per_2015['YEAR'] = 2015 # Create a new column 'YEAR'; YEAR=2015 for all rows in per_2015
per_2016 = per_2016[['ST_CASE', 'VEH_NO', 'SEAT_POS', 'SEX', 'AGE']]
per_2016['YEAR'] = 2016 # Create a new column 'YEAR'; YEAR=2016 for all rows in per_2016
per_2017 = per_2017[['ST_CASE', 'VEH_NO', 'SEAT_POS', 'SEX', 'AGE']]
per_2017['YEAR'] = 2017 # Create a new column 'YEAR'; YEAR=2017 for all rows in per_2017
per_2018 = per_2018[['ST_CASE', 'VEH_NO', 'SEAT_POS', 'SEX', 'AGE']]
per_2018['YEAR'] = 2018 # Create a new column 'YEAR'; YEAR=2018 for all rows in per_2018

# Combine PERSON data sets 2014-2018
per = pd.concat([per_2014, per_2015, per_2016, per_2017, per_2018],axis=0)

# Merge ACCIDENT 2014-2018 and VEHICLE 2014-2018 
# The key variables are YEAR and ST_CASE
acc.sort_values(by=['YEAR', 'ST_CASE']) # Sort the ACCIDENT 2014-2018 by the key variables first
veh.sort_values(by=['YEAR', 'ST_CASE']) # Sort the VEHICLE 2014-2018 by the key variables first
acc_veh = pd.merge(acc, veh, on = ['YEAR', 'ST_CASE'], how='inner')

# Merge ACCIDENT-VEHICLE 2014-2018 and PERSON 2014-2018 
# The the key variables YEAR, ST_CASE, VEH_NO
acc_veh.sort_values(by=['YEAR', 'ST_CASE', 'VEH_NO']) # Sort the ACCIDENT-VEHICLE 2014-2018 by the key variables first
per.sort_values(by=['YEAR', 'ST_CASE', 'VEH_NO']) # Sort the PERSON 2014-2018 by the key variables first
acc_veh_per = pd.merge(acc_veh, per, on = ['YEAR', 'ST_CASE', 'VEH_NO'], how='inner')

# Copy ACCIDENT-VEHICLE-PERSON 2014-2018
acc_veh_per_copy = acc_veh_per.copy()

# Filter ACCIDENT-VEHICLE-PERSON 2014-2018
# VE_TOTAL==1: single vehicle crash
# SEAT_POS==11: driver's position
# MOD_YEAR >= 1989: only consider vehicles with model years between 1989 and 2019
acc_veh_per_copy=acc_veh_per_copy[(acc_veh_per_copy.VE_TOTAL==1) & (acc_veh_per_copy.SEAT_POS==11) & 
                                  (acc_veh_per_copy.MOD_YEAR >= 1989)]

# Group weather condition
weather_map = {1:'CLEAR/NORMAL', 2:'RAIN/SLEET', 3:'RAIN/SLEET', 4:'SNOW', 5:'FOG/CLOUDY', 6:'WINDY', 7:'WINDY', 
               8:'NA', 10:'FOG/CLOUDY', 11:'SNOW', 12:'RAIN/SLEET', 98:'NA', 99:'NA'}
acc_veh_per_copy['WEATHER_GROUP'] = acc_veh_per_copy['WEATHER'].map(weather_map)

# Group light condition
light_map = {1:'LIGHT', 2:'DARK', 3:'LIGHT', 4:'DAWN/DUSK', 5:'DAWN/DUSK', 6:'NA', 7:'NA', 8:'NA', 9:'NA'}
acc_veh_per_copy['LIGHT_GROUP'] = acc_veh_per_copy['LGT_COND'].map(light_map)

# Group roadway surface condition
vsurcond_map = {0:'NA', 1:'DRY', 2:'WET', 3:'WET', 4:'WET', 5:'DRY', 6:'WET', 7:'OIL/SLUSH/MUD', 8:'NA', 10:'OIL/SLUSH/MUD', 
                11:'OIL/SLUSH/MUD', 98:'NA', 99:'NA'}
acc_veh_per_copy['SURFACE_GROUP'] = acc_veh_per_copy['VSURCOND'].map(vsurcond_map)

# Group roadway grade
vprofile_map = {0:'NA', 1:'LEVEL', 2:'GRADE', 3:'GRADE', 4:'GRADE', 5:'GRADE', 6:'GRADE', 8:'NA', 9:'NA'}
acc_veh_per_copy['GRADE_GROUP'] = acc_veh_per_copy['VPROFILE'].map(vprofile_map)

# Group roadway alignment
valign_map = {0:'NA', 1:'STRAIGHT', 2:'CURVE', 3:'CURVE', 4:'CURVE', 8:'NA', 9:'NA'}
acc_veh_per_copy['ALIGN_GROUP'] = acc_veh_per_copy['VALIGN'].map(valign_map)

# Group vehicle type
vehicle_type_map = {1:'CAR', 2:'CAR', 3:'CAR', 4:'CAR', 5:'CAR', 6:'CAR', 7:'CAR', 8:'CAR', 9:'CAR', 10:'NA', 12:'CAR', 
                    14:'SUV/CUV', 15:'SUV/CUV', 16:'SUV/CUV', 17:'CAR', 19:'NA', 20:'VAN', 21:'VAN', 22:'VAN', 28:'VAN', 
                    29:'VAN', 30:'PICKUP', 31:'PICKUP', 32:'PICKUP', 33:'PICKUP', 34:'PICKUP', 39:'PICKUP', 40:'PICKUP', 
                    42:'NA', 45:'NA', 48:'NA', 49:'NA', 50:'NA', 51:'NA', 52:'NA', 55:'NA', 58:'NA', 59:'NA', 60:'NA', 
                    61:'NA', 62:'NA', 63:'NA', 64:'NA', 65:'NA', 66:'NA', 67:'NA', 71:'NA', 72:'NA', 73:'NA', 78:'NA', 
                    79:'NA', 80:'NA', 81:'NA', 82:'NA', 83:'NA', 84:'NA', 85:'NA', 86:'NA', 87:'NA', 88:'NA', 89:'NA', 
                    90:'NA', 91:'NA', 92:'NA', 93:'NA', 94:'NA', 95:'NA', 96:'NA', 97:'NA', 98:'NA', 99:'NA'}
acc_veh_per_copy['VEHICLE_TYPE'] = acc_veh_per_copy['BODY_TYP'].map(vehicle_type_map)

# Group model year
condition1 = [
    (acc_veh_per_copy['MOD_YEAR'] >= 1989) & (acc_veh_per_copy['MOD_YEAR'] <= 2007),
    (acc_veh_per_copy['MOD_YEAR'] >= 2008) & (acc_veh_per_copy['MOD_YEAR'] <= 2010),
    (acc_veh_per_copy['MOD_YEAR'] >= 2011) & (acc_veh_per_copy['MOD_YEAR'] <= 2019),
    (acc_veh_per_copy['MOD_YEAR'] == 9998),
    (acc_veh_per_copy['MOD_YEAR'] == 9999)
    ]
model_year_value = ['1989-2007','2008-2010', '2011-2019', 'NA', 'NA']
acc_veh_per_copy['MY_GROUP'] = np.select(condition1, model_year_value)

# Group rollover
rollover_map = {0:'NON-ROLLOVER', 1:'ROLLOVER', 2:'ROLLOVER', 9:'ROLLOVER'}
acc_veh_per_copy['ROLL'] = acc_veh_per_copy['ROLLOVER'].map(rollover_map)

# Driver's gender
gender_map = {1:'MALE', 2:'FEMALE', 8:'NA', 9:'NA'}
acc_veh_per_copy['GENDER'] = acc_veh_per_copy['SEX'].map(gender_map)

# Group driver's age
condition2 = [
    (acc_veh_per_copy['AGE'] >= 16) & (acc_veh_per_copy['AGE'] <= 20),
    (acc_veh_per_copy['AGE'] >= 21) & (acc_veh_per_copy['AGE'] <= 24),
    (acc_veh_per_copy['AGE'] >= 25) & (acc_veh_per_copy['AGE'] <= 34),
    (acc_veh_per_copy['AGE'] >= 35) & (acc_veh_per_copy['AGE'] <= 44),
    (acc_veh_per_copy['AGE'] >= 45) & (acc_veh_per_copy['AGE'] <= 54),
    (acc_veh_per_copy['AGE'] >= 55) & (acc_veh_per_copy['AGE'] <= 64),
    (acc_veh_per_copy['AGE'] >= 65) & (acc_veh_per_copy['AGE'] <= 74),
    (acc_veh_per_copy['AGE'] >= 75) & (acc_veh_per_copy['AGE'] <= 103),
    (acc_veh_per_copy['AGE'] >= 1) & (acc_veh_per_copy['AGE'] <= 15),
    (acc_veh_per_copy['AGE'] == 998),
    (acc_veh_per_copy['AGE'] == 999)
    ]
age_group_value = ['16-20','21-24', '25-34', '35-44', '45-54', '55-64', '65-74', '> 74','NA','NA','NA']
acc_veh_per_copy['AGE_GROUP'] = np.select(condition2, age_group_value)

project=acc_veh_per_copy[(acc_veh_per_copy.WEATHER_GROUP != 'NA') & 
                         (acc_veh_per_copy.LIGHT_GROUP != 'NA') &
                         (acc_veh_per_copy.SURFACE_GROUP != 'NA') &
                         (acc_veh_per_copy.GRADE_GROUP != 'NA') &
                         (acc_veh_per_copy.ALIGN_GROUP != 'NA') &
                         (acc_veh_per_copy.VEHICLE_TYPE != 'NA') &
                         (acc_veh_per_copy.MY_GROUP != 'NA') &
                         (acc_veh_per_copy.GENDER != 'NA') &
                         (acc_veh_per_copy.AGE_GROUP != 'NA')
                        ]
project.shape
'''
project data set contains 61841 rows and 25 variables
'''
