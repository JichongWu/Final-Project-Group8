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

# Chi-squar test: weather conditions and rollovers
cont_weather_roll= pd.crosstab(project['WEATHER_GROUP'], project['ROLL'])

# Create a heatmap for weather conditions and rollovers
plt.figure(figsize=(12,8))
sns.heatmap(cont_weather_roll, annot=True, cmap="YlGnBu")
plt.show()

# Use Chi-square test to examine the association between the weather conditions and rollovers
print('H0: The weather conditions and rollovers are independent\nH1: The weather conditions and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_weather_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The weather conditions and rollovers are not independent, since the p-value of the ch-square test is less than 0.05')

#############################################################################################################################

# Chi-squar test: light conditions and rollovers
cont_light_roll= pd.crosstab(project['LIGHT_GROUP'], project['ROLL'])

# Create a heatmap for light conditions and rollovers
plt.figure(figsize=(12,8))
sns.heatmap(cont_light_roll, annot=True, cmap="YlGnBu")

# Use Chi-square test to examine the association between the light conditions and rollovers
print('H0: The light conditions and rollovers are independent\nH1: The light conditions and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_light_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The light conditions and rollovers are not independent, since the p-value of the ch-square test is less than 0.05')

#############################################################################################################################

# Chi-squar test: roadway surface conditions and rollovers
cont_surface_roll= pd.crosstab(project['SURFACE_GROUP'], project['ROLL'])

# Create a heatmap for roadway surface conditions and rollovers
plt.figure(figsize=(12,8))
sns.heatmap(cont_surface_roll, annot=True, cmap="YlGnBu")

# Use Chi-square test to examine the association between the roadway surface conditions and rollovers
print('H0: The roadway surface conditions and rollovers are independent\nH1: The roadway surface conditions and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_surface_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The roadway surface conditions and rollovers are not independent, since the p-value of the ch-square test is less than 0.05')

#############################################################################################################################

# Chi-squar test: roadway grade and rollovers
cont_grade_roll= pd.crosstab(project['GRADE_GROUP'], project['ROLL'])

# Create a heatmap for roadway grade and rollovers
plt.figure(figsize=(12,8))
sns.heatmap(cont_grade_roll, annot=True, cmap="YlGnBu")

# Use Chi-square test to examine the association between the roadway grade and rollovers
print('H0: The roadway grade and rollovers are independent\nH1: The roadway grade and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_grade_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The roadway grade and rollovers are not independent, since the p-value of the ch-square test is less than 0.05')

#############################################################################################################################

# Chi-squar test: roadway alignment and rollovers
cont_align_roll= pd.crosstab(project['ALIGN_GROUP'], project['ROLL'])

# Create a heatmap for roadway alignment and rollovers
plt.figure(figsize=(12,8))
sns.heatmap(cont_align_roll, annot=True, cmap="YlGnBu")

# Use Chi-square test to examine the association between the roadway alignment and rollovers
print('H0: The roadway alignment and rollovers are independent\nH1: The roadway alignment and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_align_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The roadway alignment and rollovers are not independent, since the p-value of the ch-square test is less than 0.05')

#############################################################################################################################

# Chi-squar test: vehicle types and rollovers
cont_veh_roll= pd.crosstab(project['VEHICLE_TYPE'], project['ROLL'])

# Create a heatmap for vehicle types and rollovers
plt.figure(figsize=(12,8))
sns.heatmap(cont_veh_roll, annot=True, cmap="YlGnBu")

# Use Chi-square test to examine the association between the vehicle types and rollovers
print('H0: The vehicle types and rollovers are independent\nH1: The vehicle types and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_veh_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The vehicle types and rollovers are not independent, since the p-value of the ch-square test is less than 0.05')

#############################################################################################################################

# Chi-squar test: vehicle model years and rollovers
cont_veh_roll= pd.crosstab(project['MY_GROUP'], project['ROLL'])

# Create a heatmap for vehicle model years and rollovers
plt.figure(figsize=(12,8))
sns.heatmap(cont_veh_roll, annot=True, cmap="YlGnBu")

# Use Chi-square test to examine the association between the vehicle model years and rollovers
print('H0: The vehicle model years and rollovers are independent\nH1: The vehicle model years and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_veh_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The vehicle model years and rollovers are not independent, since the p-value of the ch-square test is less than 0.05')

#############################################################################################################################

# Chi-squar test: driver's gender and rollovers
cont_gender_roll= pd.crosstab(project['GENDER_GROUP'], project['ROLL'])

# Create a heatmap for driver's gender and rollovers
plt.figure(figsize=(12,8))
sns.heatmap(cont_gender_roll, annot=True, cmap="YlGnBu")

# Use Chi-square test to examine the association between the driver's gender and rollovers
print("H0: The driver's gender and rollovers are independent\nH1: The driver's gender and rollovers are not independent")
c, p, dof, expected = chi2_contingency(cont_gender_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print("The driver's gender and rollovers are not independent, since the p-value of the ch-square test is less than 0.05")

#############################################################################################################################

# Chi-squar test: driver's age and rollovers
cont_age_roll= pd.crosstab(project['AGE_GROUP'], project['ROLL'])

# Create a heatmap for driver's age and rollovers
plt.figure(figsize=(12,8))
sns.heatmap(cont_age_roll, annot=True, cmap="YlGnBu")
plt.show()

# Use Chi-square test to examine the association between the driver's age and rollovers
print("H0: The driver's age and rollovers are independent\nH1: The driver's age and rollovers are not independent")
c, p, dof, expected = chi2_contingency(cont_age_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print("The driver's age and rollovers are not independent, since the p-value of the ch-square test is less than 0.05")
