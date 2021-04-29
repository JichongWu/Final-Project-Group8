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

# Pre-analysis
# Rollover frequency 2014-2018
# Bar plot: rollover status
sns.countplot(x='ROLL', data=project, palette='hls')
plt.show()
count_rollover = len(project[project['ROLL']=='ROLLOVER'])
count_nonrollover = len(project[project['ROLL']=='NON-ROLLOVER'])

# Rate of rollover
pct_of_rollover = count_rollover/(count_rollover+count_nonrollover)
print('Percentage of rollover', pct_of_rollover*100)
pct_of_nonrollover = count_nonrollover/(count_rollover+count_nonrollover)
print('Percentage of rollover', pct_of_nonrollover*100)

# Pre-analysis
# Rollover frequency by year
# Bar plot: Rollover status by year
pd.crosstab(project.YEAR, project.ROLL).plot(kind='bar')
plt.title('Rollover Status by Year')
plt.xlabel('Year')
plt.ylabel('Frequency of Rollover Status')

# Bar plot: Rate of rollover by weather
table = pd.crosstab(project.YEAR,project.ROLL)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Rate of Rollover by Year')
plt.xlabel('Year')
plt.ylabel('Rollover Status')

# Chi-sqrt Analysis
# Rollover vs. Weather
# Bar plot: Rollover status by weather
pd.crosstab(project.WEATHER_GROUP, project.ROLL).plot(kind='bar')
plt.title('Rollover Status by Weather')
plt.xlabel('Weather')
plt.ylabel('Frequency of Rollover Status')

# Bar plot: Rate of rollover by weather
table = pd.crosstab(project.WEATHER_GROUP,project.ROLL)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Rate of Rollover by Weather')
plt.xlabel('Weather')
plt.ylabel('Rollover Status')

# Chi-sqrt test: Association between rollover and weather
cont_weather_roll= pd.crosstab(project['WEATHER_GROUP'], project['ROLL'])
print('H0: The weather conditions and rollovers are independent\nH1: The weather conditions and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_weather_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The weather conditions and rollovers are not independent, since the p-value of the chi-square test is less than 0.05')

# Chi-sqrt Analysis
# Rollover vs. Light Condition
# Bar plot: Rollover status by light condition
pd.crosstab(project.LIGHT_GROUP, project.ROLL).plot(kind='bar')
plt.title('Rollover Status by Light Condition')
plt.xlabel('Light Condition')
plt.ylabel('Frequency of Rollover Status')

# Bar plot: Rate of rollover by weather
table = pd.crosstab(project.LIGHT_GROUP, project.ROLL)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Rate of Rollover by Light Condition')
plt.xlabel('Light Condition')
plt.ylabel('Rollover Status')

# Chi-sqrt test: Association between rollover and light condition
cont_light_roll= pd.crosstab(project['LIGHT_GROUP'], project['ROLL'])
print('H0: The light conditions and rollovers are independent\nH1: The light conditions and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_light_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The light conditions and rollovers are not independent, since the p-value of the chi-square test is less than 0.05')

# Chi-sqrt Analysis
# Rollover vs. Roadway Surface Condition
# Bar plot: Rollover status by roadway surface condition
pd.crosstab(project.SURFACE_GROUP, project.ROLL).plot(kind='bar')
plt.title('Rollover Status by Roadway Surface Condition')
plt.xlabel('Roadway Surface Condition')
plt.ylabel('Frequency of Rollover Status')

# Bar plot: Rate of rollover by roadway surface condition
table = pd.crosstab(project.SURFACE_GROUP, project.ROLL)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Rate of Rollover by Roadway Surface Condition')
plt.xlabel('Roadway Surface Condition')
plt.ylabel('Rollover Status')

# Chi-sqrt test: Association between rollover and roadway surface condition
cont_surface_roll= pd.crosstab(project['SURFACE_GROUP'], project['ROLL'])
print('H0: The roadway surface conditions and rollovers are independent\nH1: The roadway surface conditions and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_surface_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The roadway surface conditions and rollovers are not independent, since the p-value of the chi-square test is less than 0.05')

# Chi-sqrt Analysis
# Rollover vs. Roadway Grade
# Bar plot: Rollover status by roadway grade
pd.crosstab(project.GRADE_GROUP, project.ROLL).plot(kind='bar')
plt.title('Rollover Status by Roadway Grade')
plt.xlabel('Roadway Grade')
plt.ylabel('Frequency of Rollover Status')

# Bar plot: Rate of rollover by roadway grade
table = pd.crosstab(project.GRADE_GROUP, project.ROLL)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Rate of Rollover by Roadway Grade')
plt.xlabel('Roadway Grade')
plt.ylabel('Rollover Status')

# Chi-squar test: Association between rollover and roadway grade
cont_grade_roll= pd.crosstab(project['GRADE_GROUP'], project['ROLL'])
print('H0: The roadway grade and rollovers are independent\nH1: The roadway grade and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_grade_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The roadway grade and rollovers are not independent, since the p-value of the chi-square test is less than 0.05')

# Chi-sqrt Analysis
# Rollover vs. Roadway Alignment
# Bar plot: Rollover status by roadway alignment
pd.crosstab(project.ALIGN_GROUP, project.ROLL).plot(kind='bar')
plt.title('Rollover Status by Roadway Alignment')
plt.xlabel('Roadway Alignment')
plt.ylabel('Frequency of Rollover Status')

# Bar plot: Rate of rollover by roadway alignment
table = pd.crosstab(project.ALIGN_GROUP, project.ROLL)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Rate of Rollover by Roadway Alignment')
plt.xlabel('Roadway Alignment')
plt.ylabel('Rollover Status')

# Chi-squar test: Association between rollover and roadway alignment
cont_align_roll= pd.crosstab(project['ALIGN_GROUP'], project['ROLL'])
print('H0: The roadway alignment and rollovers are independent\nH1: The roadway alignment and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_align_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The roadway alignment and rollovers are not independent, since the p-value of the ch-square test is less than 0.05')

# Chi-sqrt Analysis
# Rollover vs. vehicle type
# Bar plot: Rollover status by vehicle type
pd.crosstab(project.LIGHT_GROUP, project.ROLL).plot(kind='bar')
plt.title('Rollover Status by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Frequency of Rollover Status')

# Bar plot: Rate of rollover by vehicle type
table = pd.crosstab(project.VEHICLE_TYPE, project.ROLL)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Rate of Rollover by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Rollover Status')

# Chi-sqrt test: Association between rollover and vehicle type
cont_light_roll= pd.crosstab(project['VEHICLE_TYPE'], project['ROLL'])
print('H0: The vehicle types and rollovers are independent\nH1: The vehicle types and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_light_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The vehicle type and rollovers are not independent, since the p-value of the chi-square test is less than 0.05')

# Chi-sqrt Analysis
# Rollover vs. model years
# Bar plot: Rollover status by model years
pd.crosstab(project.MY_GROUP, project.ROLL).plot(kind='bar')
plt.title('Rollover Status by Model Years')
plt.xlabel('Model Years')
plt.ylabel('Frequency of Rollover Status')

# Bar plot: Rate of rollover by model years
table = pd.crosstab(project.MY_GROUP, project.ROLL)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Rate of Rollover by Model Years')
plt.xlabel('Model Years')
plt.ylabel('Rollover Status')

# Chi-sqrt test: Association between rollover and model years
cont_light_roll= pd.crosstab(project['MY_GROUP'], project['ROLL'])
print('H0: The model years and rollovers are independent\nH1: The model years and rollovers are not independent')
c, p, dof, expected = chi2_contingency(cont_light_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print('The model years and rollovers are not independent, since the p-value of the chi-square test is less than 0.05')

# Chi-sqrt Analysis
# Rollover vs. driver's gender
# Bar plot: Rollover status by driver's gender
pd.crosstab(project.GENDER, project.ROLL).plot(kind='bar')
plt.title("Rollover Status by Driver's Gender")
plt.xlabel("Driver's Gender")
plt.ylabel('Frequency of Rollover Status')

# Bar plot: Rate of rollover by driver's gender
table = pd.crosstab(project.GENDER, project.ROLL)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title("Rate of Rollover by Driver's Gender")
plt.xlabel("Driver's Gender")
plt.ylabel('Rollover Status')

# Chi-sqrt test: Association between rollover and driver's gender
cont_light_roll= pd.crosstab(project['GENDER'], project['ROLL'])
print("H0: The driver's gender and rollovers are independent\nH1: The driver's gender and rollovers are not independent")
c, p, dof, expected = chi2_contingency(cont_light_roll)
print('Chi-square statistic:', c)
print('P-value of the Chi-square statistic:', p)
print("The driver's gender and rollovers are not independent, since the p-value of the chi-square test is less than 0.05")

# Driver's age vs. Rollover
# Box plot: Driver's age by rollover status

fig, ax = plt.subplots(figsize=(10,8))
plt.suptitle("Driver's Age by Rollover Status")
project.boxplot(column='AGE',by='ROLL', ax=ax)

# ANOVA: test if the driver's age in rollover and non-rollover events are different
import statsmodels.api as sm
from statsmodels.formula.api import ols

print("H0: The driver's age in rollover events is equal to the driver's age in non-rollover events")
print("H1: The driver's age in rollover events is not equal to the driver's age in non-rollover events")
mod = ols('AGE ~ ROLL', data=project).fit()                
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
print("The driver's age in rollover and non-rollover events are different, since the p-value of the F test is less than 0.05")
