#What are roll-over rates, are there 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
url = 'https://github.com/JichongWu/Final-Project-Group8/blob/main/Data/project.csv?raw=true'
df = pd.read_csv(url,index_col=0)
#df = pd.read_csv(url)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
print(df.head(5))

year_rollover = df.groupby(['CY']).agg({'ROLLOVER': 'sum'})
year_rollover["%"] = year_rollover.groupby(level=0).apply(lambda x:  100*x / len(x))
print("The total number of rollovers per yr, percent\n", year_rollover)

gr = df.groupby(['CY', 'ROLL']).size()
print ("ROLL vs. Non-roll per yr", gr)

print("Total number of collisions per yr", gr.groupby(level=0).sum())

data1 = [['2014', 7785, 4656],
         ['2015', 8194, 4788],
         ['2016', 8400, 4641],
         ['2017', 8164, 4304],
         ['2018', 8059, 3861]]

rollover = pd.DataFrame(data1, columns =['Year', 'NON-ROLLOVER', 'ROLLOVER'])
rollover['Total'] = rollover.sum(axis=1)
rollover['Percent ROLL'] = (rollover['ROLLOVER']/rollover['Total'])*100
rollover['Percent NON-ROLL'] = (rollover['NON-ROLLOVER']/rollover['Total'])*100
rollover1 = rollover.set_index('Year')


#Prepare data for graph
x = np.array(rollover['Year'])
y1 = np.array(rollover['Percent ROLL'])
y2 = np.array(rollover['Percent NON-ROLL'])

plt.bar(x, y1, label='ROLLOVER', color = 'blue')
plt.bar(x, y2, bottom = y1, label='NON-ROLLOVER', color = 'red')
plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.legend(["ROLLOVER", "NON-ROLLVER"])
plt.show()

#Stat Analysis
list(rollover1)


from bioinfokit.analys import stat, get_data
!pip install matplotlib-venn
!pip install adjustText
!pip install tabulate
!pip install textwrap3

data2 = pd.DataFrame(data1, columns =['Year', 'NON-ROLLOVER', 'ROLLOVER'])
data2 = (data2.set_index('Year'))
print(data2.head(10))

#This will tell you p value, degrees of freedom
res = stat()
res.chisq(df=data2)
print(res.summary)


print(res.expected_df)

import numpy as np
from scipy.stats import chi2_contingency

observed = np.array([[7785, 4656],
                     [8194, 4788],
                     [8400,4641],
                     [8164,4304],
                     [8059,3861]])
chi_val, p_val, dof, expected =  chi2_contingency(observed, correction=False)
chi_val, p_val, dof, expected =  chi2_contingency(observed, lambda_="log-likelihood")

chi_val, p_val, dof, expected =  chi2_contingency(observed, lambda_="log-likelihood")
print(chi_val, p_val, dof, expected)
