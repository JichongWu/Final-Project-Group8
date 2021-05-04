import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# List of tuples
score = [ (0.67498267, 0.67683068, 0.67914068, 0.67867868, 0.68630169,0.67914068, 0.68237468, 0.68075768, 0.6732902 , 
              0.69731978) ,
             (0.67844768, 0.67382767, 0.68260568, 0.68052668, 0.68884269,0.67382767, 0.67659968, 0.68006468, 0.67167283, 
              0.69061922) ,
             ]
# Create DataFrame object from a list of tuples
df = pd.DataFrame(score, columns = ['1st-Fold' , '2nd-Fold', '3rd-Fold' , '4th-Fold', '5th-Fold', '6th-Fold', '7th-Fold', 
                                      '8th-Fold', '9th-Fold', '10th-Fold'], 
                     index=['Logistic_Regression', 'KNN'])
print(df)

# Two sets of scores from the logistic regression and KNN algorithm

Logistic_Regression = np.array([0.67498267, 0.67683068, 0.67914068, 0.67867868, 0.68630169,0.67914068, 0.68237468, 
                                0.68075768, 0.6732902 , 0.69731978])
KNN = np.array([0.67844768, 0.67382767, 0.68260568, 0.68052668, 0.68884269,0.67382767, 0.67659968, 0.68006468, 0.67167283, 
                0.69061922])

# Wilcoxon signed-rank test
print('H0:Two sets of scores follows the same distribution\nH1:Two sets of scores do not follow the same distribution')
stat, p = wilcoxon(Logistic_Regression, KNN)
print('Test statistics: ', stat)
print('P-value: ', p)
# interpret
alpha = 0.05
if p > alpha:
    print('Not reject H0: Two sets of scores follows the same distribution')
else:
    print('Reject H0: Two sets of scores do not follow the same distribution')
