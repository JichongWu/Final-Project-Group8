import numpy as np
from scipy.stats import wilcoxon

# Two sets of scores from the logistic regression and KNN algorithm

Logistic_Algorithm_Score = np.array([0.67946645, 0.68330639, 0.67374166, 0.67333738, 0.67859309,
       0.68122094, 0.6739438 , 0.68607237, 0.69132808, 0.67596523])
KNN_Algorithm_Score = np.array([0.68573161, 0.68249798, 0.69658379, 0.67293309, 0.68829594,
       0.68041237, 0.69072165, 0.67050738, 0.66848595, 0.66949666])

# Wilcoxon signed-rank test
print('H0:Two sets of scores follows the same distribution\nH1:Two sets of scores do not follow the same distribution')
stat, p = wilcoxon(Logistic_Algorithm_Score, KNN_Algorithm_Score)
print('Test statistics: ', stat)
print('P-value: ', p)
# interpret
alpha = 0.05
if p > alpha:
    print('Not reject H0: Two sets of scores follows the same distribution')
else:
    print('Reject H0: Two sets of scores do not follow the same distribution')
