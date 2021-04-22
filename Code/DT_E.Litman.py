###DECISION TREE####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import tree
import seaborn as sns


url = 'https://github.com/JichongWu/Final-Project-Group8/blob/main/Data/project_04142021_J.Pai.csv?raw=true'
project = pd.read_csv(url, encoding='ISO-8859-1')
print(project.head(5))
list(project.columns.values)


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
column = 'AGE'
age_normal = (project[column] - project[column].min()) / (project[column].max() - project[column].min())

X=list(zip(weather_encoded, light_encoded, surface_encoded, grade_encoded, vehicle_encoded, align_encoded, model_year_encoded,
          gender_encoded,age_normal))
print(X)

le = preprocessing.LabelEncoder()
y = le.fit_transform(project['ROLL'])
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


print(y_test)
print(X_test)

#%%-----------------------------------------------------------------------
# perform training with entropy.
# Decision tree with entropy
clf_entropy = tree.DecisionTreeClassifier(criterion="entropy", random_state=100, splitter = "best", max_depth=3, min_samples_leaf=5)

# Performing training
clf_entropy.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
# make predictions

# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(X_test)
print(y_pred_entropy)
#%%----------------------------------------------------------------------

#%%-----------------------------------------------------------------------

# confusion matrix for entropy model
conf_matrix = confusion_matrix(y_test, y_pred_entropy)
print(conf_matrix)
conf_matrix.shape

class_names = np.unique(y_test)
print(class_names)
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
print(df_cm)
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------
# display decision tree
class_names = ['0','1']
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

dot_data = export_graphviz(clf_entropy, filled=True, rounded=True, class_names=class_names, feature_names= (weather_encoded, light_encoded, surface_encoded, grade_encoded, vehicle_encoded, align_encoded, model_year_encoded,
          gender_encoded,age_normal), out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_entropy_project.pdf")
webbrowser.open_new(r'decision_tree_entropy_project.pdf')

print ('-'*40 + 'End Console' + '-'*40 + '\n')

print("Accuracy with 3 levels:",metrics.accuracy_score(y_test, y_pred))

#######
