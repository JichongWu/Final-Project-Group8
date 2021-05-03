##################################################
##################################################
##################################################
### Created by Jichong Wu
### Project Name : Vehicle Rollover Prediction in a Fatal Car Accident
### Date 05/02/2021
##################################################
##################################################
##################################################


from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
#from PyQt5.QtGui import QPixmap, QIcon   # image
import urllib.request       # image
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt  # Control status
# Figure packages
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure     # Figure

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve, auc
from scipy.stats import chi2_contingency
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest

# 10-fold cross validation:logistic regression
from sklearn.model_selection import cross_val_score

# classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# draw tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

#=========================================================
#=========================================================
# Global Parameters
## load master dataset
project = pd.read_csv('https://raw.githubusercontent.com/JichongWu/Final-Project-Group8/main/Data/project%20dataset%20FINAL.csv',sep=',')
#feature_list = ['WEATHER_GROUP','LIGHT_Condition','ROAD_SURFACE','ROADWAY_GRADE','VEHICLE_TYPE','ROADWAY_ALIGNMENT', 'VEHICLE_YEAR','GENDER','AGE']
#=========================================================
#=========================================================
# Decision Tree code
le = LabelEncoder()
weather_encoded = le.fit_transform(project['WEATHER_GROUP'])
light_encoded = le.fit_transform(project['LIGHT_Condition'])
surface_encoded = le.fit_transform(project['ROAD_SURFACE'])
grade_encoded = le.fit_transform(project['ROADWAY_GRADE'])
vehicle_encoded = le.fit_transform(project['VEHICLE_TYPE'])
align_encoded = le.fit_transform(project['ROADWAY_ALIGNMENT'])
model_year_encoded = le.fit_transform(project['VEHICLE_YEAR'])
gender_encoded = le.fit_transform(project['GENDER'])
# Standardize AGE column
column = 'AGE'
age_normal = (project[column] - project[column].min()) / (project[column].max() - project[column].min())
# create a new dataset
ziplist = list(zip(weather_encoded, light_encoded, surface_encoded, grade_encoded, vehicle_encoded, align_encoded,
        model_year_encoded, gender_encoded, age_normal))

df_tree = pd.DataFrame(ziplist, columns=['WEATHER_GROUP','LIGHT_Condition','ROAD_SURFACE','ROADWAY_GRADE','VEHICLE_TYPE',
                                         'ROADWAY_ALIGNMENT', 'VEHICLE_YEAR','GENDER','AGE'])
#=======================================
#=======================================
# Random Forest & Logistic Regression code

## data_final 1
project_logit = project.copy()
cat_vars=['ROLL','WEATHER_GROUP','LIGHT_Condition','ROAD_SURFACE','ROADWAY_GRADE','ROADWAY_ALIGNMENT','VEHICLE_TYPE','VEHICLE_YEAR','GENDER']

for var in cat_vars:
    cat_list ='var'+'_'+var
    cat_list = pd.get_dummies(project_logit[var], prefix=var)
    data1 = project_logit.join(cat_list)
    project_logit = data1

data_vars = project_logit.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
data_final = project_logit[to_keep]

## data_final 2
project_logit_2 = project.copy()

vsurcond_map = {0:'NA', 1:'DRY', 2:'WET/OIL/SLUSH/MUD', 3:'WET/OIL/SLUSH/MUD', 4:'WET/OIL/SLUSH/MUD', 5:'DRY',
                6:'WET/OIL/SLUSH/MUD', 7:'WET/OIL/SLUSH/MUD', 8:'NA', 10:'WET/OIL/SLUSH/MUD', 11:'WET/OIL/SLUSH/MUD',
                98:'NA', 99:'NA'}
project_logit_2['ROAD_SURFACE'] = project_logit_2['VSURCOND'].map(vsurcond_map)
cat_vars_2=['ROLL','WEATHER_GROUP','LIGHT_Condition','ROAD_SURFACE','ROADWAY_GRADE','ROADWAY_ALIGNMENT','VEHICLE_TYPE','VEHICLE_YEAR']
for var in cat_vars_2:
    cat_list_2='var'+'_'+var
    cat_list_2 = pd.get_dummies(project_logit_2[var], prefix=var)
    data2 = project_logit_2.join(cat_list_2)
    project_logit_2 = data2
data_vars_2 = project_logit_2.columns.values.tolist()
to_keep = [i for i in data_vars_2 if i not in cat_vars_2]
data_final_2 = project_logit_2[to_keep]

#=========================================================
#=========================================================
# GUI code
#=========================================================
#=========================================================


#=========================================================
# Under the 'Model' Menu
#=========================================================


class RandomForest(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = 'Random Forest'
        self.setWindowTitle(self.Title)
        self.statusBar().setStyleSheet('background-color: tomato')  # status bar
        self.setStyleSheet("background-color: lavenderblush")  # background color

        self.main_widget = QWidget(self)
        # create H-layout
        self.layout = QHBoxLayout(self.main_widget)

        #==============================
        # create groupbox 1
        self.groupBox1 = QGroupBox('Random Forest Model Features')
        self.groupBox1.setStyleSheet('font-size: 20px;font-weight: bold; color: red')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        # create features checkbox
        self.x1 = QCheckBox('F1: AGE', self)
        self.x1.setStyleSheet('font-size: 15px;color: black')
        self.x2 = QCheckBox('F2: WEATHER_GROUP_CLEAR/NORMAL', self)
        self.x2.setStyleSheet('font-size: 15px;color: black')
        self.x3 = QCheckBox('F3: WEATHER_GROUP_FOG/CLOUDY', self)
        self.x3.setStyleSheet('font-size: 15px;color: black')
        self.x4 = QCheckBox('F4: WEATHER_GROUP_RAIN/SLEET', self)
        self.x4.setStyleSheet('font-size: 15px;color: black')
        self.x5 = QCheckBox('F5: WEATHER_GROUP_SNOW', self)
        self.x5.setStyleSheet('font-size: 15px;color: black')
        self.x6 = QCheckBox('F6: LIGHT_GROUP_DAWN/DUSK', self)
        self.x6.setStyleSheet('font-size: 15px;color: black')
        self.x7 = QCheckBox('F7: LIGHT_GROUP_LIGHT', self)
        self.x7.setStyleSheet('font-size: 15px;color: black')
        self.x8 = QCheckBox('F8: SURFACE_GROUP_DRY', self)
        self.x8.setStyleSheet('font-size: 15px;color: black')
        self.x9 = QCheckBox('F9: SURFACE_GROUP_WET', self)
        self.x9.setStyleSheet('font-size: 15px;color: black')
        self.x10 = QCheckBox('F10: GRADE_GROUP_LEVEL', self)
        self.x10.setStyleSheet('font-size: 15px;color: black')
        self.x11 = QCheckBox('F11: ALIGN_GROUP_STRAIGHT', self)
        self.x11.setStyleSheet('font-size: 15px;color: black')
        self.x12 = QCheckBox('F12: VEHICLE_TYPE_CAR', self)
        self.x12.setStyleSheet('font-size: 15px;color: black')
        self.x13 = QCheckBox('F13: VEHICLE_TYPE_PICKUP', self)
        self.x13.setStyleSheet('font-size: 15px;color: black')
        self.x14 = QCheckBox('F14: VEHICLE_TYPE_VAN', self)
        self.x14.setStyleSheet('font-size: 15px;color: black')
        self.x15 = QCheckBox('F15: VEHICLE_YEAR 2008-2019', self)
        self.x15.setStyleSheet('font-size: 15px;color: black')
        self.x16 = QCheckBox('F16: VEHICLE_YEAR 2011-2019', self)
        self.x16.setStyleSheet('font-size: 15px;color: black')
        self.x17 = QCheckBox('F17: GENDER_FEMALE', self)
        self.x17.setStyleSheet('font-size: 15px;color: black')
        self.x1.setChecked(True)
        self.x2.setChecked(True)
        self.x3.setChecked(True)
        self.x4.setChecked(True)
        self.x5.setChecked(True)
        self.x6.setChecked(True)
        self.x7.setChecked(True)
        self.x8.setChecked(True)
        self.x9.setChecked(True)
        self.x10.setChecked(True)
        self.x11.setChecked(True)
        self.x12.setChecked(True)
        self.x13.setChecked(True)
        self.x14.setChecked(True)
        self.x15.setChecked(True)
        self.x16.setChecked(True)
        self.x17.setChecked(True)

        # add checkbox to layout
        self.groupBox1Layout.addWidget(self.x1)
        self.groupBox1Layout.addWidget(self.x2)
        self.groupBox1Layout.addWidget(self.x3)
        self.groupBox1Layout.addWidget(self.x4)
        self.groupBox1Layout.addWidget(self.x5)
        self.groupBox1Layout.addWidget(self.x6)
        self.groupBox1Layout.addWidget(self.x7)
        self.groupBox1Layout.addWidget(self.x8)
        self.groupBox1Layout.addWidget(self.x9)
        self.groupBox1Layout.addWidget(self.x10)
        self.groupBox1Layout.addWidget(self.x11)
        self.groupBox1Layout.addWidget(self.x12)
        self.groupBox1Layout.addWidget(self.x13)
        self.groupBox1Layout.addWidget(self.x14)
        self.groupBox1Layout.addWidget(self.x15)
        self.groupBox1Layout.addWidget(self.x16)
        self.groupBox1Layout.addWidget(self.x17)

        # create test split label input
        self.lblPercentTest = QLabel('Test Dataset Split (%):')
        self.lblPercentTest.setStyleSheet('font-size: 20px;font-weight: bold; color: red')
        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setStyleSheet('font-size: 20px;font-weight: bold; color: black; background-color: white')
        # create N of Trees label input
        self.lbltreeN = QLabel('# of Trees in Forest (n_estimators):')
        self.lbltreeN.setStyleSheet('font-size: 20px;font-weight: bold; color: red')
        self.txttreeN = QLineEdit(self)
        self.txttreeN.setStyleSheet('font-size: 20px;font-weight: bold; color: black; background-color: white')
        # create Execute button
        self.btnExecute = QPushButton("Run Selected Features")
        self.btnExecute.setStyleSheet('font-size: 15px; color:black; background-color: aquamarine')
        self.btnExecute.clicked.connect(self.update)
        # create accuracy button
        self.lblAccuracy = QLabel('Accuracy (%):')
        self.lblAccuracy.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.txtAccuracy = QLineEdit()
        self.txtAccuracy.setStyleSheet('font-size: 20px; color: black; background-color: white')

        # add these labels above
        self.groupBox1Layout.addWidget(self.lblPercentTest)
        self.groupBox1Layout.addWidget(self.txtPercentTest)
        self.groupBox1Layout.addWidget(self.lbltreeN)
        self.groupBox1Layout.addWidget(self.txttreeN)
        self.groupBox1Layout.addWidget(self.btnExecute)
        self.groupBox1Layout.addWidget(self.lblAccuracy)
        self.groupBox1Layout.addWidget(self.txtAccuracy)
        #==============================================
        # groupbox2
        #==============================================
        self.groupBox2 = QGroupBox('Model Metrics & Results')
        self.groupBox2.setStyleSheet('font-size: 20px;font-weight: bold; color: red')
        self.groupBox2Layout = QGridLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        #==========================================
        # Chart 1 : Confusion Matrix
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes1 = [self.ax1]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.groupBox2Layout.addWidget(self.canvas, 0, 0)

        # ==========================================
        # Chart 2: Features Importance
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.groupBox2Layout.addWidget(self.canvas2, 0, 1)

        #==========================================
        # Classification report
        self.txtClassification = QPlainTextEdit()
        self.txtClassification.setStyleSheet('font-size: 20px; font-weight: bold; color: black; background-color: white')
        self.groupBox2Layout.addWidget(self.txtClassification, 1,0)

        #==========================================
        # Chart 3: ROC Curve
        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)
        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas3.updateGeometry()
        self.groupBox2Layout.addWidget(self.canvas3, 1,1)

        #:==========================================
        #:==========================================
        # add groupbox 1,2 to master layout
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)

        # Creates the window with all the elements
        self.setCentralWidget(self.main_widget)
        self.resize(1700, 900)
        self.show()

    def update(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.txtClassification.clear()
        self.txtClassification.setUndoRedoEnabled(False)

        # process the parameters
        feature_list = ['AGE', 'WEATHER_GROUP_CLEAR/NORMAL', 'WEATHER_GROUP_FOG/CLOUDY', 'WEATHER_GROUP_RAIN/SLEET',
                        'WEATHER_GROUP_SNOW',
                        'LIGHT_Condition_DAWN/DUSK', 'LIGHT_Condition_LIGHT', 'ROAD_SURFACE_DRY', 'ROAD_SURFACE_WET',
                        'ROADWAY_GRADE_LEVEL',
                        'ROADWAY_ALIGNMENT_STRAIGHT', 'VEHICLE_TYPE_CAR', 'VEHICLE_TYPE_PICKUP', 'VEHICLE_TYPE_VAN',
                        'VEHICLE_YEAR_2008-2019',
                        'VEHICLE_YEAR_2011-2019', 'GENDER_FEMALE']

        self.F_List = pd.DataFrame([])
        feature_list = ['AGE', 'WEATHER_GROUP_CLEAR/NORMAL','WEATHER_GROUP_FOG/CLOUDY','WEATHER_GROUP_RAIN/SLEET','WEATHER_GROUP_SNOW',
                        'LIGHT_Condition_DAWN/DUSK','LIGHT_Condition_LIGHT','ROAD_SURFACE_DRY','ROAD_SURFACE_WET','ROADWAY_GRADE_LEVEL',
                        'ROADWAY_ALIGNMENT_STRAIGHT','VEHICLE_TYPE_CAR','VEHICLE_TYPE_PICKUP','VEHICLE_TYPE_VAN','VEHICLE_YEAR_2008-2019',
                        'VEHICLE_YEAR_2011-2019','GENDER_FEMALE']

        if self.x1.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[0]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[0]]], axis=1)

        if self.x2.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[1]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[1]]], axis=1)

        if self.x3.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[2]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[2]]], axis=1)

        if self.x4.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[3]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[3]]], axis=1)

        if self.x5.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[4]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[4]]], axis=1)

        if self.x6.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[5]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[5]]], axis=1)

        if self.x7.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[6]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[6]]], axis=1)

        if self.x8.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[7]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[7]]], axis=1)

        if self.x9.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[8]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[8]]], axis=1)

        if self.x10.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[9]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[9]]], axis=1)

        if self.x11.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[10]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[10]]], axis=1)

        if self.x12.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[11]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[11]]], axis=1)

        if self.x13.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[12]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[12]]], axis=1)

        if self.x14.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[13]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[13]]], axis=1)

        if self.x15.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[14]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[14]]], axis=1)

        if self.x16.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[15]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[15]]], axis=1)

        if self.x17.isChecked():
            if len(self.F_List) == 0:
                self.F_List = data_final[feature_list[16]]
            else:
                self.F_List = pd.concat([self.F_List, data_final[feature_list[16]]], axis=1)

        Xysplit = float(self.txtPercentTest.text()) / 100
        n = int(self.txttreeN.text())

        # ==========================
        # Build the model
        # ==========================
        #feature_list = ['AGE', 'WEATHER_GROUP_CLEAR/NORMAL', 'WEATHER_GROUP_FOG/CLOUDY', 'WEATHER_GROUP_RAIN/SLEET',
        #            'WEATHER_GROUP_SNOW',
         #           'LIGHT_Condition_DAWN/DUSK', 'LIGHT_Condition_LIGHT', 'ROAD_SURFACE_DRY', 'ROAD_SURFACE_WET',
          #          'ROADWAY_GRADE_LEVEL',
          #          'ROADWAY_ALIGNMENT_STRAIGHT', 'VEHICLE_TYPE_CAR', 'VEHICLE_TYPE_PICKUP', 'VEHICLE_TYPE_VAN',
          #          'VEHICLE_YEAR_2008-2019',
          #          'VEHICLE_YEAR_2011-2019', 'GENDER_FEMALE']

        X = self.F_List
        #X = data_final[feature_list]
        y = le.fit_transform(project['ROLL'])

        # split X and y
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8, test_size=Xysplit)
        # perform training with entropy.
        # create decision tree classifier
        self.clf = RandomForestClassifier(n_estimators=n, random_state=8)

        # Performing training
        self.clf.fit(X_train, y_train)
        # make predictions using entropy

        # predicton on test using all features
        y_pred = self.clf.predict(X_test)
        y_pred_proba = self.clf.predict_proba(X_test)

        # confusion matrix for gini model
        conf_matrix = confusion_matrix(y_test, y_pred)

        # clasification report
        self.class_rep = classification_report(y_test, y_pred)
        self.txtClassification.appendPlainText(self.class_rep)

        # accuracy score
        self.accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtAccuracy.setText(str(self.accuracy_score))
        ACC_Forest = self.accuracy_score

        # ==============================================
        # Chart 1 -- Confusion Matrix
        # ==============================================

        class_names = ['Rollover', 'Non-Rollover']

        self.ax1.matshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        self.ax1.set_yticklabels(['', 'Rollover', 'Non-Rollover'])
        self.ax1.set_xticklabels(['', 'Rollover', 'Non-Rollover'], rotation=90)
        self.ax1.set_xlabel('Predicted Label')
        self.ax1.set_ylabel('True Label')

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        # ====================================
        # Chart 2 - Feature Importance
        # ====================================

        importances = self.clf.feature_importances_
        f_importances = pd.Series(importances, X.columns)
        f_importances.sort_values(ascending=True, inplace=True)

        X_Features = f_importances.index
        y_Importance = list(f_importances)

        self.ax2.barh(X_Features, y_Importance)
        self.ax2.set_title('Feature Importance')

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        # ====================================
        # Chart 3 - ROC Cure
        # ====================================
        y_pred_proba = self.clf.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])

        self.ax3.plot(fpr, tpr, color='lightcoral', lw=5, label='ROC Curve (area = %0.2f)' % auc)
        self.ax3.plot([0, 1], [0, 1], color='turquoise', lw=5, ls='--')
        self.ax3.legend(loc="lower right")

        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

class KNNmodel(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KNNmodel, self).__init__()
        self.Title = 'K-Nearest Neighbors (KNN)'
        self.setWindowTitle(self.Title)
        self.statusBar().setStyleSheet('background-color: tomato')  # status bar
        self.setStyleSheet('background-color: lavenderblush')  # background color
        self.main_widget = QWidget(self)
        # create H-layout layout
        self.layout = QHBoxLayout(self.main_widget)
        # create groupbox1 layout
        self.groupBox1 = QGroupBox('KNN Model Features')
        self.groupBox1.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        # create checkbox
        self.x1 = QCheckBox('F1: Weather', self)
        self.x1.setStyleSheet('font-size: 15px; font-weight: bold; color: black')
        self.x2 = QCheckBox('F2: Light Condition', self)
        self.x2.setStyleSheet('font-size: 15px; font-weight: bold; color: black')
        self.x3 = QCheckBox('F3: Road Surface', self)
        self.x3.setStyleSheet('font-size: 15px; font-weight: bold; color: black')
        self.x4 = QCheckBox('F4: Roadway Grade', self)
        self.x4.setStyleSheet('font-size: 15px; font-weight: bold; color: black')
        self.x5 = QCheckBox('F5: Vehicle Type', self)
        self.x5.setStyleSheet('font-size: 15px; font-weight: bold; color: black')
        self.x6 = QCheckBox('F6: Roadway Alignment', self)
        self.x6.setStyleSheet('font-size: 15px; font-weight: bold; color: black')
        self.x7 = QCheckBox('F7: LIGHT_GROUP_LIGHT', self)
        self.x7.setStyleSheet('font-size: 15px; font-weight: bold; color: black')
        self.x8 = QCheckBox('F8: Vehicle Year', self)
        self.x8.setStyleSheet('font-size: 15px; font-weight: bold; color: black')
        self.x9 = QCheckBox('F9: Gender', self)
        self.x9.setStyleSheet('font-size: 15px; font-weight: bold; color: black')
        self.x1.setChecked(True)
        self.x2.setChecked(True)
        self.x3.setChecked(True)
        self.x4.setChecked(True)
        self.x5.setChecked(True)
        self.x6.setChecked(True)
        self.x7.setChecked(True)
        self.x8.setChecked(True)
        self.x9.setChecked(True)

        # create find_K button
        self.btnK = QPushButton('Find the Best K (1-50)')
        self.btnK.setStyleSheet('font-size: 20px; color: black; background-color: aquamarine')
        self.btnK.clicked.connect(self.updateK)

        # create Execute button
        self.btnExecute = QPushButton("Run Selected Features")
        self.btnExecute.setStyleSheet('font-size: 20px; color: black; background-color: aquamarine')
        self.btnExecute.clicked.connect(self.update)

        # add checkbox to layout
        self.groupBox1Layout.addWidget(self.x1)
        self.groupBox1Layout.addWidget(self.x2)
        self.groupBox1Layout.addWidget(self.x3)
        self.groupBox1Layout.addWidget(self.x4)
        self.groupBox1Layout.addWidget(self.x5)
        self.groupBox1Layout.addWidget(self.x6)
        self.groupBox1Layout.addWidget(self.x7)
        self.groupBox1Layout.addWidget(self.x8)
        self.groupBox1Layout.addWidget(self.x9)
        self.groupBox1Layout.addWidget(self.btnK)
        self.groupBox1Layout.addWidget(self.btnExecute)

        # show K element
        self.lblK = QLabel('The Best K is: ')
        self.lblK.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.txtK = QLineEdit()
        self.txtK.setStyleSheet('font-size: 20px; font-weight: bold; color: black; background-color: white')
        # roc element
        self.lblroc = QLabel('ROC_AUC Score (%):')
        self.lblroc.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.txtroc = QLineEdit()
        self.txtroc.setStyleSheet('font-size: 20px; font-weight: bold; color: black; background-color: white')
        # accuracy element
        self.lblAccuracy = QLabel('Single Accuracy (%):')
        self.lblAccuracy.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.txtAccuracy = QLineEdit()
        self.txtAccuracy.setStyleSheet('font-size: 20px; color: black; background-color: white')
        # accuracy10 element
        self.lblAccuracy10 = QLabel('10-Fold Accuracy (%):')
        self.lblAccuracy10.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.txtAccuracy10 = QLineEdit()
        self.txtAccuracy10.setStyleSheet('font-size: 20px; color: black; background-color: white')
        # average accuracy element
        self.lblAccuracy0 = QLabel('Average Accuracy (%):')
        self.lblAccuracy0.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.txtAccuracy0 = QLineEdit()
        self.txtAccuracy0.setStyleSheet('font-size: 20px; font-weight: bold; color: black; background-color: white')
        # add elements to groupbox1
        self.groupBox1Layout.addWidget(self.lblK)
        self.groupBox1Layout.addWidget(self.txtK)
        self.groupBox1Layout.addWidget(self.lblroc)
        self.groupBox1Layout.addWidget(self.txtroc)
        self.groupBox1Layout.addWidget(self.lblAccuracy)
        self.groupBox1Layout.addWidget(self.txtAccuracy)
        self.groupBox1Layout.addWidget(self.lblAccuracy10)
        self.groupBox1Layout.addWidget(self.txtAccuracy10)
        self.groupBox1Layout.addWidget(self.lblAccuracy0)
        self.groupBox1Layout.addWidget(self.txtAccuracy0)

        # create groupbox2 layout
        self.groupBox2 = QGroupBox('KNN Model Results')
        self.groupBox2.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        # groupbox2 create elements

        ## classification elements
        self.txtClassification = QPlainTextEdit()
        self.txtClassification.setStyleSheet(
            'font-size: 15px; font-weight: bold; color: black; background-color: white')
        self.groupBox2Layout.addWidget(self.txtClassification)

        ## chart 1 - confustion matrix
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet('background-color: lavenderblush')
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.groupBox2Layout.addWidget(self.canvas)

        ## chart 2 - Best K
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.groupBox2Layout.addWidget(self.canvas2)

        # add groupbox 1,2 to layout
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)

        # Creates the window with all the elements
        self.setCentralWidget(self.main_widget)
        self.resize(1200, 1000)
        self.show()

    def updateK(self):
        self.ax2.clear()

        ''''# process the features
        le = preprocessing.LabelEncoder()
        weather_encoded = le.fit_transform(project['WEATHER_GROUP'])
        light_encoded = le.fit_transform(project['LIGHT_Condition'])
        surface_encoded = le.fit_transform(project['ROAD_SURFACE'])
        grade_encoded = le.fit_transform(project['ROADWAY_GRADE'])
        vehicle_encoded = le.fit_transform(project['VEHICLE_TYPE'])
        align_encoded = le.fit_transform(project['ROADWAY_ALIGNMENT'])
        model_year_encoded = le.fit_transform(project['VEHICLE_YEAR'])
        gender_encoded = le.fit_transform(project['GENDER'])

        # Standarize AGE
        column = 'AGE'
        age_normal = (project[column] - project[column].min()) / (project[column].max() - project[column].min())

        # Combine all converted features into a list of tuples
        X = list(zip(weather_encoded, light_encoded, surface_encoded, grade_encoded, vehicle_encoded, align_encoded,
                     model_year_encoded,
                     gender_encoded, age_normal))

        # Convert string labels of ROLL into numbers: 0:ROLLOVER, 1:NON-ROLLOVER
        y = le.fit_transform(project['ROLL'])

        # build the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8, test_size=0.3)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_std = scaler.transform(X_train)
        X_test_std = scaler.transform(X_test)

        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train_std, y_train)
        y_pred = clf.predict(X_test_std)
        y_pred_proba = clf.predict_proba(X_test_std)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # find best K
        accuracy = []
        for i in range(1, 20):
            clf = KNeighborsClassifier(n_neighbors=i)
            clf.fit(X_train_std, y_train)
            y_pred = clf.predict(X_test_std)
            accuracy.append(accuracy_score(y_test, y_pred))

        self.bestK = max(accuracy)
        self.txtClassification.appendPlainText(self.bestK)'''

        self.bestK=40
        K50_acc = [0.5912251387915701,0.6397348137767477,0.6292782838354983,0.6505147415512316,0.6449630787473724,0.6556352072441115,0.6525629278283835,0.6610251711313534,
 0.6578450924378807,0.6639896512693365,0.6595159812429257,0.6640974505470814,0.6654449415188918,0.6684094216568749,0.6651215436856573,
 0.6676548267126611,0.6670619306850644,0.66830162237913,0.6698108122675578,0.6714278014337304,0.6678704252681507,0.6708349054061338,
 0.66830162237913,0.6736376866274996,0.672020697461327,0.6734759877108824,0.672397994933434,0.6754702743491618,0.6725596938500512,
 0.6745000808494583, 0.6712122028782407,0.6746617797660756,0.6748234786826929,0.674068883738479,0.6736376866274996,0.6755241739880343,
 0.6748773783215652,0.6766021667654827,0.6762787689322481,0.6767638656820999,0.6746078801272032,0.6760092707378861,0.674338381932841,
 0.6755241739880343,0.6753085754325446,0.6765482671266102,0.6746078801272032,0.6760631703767584,0.6749851775993101]

        self.txtK.setText(str(self.bestK))
        self.ax2.plot(np.arange(1,50), K50_acc, color='lightcoral', ls='dashed', marker='o', markerfacecolor='turquoise', markersize=10)
        #self.ax2.title('K 1-50 Accuracy Scores')
       # self.ax2.xlabel('K value')
       # self.ax2.ylabel('Accuracy')

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

    def update(self):
        self.ax1.clear()
        self.txtClassification.clear()
        self.txtClassification.setUndoRedoEnabled(False)

        # process the features
        le = LabelEncoder()
        weather_encoded = le.fit_transform(project['WEATHER_GROUP'])
        light_encoded = le.fit_transform(project['LIGHT_Condition'])
        surface_encoded = le.fit_transform(project['ROAD_SURFACE'])
        grade_encoded = le.fit_transform(project['ROADWAY_GRADE'])
        vehicle_encoded = le.fit_transform(project['VEHICLE_TYPE'])
        align_encoded = le.fit_transform(project['ROADWAY_ALIGNMENT'])
        model_year_encoded = le.fit_transform(project['VEHICLE_YEAR'])
        gender_encoded = le.fit_transform(project['GENDER'])

        # Standarize AGE
        column = 'AGE'
        age_normal = (project[column] - project[column].min()) / (project[column].max() - project[column].min())

        # Combine all converted features into a list of tuples
        X = list(zip(weather_encoded, light_encoded, surface_encoded, grade_encoded, vehicle_encoded, align_encoded,
                     model_year_encoded,
                     gender_encoded, age_normal))

        # Convert string labels of ROLL into numbers: 0:ROLLOVER, 1:NON-ROLLOVER
        y = le.fit_transform(project['ROLL'])

        # build the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8, test_size=0.3)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_std = scaler.transform(X_train)
        X_test_std = scaler.transform(X_test)

        clf = KNeighborsClassifier(n_neighbors=40)
        clf.fit(X_train_std, y_train)
        y_pred = clf.predict(X_test_std)
        y_pred_proba = clf.predict_proba(X_test_std)
        conf_matrix = confusion_matrix(y_test, y_pred)

        #============
        # clasification report
        self.cla_report = classification_report(y_test, y_pred)
        self.txtClassification.appendPlainText(self.cla_report)

        #============
        # chart 1 - confusion matrix
        class_names = ['Rollover', 'Non-Rollover']
        self.ax1.matshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        self.ax1.set_yticklabels(['', 'Rollover', 'Non-Rollover'])
        self.ax1.set_xticklabels(['', 'Rollover', 'Non-Rollover'], rotation=90)
        self.ax1.set_xlabel('Predicted Label')
        self.ax1.set_ylabel('True Label')

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                self.ax1.text(j, i, str(conf_matrix[i][j]))
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #==================
        # roc score
        self.roc_score = roc_auc_score(y_test, y_pred_proba[:, 1]) * 100
        self.txtroc.setText(str(self.roc_score))
        # accuracy score
        self.acc_score = accuracy_score(y_test, y_pred) * 100
        self.txtAccuracy.setText(str(self.acc_score))
        # accuracy 10 score
        K_Fold = KFold(n_splits=10, random_state=8, shuffle=True)
        self.acc_score10 = cross_val_score(clf, X_train_std, y_train, cv=K_Fold, n_jobs=-1) * 100
        self.txtAccuracy10.setText(str(self.acc_score10))
        # average accuracy score
        self.acc_score0 = sum(self.acc_score10) / 10
        self.txtAccuracy0.setText(str(self.acc_score0))

        # grab the accuracy score for comparison
        ACC_KNN = self.acc_score0

class LogRegression(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(LogRegression, self).__init__()
        self.Title = 'Logistic Regression'
        self.setWindowTitle(self.Title)
        self.statusBar().setStyleSheet('background-color: tomato')  # status bar
        self.setStyleSheet('background-color: lavenderblush')  # background color
        self.main_widget = QWidget(self)
        # create H-layout layout
        self.layout = QHBoxLayout(self.main_widget)
        # create groupbox1 layout
        self.groupBox1 = QGroupBox('Logistic Regression Model Features')
        self.groupBox1.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        # create checkbox
        self.x1 = QCheckBox('F1: AGE', self)
        self.x1.setStyleSheet('font-size: 15px;color: black')
        self.x2 = QCheckBox('F2: WEATHER_GROUP_CLEAR/NORMAL', self)
        self.x2.setStyleSheet('font-size: 15px;color: black')
        self.x3 = QCheckBox('F3: WEATHER_GROUP_FOG/CLOUDY', self)
        self.x3.setStyleSheet('font-size: 15px;color: black')
        self.x4 = QCheckBox('F4: WEATHER_GROUP_RAIN/SLEET', self)
        self.x4.setStyleSheet('font-size: 15px;color: black')
        self.x5 = QCheckBox('F5: WEATHER_GROUP_SNOW', self)
        self.x5.setStyleSheet('font-size: 15px;color: black')
        self.x6 = QCheckBox('F6: LIGHT_GROUP_DAWN/DUSK', self)
        self.x6.setStyleSheet('font-size: 15px;color: black')
        self.x7 = QCheckBox('F7: LIGHT_GROUP_LIGHT', self)
        self.x7.setStyleSheet('font-size: 15px;color: black')
        self.x8 = QCheckBox('F8: SURFACE_GROUP_DRY', self)
        self.x8.setStyleSheet('font-size: 15px;color: black')
        self.x9 = QCheckBox('F9: SURFACE_GROUP_WET', self)
        self.x9.setStyleSheet('font-size: 15px;color: black')
        self.x89 = QCheckBox('F8+9: SURFACE_GROUP_WET/OIL/SLUSH/MUD', self)
        self.x89.setStyleSheet('font-size: 15px;color: black')
        self.x10 = QCheckBox('F10: GRADE_GROUP_LEVEL', self)
        self.x10.setStyleSheet('font-size: 15px;color: black')
        self.x11 = QCheckBox('F11: ALIGN_GROUP_STRAIGHT', self)
        self.x11.setStyleSheet('font-size: 15px;color: black')
        self.x12 = QCheckBox('F12: VEHICLE_TYPE_CAR', self)
        self.x12.setStyleSheet('font-size: 15px;color: black')
        self.x13 = QCheckBox('F13: VEHICLE_TYPE_PICKUP', self)
        self.x13.setStyleSheet('font-size: 15px;color: black')
        self.x14 = QCheckBox('F14: VEHICLE_TYPE_VAN', self)
        self.x14.setStyleSheet('font-size: 15px;color: black')
        self.x15 = QCheckBox('F15: VEHICLE_YEAR 2008-2019', self)
        self.x15.setStyleSheet('font-size: 15px;color: black')
        self.x16 = QCheckBox('F16: VEHICLE_YEAR 2011-2019', self)
        self.x16.setStyleSheet('font-size: 15px;color: black')
        self.x17 = QCheckBox('F17: GENDER_FEMALE', self)
        self.x17.setStyleSheet('font-size: 15px;color: black')
        self.x1.setChecked(True)
        self.x2.setChecked(True)
        self.x3.setChecked(True)
        self.x4.setChecked(True)
        self.x5.setChecked(True)
        self.x6.setChecked(True)
        self.x7.setChecked(True)
        self.x8.setChecked(True)
        self.x9.setChecked(True)
        self.x89.setChecked(False)
        self.x10.setChecked(True)
        self.x11.setChecked(True)
        self.x12.setChecked(True)
        self.x13.setChecked(True)
        self.x14.setChecked(True)
        self.x15.setChecked(True)
        self.x16.setChecked(True)
        self.x17.setChecked(True)

        # create Execute button
        self.btnExecute = QPushButton("Run Selected Features")
        self.btnExecute.setStyleSheet('font-size: 20px; font-weight: bold; color: black; background-color: aquamarine')
        self.btnExecute.clicked.connect(self.update)

        # add checkbox to layout
        self.groupBox1Layout.addWidget(self.x1)
        self.groupBox1Layout.addWidget(self.x2)
        self.groupBox1Layout.addWidget(self.x3)
        self.groupBox1Layout.addWidget(self.x4)
        self.groupBox1Layout.addWidget(self.x5)
        self.groupBox1Layout.addWidget(self.x6)
        self.groupBox1Layout.addWidget(self.x7)
        self.groupBox1Layout.addWidget(self.x8)
        self.groupBox1Layout.addWidget(self.x9)
        self.groupBox1Layout.addWidget(self.x89)
        self.groupBox1Layout.addWidget(self.x10)
        self.groupBox1Layout.addWidget(self.x11)
        self.groupBox1Layout.addWidget(self.x12)
        self.groupBox1Layout.addWidget(self.x13)
        self.groupBox1Layout.addWidget(self.x14)
        self.groupBox1Layout.addWidget(self.x15)
        self.groupBox1Layout.addWidget(self.x16)
        self.groupBox1Layout.addWidget(self.x17)
        self.groupBox1Layout.addWidget(self.btnExecute)

        # create groupbox2 layout
        self.groupBox2 = QGroupBox()
        # groupbox2 layout
        self.groupBox2 = QGroupBox('Logistic Regression Model Results')
        self.groupBox2.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        # groupbox2 create elements
        ## chart 1
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet('background-color: lavenderblush')
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.groupBox2Layout.addWidget(self.canvas)

        # rest elements
        self.txtClassification = QPlainTextEdit()
        self.txtClassification.setStyleSheet('font-size: 15px; font-weight: bold; color: black; background-color: white')
        #self.txtClassification.setAlignment(QtCore.Qt.AlignLeft)

        # roc element
        self.lblroc = QLabel('ROC_AUC Score (%):')
        self.lblroc.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.txtroc = QLineEdit()
        self.txtroc.setStyleSheet('font-size: 20px; font-weight: bold; color: black; background-color: white')
        # accuracy element
        self.lblAccuracy = QLabel('Single Accuracy (%):')
        self.lblAccuracy.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.txtAccuracy = QLineEdit()
        self.txtAccuracy.setStyleSheet('font-size: 20px; color: black; background-color: white')
        # accuracy10 element
        self.lblAccuracy10 = QLabel('10-Fold Accuracy (%):')
        self.lblAccuracy10.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.txtAccuracy10 = QLineEdit()
        self.txtAccuracy10.setStyleSheet('font-size: 15px; color: black; background-color: white')
        # average accuracy element
        self.lblAccuracy0 = QLabel('Average Accuracy (%):')
        self.lblAccuracy0.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.txtAccuracy0 = QLineEdit()
        self.txtAccuracy0.setStyleSheet('font-size: 20px; font-weight: bold; color: black; background-color: white')
        # add elements to groupbox2
        self.groupBox2Layout.addWidget(self.canvas)
        self.groupBox2Layout.addWidget(self.txtClassification)
        self.groupBox2Layout.addWidget(self.lblroc)
        self.groupBox2Layout.addWidget(self.txtroc)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)
        self.groupBox2Layout.addWidget(self.lblAccuracy10)
        self.groupBox2Layout.addWidget(self.txtAccuracy10)
        self.groupBox2Layout.addWidget(self.lblAccuracy0)
        self.groupBox2Layout.addWidget(self.txtAccuracy0)
        # add groupbox 1,2 to layout
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)

        # Creates the window with all the elements
        self.setCentralWidget(self.main_widget)
        self.resize(1200, 1000)
        self.show()

    def update(self):
        self.ax1.clear()
        self.txtClassification.clear()
        self.txtClassification.setUndoRedoEnabled(False)

        if self.x89.isChecked():
            ROLLOVER = data_final_2['ROLL_ROLLOVER']
            AGE = data_final_2['AGE']
            WEATHER_CLEAR_NORMAL = data_final_2['WEATHER_GROUP_CLEAR/NORMAL']
            WEATHER_FOG_CLOUDY = data_final_2['WEATHER_GROUP_FOG/CLOUDY']
            WEATHER_RAIN_SLEET = data_final_2['WEATHER_GROUP_RAIN/SLEET']
            WEATHER_SNOW = data_final_2['WEATHER_GROUP_SNOW']
            LIGHT_DAWN_DUSK = data_final_2['LIGHT_Condition_DAWN/DUSK']
            LIGHT_LIGHT = data_final_2['LIGHT_Condition_LIGHT']
            SURFACE_WET_OIL = data_final_2['ROAD_SURFACE_WET/OIL/SLUSH/MUD']
            GRADE_LEVEL = data_final_2['ROADWAY_GRADE_LEVEL']
            ALIGN_STRAIGHT = data_final_2['ROADWAY_ALIGNMENT_STRAIGHT']
            VEHICLE_CAR = data_final_2['VEHICLE_TYPE_CAR']
            VEHICLE_PICKUP = data_final_2['VEHICLE_TYPE_PICKUP']
            VEHICLE_VAN = data_final_2['VEHICLE_TYPE_VAN']
            MY_2008_2019 = data_final_2['VEHICLE_YEAR_2008-2019']
            MY_2011_2019 = data_final_2['VEHICLE_YEAR_2011-2019']
            # Features
            X_2 = list(zip(AGE, WEATHER_CLEAR_NORMAL, WEATHER_FOG_CLOUDY, WEATHER_RAIN_SLEET, WEATHER_SNOW, LIGHT_DAWN_DUSK,
                    LIGHT_LIGHT, SURFACE_WET_OIL, GRADE_LEVEL, ALIGN_STRAIGHT, VEHICLE_CAR, VEHICLE_PICKUP, VEHICLE_VAN,MY_2008_2019,MY_2011_2019))
            # Target
            y_2 = list(ROLLOVER)

            #build model 2
            X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, random_state=8, test_size=0.3)
            clf = LogisticRegression()
            clf.fit(X_train_2, y_train_2)
            y_pred = clf.predict(X_test_2)
            y_pred_proba = clf.predict_proba(X_test_2)
            conf_matrix = confusion_matrix(y_test_2, y_pred)

            # confusion matrix chart
            class_names = ['Rollover', 'Non-Rollover']
            self.ax1.matshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
            self.ax1.set_yticklabels(['', 'Rollover', 'Non-Rollover'])
            self.ax1.set_xticklabels(['', 'Rollover', 'Non-Rollover'], rotation=90)
            self.ax1.set_xlabel('Predicted Label')
            self.ax1.set_ylabel('True Label')

            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    self.ax1.text(j, i, str(conf_matrix[i][j]))

            self.fig.tight_layout()
            self.fig.canvas.draw_idle()

            # clasification report
            self.cla_report = classification_report(y_test_2, y_pred)
            self.txtClassification.appendPlainText(self.cla_report)
            # roc score
            self.roc_score = roc_auc_score(y_test_2, y_pred_proba[:, 1]) * 100
            self.txtroc.setText(str(self.roc_score))
            # accuracy score
            self.acc_score = accuracy_score(y_test_2, y_pred) * 100
            self.txtAccuracy.setText(str(self.acc_score))
            # accuracy 10 score
            K_Fold = KFold(n_splits=10, random_state=8, shuffle=True)
            self.acc_score10 = cross_val_score(clf, X_train_2, y_train_2, cv=K_Fold, n_jobs=-1)*100
            self.txtAccuracy10.setText(str(self.acc_score10))
            # average accuracy score
            self.acc_score0 = sum(self.acc_score10) / 10
            self.txtAccuracy0.setText(str(self.acc_score0))

            # grab the accuracy score for comparison
            ACC_Logit = self.acc_score0

        #::=====================================
        #::=====================================
        #::=====================================

        if not self.x89.isChecked():

            # features
            ROLLOVER = data_final['ROLL_ROLLOVER']
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
            X = list(zip(AGE, WEATHER_CLEAR_NORMAL, WEATHER_FOG_CLOUDY, WEATHER_RAIN_SLEET, WEATHER_SNOW, LIGHT_DAWN_DUSK,
                         LIGHT_LIGHT,
                         SURFACE_DRY, SURFACE_WET, GRADE_LEVEL, ALIGN_STRAIGHT, VEHICLE_CAR, VEHICLE_PICKUP, VEHICLE_VAN,
                         MY_2008_2019,
                         MY_2011_2019, GENDER_FEMALE))

            y = list(ROLLOVER)

            # build model 1
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8, test_size=0.3)
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # confusion matrix chart
            class_names = ['Rollover','Non-Rollover']
            self.ax1.matshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
            self.ax1.set_yticklabels(['','Rollover','Non-Rollover'])
            self.ax1.set_xticklabels(['','Rollover','Non-Rollover'],rotation = 90)
            self.ax1.set_xlabel('Predicted Label')
            self.ax1.set_ylabel('True Label')

            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    self.ax1.text(j, i, str(conf_matrix[i][j]))

            self.fig.tight_layout()
            self.fig.canvas.draw_idle()

            # clasification report
            self.cla_report = classification_report(y_test, y_pred)
            self.txtClassification.appendPlainText(self.cla_report)
            # roc score
            self.roc_score = roc_auc_score(y_test, y_pred_proba[:,1]) * 100
            self.txtroc.setText(str(self.roc_score))
            # accuracy score
            self.acc_score = accuracy_score(y_test, y_pred) * 100
            self.txtAccuracy.setText(str(self.acc_score))
            # accuracy 10 score
            K_Fold = KFold(n_splits=10, random_state=8, shuffle=True)
            self.acc_score10 = cross_val_score(clf, X_train, y_train, cv=K_Fold, n_jobs=-1)*100
            self.txtAccuracy10.setText(str(self.acc_score10))
            # average accuracy score
            self.acc_score0 = sum(self.acc_score10)/10
            self.txtAccuracy0.setText(str(self.acc_score0))

class DecisionTree(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree, self).__init__()
        self.Title = 'Decision Tree'
        self.setWindowTitle(self.Title)
        self.resize(1200, 800)  # Resize the window
        self.statusBar().setStyleSheet("background-color: tomato")  # status bar
        self.setStyleSheet("background-color: lavenderblush")        # background color

        # pass the ACC scores to model comparison windows
       # self.model_compare = ModelCompare()
        #=========================================
        #=========================================

        self.main_widget = QWidget(self)
        # create grid layout
        self.layout = QGridLayout(self.main_widget)
        # create groupbox 1
        self.groupBox1 = QGroupBox('Decision Tree Model Features')
        self.groupBox1.setStyleSheet('font-size: 20px;font-weight: bold; color: red')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        feature_list = ['Weather', 'Light Condition', 'Road Surface', 'Roadway Grade', 'Vehicle Type',
                        'Roadway Alignment', 'Vehicle Year', 'Gender', 'Age']

        # create features checkbox
        self.feature0 = QCheckBox(feature_list[0], self)
        self.feature0.setStyleSheet('font-size: 15px;color: black')
        self.feature1 = QCheckBox(feature_list[1], self)
        self.feature1.setStyleSheet('font-size: 15px;color: black')
        self.feature2 = QCheckBox(feature_list[2], self)
        self.feature2.setStyleSheet('font-size: 15px;color: black')
        self.feature3 = QCheckBox(feature_list[3], self)
        self.feature3.setStyleSheet('font-size: 15px;color: black')
        self.feature4 = QCheckBox(feature_list[4], self)
        self.feature4.setStyleSheet('font-size: 15px;color: black')
        self.feature5 = QCheckBox(feature_list[5], self)
        self.feature5.setStyleSheet('font-size: 15px;color: black')
        self.feature6 = QCheckBox(feature_list[6], self)
        self.feature6.setStyleSheet('font-size: 15px;color: black')
        self.feature7 = QCheckBox(feature_list[7], self)
        self.feature7.setStyleSheet('font-size: 15px;color: black')
        self.feature8 = QCheckBox(feature_list[8], self)
        self.feature8.setStyleSheet('font-size: 15px;color: black')

        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        self.feature7.setChecked(True)
        self.feature8.setChecked(True)

        # add checkbox to layout
        self.groupBox1Layout.addWidget(self.feature0)
        self.groupBox1Layout.addWidget(self.feature1)
        self.groupBox1Layout.addWidget(self.feature2)
        self.groupBox1Layout.addWidget(self.feature3)
        self.groupBox1Layout.addWidget(self.feature4)
        self.groupBox1Layout.addWidget(self.feature5)
        self.groupBox1Layout.addWidget(self.feature6)
        self.groupBox1Layout.addWidget(self.feature7)
        self.groupBox1Layout.addWidget(self.feature8)

        # create test split label input
        self.lblPercentTest = QLabel('Test Dataset Split (%):')
        self.lblPercentTest.setStyleSheet('font-size: 15px;font-weight: bold; color: red')
        self.lblPercentTest.adjustSize()
        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setStyleSheet('font-size: 15px;font-weight: bold; color: black; background-color: white')
        # create Tree max Depth
        self.lblMaxDepth = QLabel('Max Depth of the Tree:')
        self.lblMaxDepth.setStyleSheet('font-size: 15px;font-weight: bold; color: red')
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setStyleSheet('font-size: 15px;font-weight: bold; color: black; background-color: white')
        # create Execute button
        self.btnExecute = QPushButton("Run Selected Features")
        self.btnExecute.setStyleSheet('font-size: 15px; color:black; background-color: aquamarine')
        self.btnExecute.clicked.connect(self.update)
        # create view tree button
        self.btnDTFigure = QPushButton("Display the Tree")
        self.btnDTFigure.setStyleSheet('font-size: 15px; color:black;background-color: aquamarine')
        self.btnDTFigure.clicked.connect(self.view_tree)

        # add these labels above
        self.groupBox1Layout.addWidget(self.lblPercentTest)
        self.groupBox1Layout.addWidget(self.txtPercentTest)
        self.groupBox1Layout.addWidget(self.lblMaxDepth)
        self.groupBox1Layout.addWidget(self.txtMaxDepth)
        self.groupBox1Layout.addWidget(self.btnExecute)
        self.groupBox1Layout.addWidget(self.btnDTFigure)

        #:==============================================
        # groupbox2
        self.groupBox2 = QGroupBox('Model Metrics')
        self.groupBox2.setStyleSheet('font-size: 20px;font-weight: bold; color: red')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Classification Report:')
        self.lblResults.setStyleSheet('font-size: 15px;font-weight: bold; color: black')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.txtResults.setStyleSheet('font-size: 20px;font-weight: bold; color: black; background-color: white')
        self.lblAccuracy = QLabel('Accuracy:')
        self.lblAccuracy.setStyleSheet('font-size: 20px;font-weight: bold; color: red')
        self.txtAccuracy = QLineEdit()
        self.txtAccuracy.setStyleSheet('font-size: 20px;font-weight: bold; color: black;background-color: white')

        #=====
       # self.btnPassK = QPushButton('Grab Accuracy')
      #  self.btnPassK.setStyleSheet('font-size: 15px; color:black;background-color: aquamarine')
      #  self.btnPassK.clicked.connect(self.passValue)
        #=====

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)
       # self.groupBox2Layout.addWidget(self.btnPassK)

        #:==========================================
        # Chart 1 : Confusion Matrix
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1.setStyleSheet('font-size: 20px;font-weight: bold; color: red')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas)

        #:==========================================
        # Chart 2 : ROC Curve
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2.setStyleSheet('font-size: 20px;font-weight: bold; color: red')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)
        self.groupBoxG2Layout.addWidget(self.canvas2)

        #:===============================================
        # groupbox layout
        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 1, 0)
        self.layout.addWidget(self.groupBoxG2, 1, 1)

        self.setCentralWidget(self.main_widget)  # Creates the window with all the elements
        self.resize(1200, 1000)
        self.show()

    def update(self):
        self.ax1.clear()
        self.ax2.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        # process the parameters
        self.F_List = pd.DataFrame([])
        feature_list = df_tree.columns

        if self.feature0.isChecked():
            if len(self.F_List) == 0:
                self.F_List = df_tree[feature_list[0]]
            else:
                self.F_List = pd.concat([self.F_List, df_tree[feature_list[0]]], axis=1)

        if self.feature1.isChecked():
            if len(self.F_List) == 0:
                self.F_List = df_tree[feature_list[1]]
            else:
                self.F_List = pd.concat([self.F_List, df_tree[feature_list[1]]], axis=1)

        if self.feature2.isChecked():
            if len(self.F_List) == 0:
                self.F_List = df_tree[feature_list[2]]
            else:
                self.F_List = pd.concat([self.F_List, df_tree[feature_list[2]]], axis=1)

        if self.feature3.isChecked():
            if len(self.F_List) == 0:
                self.F_List = df_tree[feature_list[3]]
            else:
                self.F_List = pd.concat([self.F_List, df_tree[feature_list[3]]], axis=1)

        if self.feature4.isChecked():
            if len(self.F_List) == 0:
                self.F_List = df_tree[feature_list[4]]
            else:
                self.F_List = pd.concat([self.F_List, df_tree[feature_list[4]]], axis=1)

        if self.feature5.isChecked():
            if len(self.F_List) == 0:
                self.F_List = df_tree[feature_list[5]]
            else:
                self.F_List = pd.concat([self.F_List, df_tree[feature_list[5]]], axis=1)

        if self.feature6.isChecked():
            if len(self.F_List) == 0:
                self.F_List = df_tree[feature_list[6]]
            else:
                self.F_List = pd.concat([self.F_List, df_tree[feature_list[6]]], axis=1)

        if self.feature7.isChecked():
            if len(self.F_List) == 0:
                self.F_List = df_tree[feature_list[7]]
            else:
                self.F_List = pd.concat([self.F_List, df_tree[feature_list[7]]], axis=1)

        if self.feature8.isChecked():
            if len(self.F_List) == 0:
                self.F_List = df_tree[feature_list[8]]
            else:
                self.F_List = pd.concat([self.F_List, df_tree[feature_list[8]]], axis=1)

        Xysplit = float(self.txtPercentTest.text()) / 100
        max_depth = float(self.txtMaxDepth.text())

        #==========================
        # Build the model
        #==========================
        X = self.F_List
        y = le.fit_transform(project['ROLL'])

        # split X and y
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8, test_size=Xysplit)
        # perform training with entropy.
        # create decision tree classifier
        self.clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=8, splitter="best", max_depth=max_depth,
                                             min_samples_leaf=5)
        # Performing training
        self.clf_entropy.fit(X_train, y_train)
        # make predictions using entropy
        y_pred_entropy = self.clf_entropy.predict(X_test)

        # confusion matrix for entropy model
        conf_matrix = confusion_matrix(y_test, y_pred_entropy)

        # clasification report
        self.class_rep = classification_report(y_test, y_pred_entropy)
        self.txtResults.appendPlainText(self.class_rep)

        # accuracy score
        self.accuracy_score = accuracy_score(y_test, y_pred_entropy) * 100
        self.txtAccuracy.setText(str(self.accuracy_score))

        #==============================================
        # Chart 1 -- Confusion Matrix
        #==============================================
        class_names = ['Rollover','Non-Rollover']

        self.ax1.matshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        self.ax1.set_yticklabels(['', 'Rollover', 'Non-Rollover'])
        self.ax1.set_xticklabels(['', 'Rollover', 'Non-Rollover'], rotation=90)
        self.ax1.set_xlabel('Predicted Label')
        self.ax1.set_ylabel('True Label')

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        #====================================
        # Chart 2 - ROC Cure
        #====================================
        y_pred_proba = self.clf_entropy.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])

        self.ax2.plot(fpr, tpr, color='lightcoral',lw=5, label='ROC Curve (area = %0.2f)' % auc)
        self.ax2.plot([0, 1], [0, 1], color='turquoise', lw=5, ls='--')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

    def view_tree(self):
        class_names = ['Non-Rollover', 'Rollover']
        #feature = ['weather', 'light', 'surface', 'grade', 'vehicle', 'alignment', 'vehicle_year','gender', 'age']

        dot_data = export_graphviz(self.clf_entropy, filled=True, rounded=True, class_names=class_names,
                                   feature_names=self.F_List.columns, out_file=None)

        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_entropy.pdf")
        webbrowser.open_new(r'decision_tree_entropy.pdf')

  #  def passValue(self):
   #     self.model_compare.txtACC_TREE.setText(str(self.accuracy_score))

#========================================
#========================================
class ModelCompare(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(ModelCompare, self).__init__()
        self.Title = 'Missing Value Summary'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.statusBar().setStyleSheet('background-color: tomato')  # status bar
        self.setStyleSheet('background-color: lavenderblush')  # background color
        self.layout = QVBoxLayout(self.main_widget)

       # self.txtACC_TREE = QLineEdit()

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        # ACC scores
        ACC_Forest = 63.60157386945507
        ACC_TREE = 67.55780736269067
        ACC_Logit = 68.06969471248583
        ACC_KNN = 67.53142574538137

        SCORES = [ACC_Forest, ACC_TREE, ACC_Logit, ACC_KNN]
        SCORES.sort()
        Models = ['Random Forest','KNN','Decision Tree','Logistic Regression']

        self.ax1.barh(Models, SCORES, 0.6, color='turquoise', edgecolor = 'lightcoral', lw=6)

        for i, v in enumerate(SCORES):
            self.ax1.text(v + .1, i - .1, str(round(v,2)) + '%', color='black', fontweight='bold', fontsize='20')

        self.ax1.set_title('Model Accuracy Score Ranking', fontsize= 25, fontweight='bold')
        self.ax1.set_xlabel('Model', fontsize=25, fontweight='bold')
        #self.ax1.set_yticklabels(fontsize=20)
        self.ax1.set_ylabel('Accuracy Score', fontsize=25, fontweight='bold')

        self.fig.canvas.draw_idle()

        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)  # Creates the window with all the elements
        self.resize(1500, 900)  # Resize the window
        self.show()

#========================================
#========================================
# The "About" menu
#========================================
#========================================

class TEAM(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(TEAM, self).__init__()
        self.Title = 'Team 8 Member Introduction'
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)
        self.setWindowTitle(self.Title)
        self.resize(1000, 900)
        self.setStyleSheet('background-color: black')
        self.label1 = QLabel('Jia-Ern Pai')
        self.label1.setStyleSheet('font-size: 70px; font-weight: bold; background-color: lightcyan')
        self.label2 = QLabel('Ethan Litman')
        self.label2.setStyleSheet('font-size: 70px; font-weight: bold; background-color: lightyellow')
        self.label3 = QLabel('Jichong Wu')
        self.label3.setStyleSheet('font-size: 70px; font-weight: bold; background-color: salmon')

        self.label1.setAlignment(QtCore.Qt.AlignTop)
        self.label2.setAlignment(QtCore.Qt.AlignCenter)
        self.label3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignBottom)

        self.layout.addWidget(self.label1)
        self.layout.addWidget(self.label2)
        self.layout.addWidget(self.label3)
        self.setCentralWidget(self.main_widget)

class Jafari(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(Jafari, self).__init__()
        self.Title = 'Professor Amir Jafari'
        self.setWindowTitle(self.Title)
        self.setStyleSheet("background-color: black")
        self.resize(1000, 900)  # Resize the window
        self.statusBar().setStyleSheet("background-color: tomato")  # status bar

        self.AJurl = 'https://media-exp1.licdn.com/dms/image/C4E03AQE87ZLXCHD4LA/profile-displayphoto-shrink_800_800/0/1547647710233?e=1624492800&v=beta&t=HtFLZh8Jqxt_7w_DfYCYdVO_ZGyqt039IywF9yz6UU8'
        self.AJimg = QPixmap(self.AJurl)
        self.data = urllib.request.urlopen(self.AJurl).read()

        self.label = QLabel(self)
        self.AJimg.loadFromData(self.data)
        self.label.setPixmap(self.AJimg)
        self.label.setGeometry(200, 20, 500, 400)
        self.label.move(250,200)

#========================================
#========================================
# EDA menu
#========================================
#========================================

class MissingValue(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(MissingValue, self).__init__()
        self.Title = 'Missing Value Summary'

        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.statusBar().setStyleSheet('background-color: teal')  # status bar
        self.setStyleSheet("background-color: paleturquoise")  # background color
        self.layout = QVBoxLayout(self.main_widget)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        #self.canvas.updateGeometry()

        missing = [0.0,0.0,0.7,1.2,2.3,3.7,3.9,4.8,7.6]
        index = ['Vehicle Type','Year','Roadway Surface','Roadway Alignment','Light Condition',
                 "Driver's Gender",'Model Year','Weather Condition','Roadway Grade']

        self.ax1.barh(index, missing, 0.8, color='red',fill=False,lw=2,hatch='xx')
        # color=['b','g','cornflowerblue','violet','springgreen','yellow','deeppink','lightseagreen','cyan']
        for i,v in enumerate(missing):
            self.ax1.text(v+.1,i-.1,str(v)+'%',color='red',fontweight='bold', fontsize='20')

        self.ax1.set_title('Missing Values Summary (%)')
        self.ax1.set_xlabel('# of Missing Values')
        self.ax1.set_ylabel('Features')

        self.fig.canvas.draw_idle()
        self.layout.addWidget(self.canvas)
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(1500, 900)                        # Resize the window
        self.show()

class TotalCount(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(TotalCount, self).__init__()
        self.Title = 'Rollover Status Overview'
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.statusBar().setStyleSheet('background-color: teal')  # status bar
        self.setStyleSheet('background-color: lightcyan')  # background color
        self.layout = QVBoxLayout(self.main_widget)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        count_rollover = len(project[project['ROLL'] == 'ROLLOVER'])
        count_nonrollover = len(project[project['ROLL'] == 'NON-ROLLOVER'])
        pct_of_rollover = count_rollover / (count_rollover + count_nonrollover)
        pct_of_nonrollover = count_nonrollover / (count_rollover + count_nonrollover)

        self.ax1.bar(['Rollover', 'Non-rollover'], [count_rollover, count_nonrollover], color = ['lightcoral', 'turquoise'])
        self.ax1.set_title("Rollover vs Non-rollover")
        self.ax1.set_xlabel("Rollover")
        self.ax1.set_ylabel("Count")

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        # section 2
        self.lblResults1 = QLabel('% of Rollover: ')
        self.lblResults1.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.txt1 = QLineEdit(self)
        self.txt1.setText(str(pct_of_rollover * 100))
        self.txt1.setStyleSheet('font-size: 20px; font-weight: bold; background-color: whitesmoke')
        self.lblResults2 = QLabel('% of Non-Rollover:')
        self.lblResults2.setStyleSheet('font-size: 20px; font-weight: bold; color: red')
        self.txt2 = QLineEdit(self)
        self.txt2.setText(str(pct_of_nonrollover * 100))
        self.txt2.setStyleSheet('font-size: 20px; font-weight: bold; background-color: whitesmoke')

        self.layout.addWidget(self.lblResults1)
        self.layout.addWidget(self.txt1)
        self.layout.addWidget(self.lblResults2)
        self.layout.addWidget(self.txt2)
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(1000, 800)                        # Resize the window
        self.show()

class AGE(QMainWindow):
    def __init__(self):
        super(AGE, self).__init__()
        self.Title = 'Boxplot: Age vs Rollover'
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.statusBar().setStyleSheet('background-color: teal')  # status bar
        self.setStyleSheet("background-color: lightcyan")  # background color
        self.layout = QVBoxLayout(self.main_widget)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        # boxplot and plot customization
        roll = project.loc[project['ROLL'] == 'ROLLOVER', 'AGE']
        nonroll = project.loc[project['ROLL'] == 'NON-ROLLOVER', 'AGE']

        box = dict(facecolor='turquoise', color='lightcoral', linewidth=3)
        line = dict(color="lightcoral", alpha=0.9, linestyle="dashdot", lw=3)
        outlier = dict(marker="o", markersize=8, markeredgecolor='lightcoral')
        cap = dict(color='lightcoral')

        self.ax1.boxplot([nonroll, roll], whiskerprops=line, widths=0.5, patch_artist=True, boxprops=box,
                    capprops=cap, flierprops=outlier, labels=['Non-Rollover', 'Rollover'])
        self.ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)

        vtitle = 'Boxplot: Age vs Rollover'
        self.ax1.set_title(vtitle)
        self.ax1.set_ylabel("Driver's Age")

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)  # Creates the window with all the elements
        self.resize(800, 800)  # Resize the window
        self.show()

class FeatureVRollover(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(FeatureVRollover, self).__init__()
        self.Title = 'Rollover (Target) vs Features'
        self.setWindowTitle(self.Title)
        self.resize(900, 900)
        self.statusBar().setStyleSheet('background-color: teal')  # status bar
        self.setStyleSheet("background-color: lightcyan")  # background color
        self.main_widget = QWidget(self)
        # chart
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        # dropdown menu
        self.dropdown = QComboBox()
        self.dropdown.addItems(['YEAR','WEATHER_GROUP','LIGHT_Condition','ROAD_SURFACE','ROADWAY_GRADE','ROADWAY_ALIGNMENT','VEHICLE_TYPE','VEHICLE_YEAR','GENDER','AGE_GROUP','AGE'])
        self.dropdown.setStyleSheet('font-size: 15px; background-color: whitesmoke')
        self.dropdown.currentIndexChanged.connect(self.update)
        # age boxplot menu
        self.checkbox = QCheckBox('Age', self)
        self.checkbox.setStyleSheet('font-size: 20px;')
        self.checkbox.stateChanged.connect(self.update)
        # chart 2
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        # add all elements
        self.label1= QLabel('Select Features for Plots')
        self.label1.setStyleSheet('font-size: 20px;font-weight: bold; color: red')
        self.label2 = QLabel('Rollover Frequency')
        self.label2.setStyleSheet('font-size: 20px;font-weight: bold; color: red')
        self.label3 = QLabel('Rollover Rate')
        self.label3.setStyleSheet('font-size: 20px;font-weight: bold; color: red')

        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(self.label1)
        self.layout.addWidget(self.dropdown)
        self.layout.addWidget(self.checkbox)
        self.layout.addWidget(self.label2)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.label3)
        self.layout.addWidget(self.canvas2)

        self.setCentralWidget(self.main_widget)
        self.update()

    def update(self):
        # chart1
        self.ax1.clear()
        feature_picked = self.dropdown.currentText()
        table = pd.crosstab(project[feature_picked], project.ROLL)

        self.ax1.bar(x=table.index, height=table['NON-ROLLOVER'], width=-0.4, align = 'edge', color='turquoise', label='No Rollover')
        self.ax1.bar(x=table.index, height=table['ROLLOVER'], width=0.4, align='edge',color='lightcoral', label='Rollover')

        vtitle = "Freqency Count: Rollover vs " + feature_picked + " 2014-2018"
        self.ax1.set_title(vtitle)
        self.ax1.set_xlabel(feature_picked)
        self.ax1.set_ylabel('Rollover Frequency')
        self.ax1.legend()
        # chart 2
        self.ax2.clear()
        self.ax2.bar(table.index, table['NON-ROLLOVER']/(table['NON-ROLLOVER']+table['ROLLOVER'])*100, color='turquoise', label='No Rollover')
        self.ax2.bar(table.index, table['ROLLOVER']/(table['NON-ROLLOVER']+table['ROLLOVER'])*100, bottom=table['NON-ROLLOVER']/(table['NON-ROLLOVER']+table['ROLLOVER'])*100, color='lightcoral', label='Rollover')

        vtitle = "Percentage: Rollover vs " + feature_picked + " 2014-2018"
        self.ax2.set_title(vtitle)
        self.ax2.set_xlabel(feature_picked)
        self.ax2.set_ylabel('Rollover Rate')
        self.ax2.legend()

        if self.checkbox.isChecked():
            self.checkbox.clicked.connect(self.age)
            self.dialogs = list()

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

    def age(self):
        dialog = AGE()
        self.dialogs.append(dialog)
        dialog.show()

class CarRollover(QMainWindow):
    def __init__(self):
        super(CarRollover, self).__init__()

        self.setWindowTitle('DATS6103 Team8: Prediction of Vehicle Rollover Event')
        self.gwuurl = 'https://www.alexanderjsingleton.com/wp-content/uploads/2016/06/GWU_Logo.png'
        self.gwulogo = QPixmap()
        self.data = urllib.request.urlopen(self.gwuurl).read()
        self.setStyleSheet("background-color: gold")
        self.initUI()

    def initUI(self):
        #self.label = QtWidgets.QLabel(self)
        self.setGeometry(800, 200, 800, 800)
        self.statusBar().showMessage('DATS6103 Group Project | Group 8')
        self.statusBar().setStyleSheet('background-color: mediumturquoise')     # statusBar color

        self.label1 = QLabel('Hello professor o(*.*)o', self)
        self.label1.setStyleSheet('font-size: 50px;')
        self.label1.setGeometry(40, 100, 550,50)
        self.label1.move(130, 50)

        self.label3 = QLabel('Hello class ^_^ !', self)
        self.label3.setStyleSheet('font-size: 50px;')
        self.label3.setGeometry(40, 100, 500, 50)
        self.label3.move(130, 130)

        self.label2 = QLabel(self)
        self.gwulogo.loadFromData(self.data)
        self.label2.setPixmap(self.gwulogo)
        self.label2.setGeometry(100, 150, 500, 400)
        self.label2.move(150, 220)

        # set font size
        self.label1.setFont(QFont('Helvetica [Cronyx]', 8))

        # create the menu bar
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color:tomato')
        aboutMenu = mainMenu.addMenu('About')
        EDAMenu = mainMenu.addMenu('EDA')
        modelMenu = mainMenu.addMenu('Models')
        compareMenu = mainMenu.addMenu('Conclusion')

        # add buttons to 'Main' menu
        ## add team button to 'Main' menu
        teamButton = QAction(QIcon('enter.png'), 'Team8',self)
        teamButton.setStatusTip('This team is awesome!!!')
        teamButton.triggered.connect(self.team)
        aboutMenu.addAction(teamButton)

        ## add AJ button to 'Main' menu
        AJButton = QAction('Prof. Jafari', self)
        AJButton.setStatusTip('This class is so hard!!!')
        AJButton.triggered.connect(self.AJ)
        aboutMenu.addAction(AJButton)

        ## add exit button to 'Main' menu
        exitButton = QAction('Exit', self)
        exitButton.setStatusTip('Warning! Window will be closed!!!')
        exitButton.triggered.connect(self.close)
        aboutMenu.addAction(exitButton)

        # add buttons under 'EDA' menu
        ## nan button
        nanButton = QAction('Missing Value', self)
        nanButton.setStatusTip('How was missing value handled in the data cleaning process')
        nanButton.triggered.connect(self.nan)
        EDAMenu.addAction(nanButton)

        ## rollover count button
        countButton = QAction('Rollover Status Overview', self)
        countButton.setStatusTip('Overal rollover rate and frequency')
        countButton.triggered.connect(self.count)
        EDAMenu.addAction(countButton)

        ## Features vs Rollover button
        fvrButton = QAction('Rollover (Target) vs Features', self)
        fvrButton.setStatusTip('Exam relationship between each of the 9 features and rollover over 2014-2018')
        fvrButton.triggered.connect(self.FvR)
        EDAMenu.addAction(fvrButton)

        # add buttons under "Models" menu
        ## add decision tree button under 'Models' menu
        treeButton = QAction('Decision Tree', self)
        treeButton.setStatusTip('Decision Tree model results of the car rollover dataset')
        treeButton.triggered.connect(self.tree)
        modelMenu.addAction(treeButton)

        ## add random forest button under 'Models' menu
        forestButton = QAction('Random Forest', self)
        forestButton.setStatusTip('Random Forest model results of the car rollover dataset')
        forestButton.triggered.connect(self.Forest)
        modelMenu.addAction(forestButton)

        ## add logistic button under 'Models' menu
        logButton = QAction('Logistic Regression', self)
        logButton.setStatusTip('Logistic Regression model results of the car rollover dataset')
        logButton.triggered.connect(self.Logistic)
        modelMenu.addAction(logButton)

        ## add KNN button under 'Models' menu
        knnButton = QAction('KNN', self)
        knnButton.setStatusTip('KNN model results of the car rollover dataset')
        knnButton.triggered.connect(self.knn)
        modelMenu.addAction(knnButton)

        # add button under 'Model Comparison' menu
        compareButton = QAction('Model Comparison', self)
        compareButton.setStatusTip('Comparing models metrics and results to find the best model')
        compareButton.triggered.connect(self.modelcompare)  # upadate
        compareMenu.addAction(compareButton)

        self.dialogs = list()
        self.show()

    # under "About" menu
    def team(self):
        dialog = TEAM()
        self.dialogs.append(dialog)
        dialog.show()

    def AJ(self):
        dialog = Jafari()
        self.dialogs.append(dialog)
        dialog.show()

    # under "EDA" menu
    def nan(self):
        dialog = MissingValue()
        self.dialogs.append(dialog)
        dialog.show()

    def count(self):
        dialog = TotalCount()
        self.dialogs.append(dialog)
        dialog.show()

    def FvR(self):
        dialog = FeatureVRollover()
        self.dialogs.append(dialog)
        dialog.show()

    # under "Models" menu
    def tree(self):
        dialog = DecisionTree()
        self.dialogs.append(dialog)
        dialog.show()

    def Forest(self):
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()

    def Logistic(self):
        dialog = LogRegression()
        self.dialogs.append(dialog)
        dialog.show()

    def knn(self):
        dialog = KNNmodel()
        self.dialogs.append(dialog)
        dialog.show()

    # under conclusion menu
    def modelcompare(self):
        dialog = ModelCompare()
        self.dialogs.append(dialog)
        dialog.show()

def mainwindow():
    app = QApplication(sys.argv)
    widget = CarRollover()

    widget.show()
    sys.exit(app.exec_())

mainwindow()




