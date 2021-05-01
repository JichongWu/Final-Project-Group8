import sys

#from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
#from PyQt5.QtGui import QPixmap, QIcon   # image
import urllib.request       # image
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
#from PyQt5.QtWidgets import QSizePolicy
#from PyQt5.QtWidgets import QMessageBox
#from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt  # Control status
#from PyQt5.QtWidgets import  QWidget,QLabel, QVBoxLayout, QHBoxLayout, QGridLayout

# Figure packages
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure     # Figure
#from PyQt5.QtGui import QIcon

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

# draw tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

#=========================================================
# load clean data
project = pd.read_csv('https://raw.githubusercontent.com/JichongWu/Final-Project-Group8/main/Data/project_04142021_J.Pai.csv')
#=========================================================

#=========================================================
# Chi-Squared Test code
## Chi-squar test: weather conditions and rollovers
cont_weather_roll= pd.crosstab(project['WEATHER_GROUP'], project['ROLL'])
# Create a heatmap for weather conditions and rollovers
#plt.figure(figsize=(12,8))
#sns.heatmap(cont_weather_roll, annot=True, cmap="YlGnBu")
#plt.show()

## Chi-squar test: light conditions and rollovers
cont_light_roll= pd.crosstab(project['LIGHT_GROUP'], project['ROLL'])
#=========================================================

#=========================================================
# Decision Tree code
feature_list = ['Weather','Light','Surface','Grade','Vehicle Type','Align', 'My_Group','Gender','Age']

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

ziplist=list(zip(weather_encoded, light_encoded, surface_encoded, grade_encoded, vehicle_encoded, align_encoded, model_year_encoded,
          gender_encoded,age_normal))

df_tree=pd.DataFrame(ziplist, columns = ['weather','light','surface','grade','vehicle','align','model_year','gender','age'])

le = preprocessing.LabelEncoder()
y = le.fit_transform(project['ROLL'])
X=df_tree

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, splitter = "best", max_depth=3, min_samples_leaf=5)
# Performing training
clf_entropy.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
# make predictions
# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(X_test)
# display decision tree
class_names = ['No Rollover','Rollover']
feature = X.columns

dot_data = export_graphviz(clf_entropy, filled=True, rounded=True, class_names=class_names, feature_names=feature, out_file=None)

model_list = ['Decision Tree','Random Forest','Logistic Regression','kNN']
#=========================================================
# GUI code
#=========================================================

class ModelCompare(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(ModelCompare, self).__init__()
        self.Title = 'Model Comparison'
        self.setWindowTitle(self.Title)
        self.resize(1200, 800)  # Resize the window
        self.statusBar().setStyleSheet("background-color: darkgreen")  # status bar
        self.setStyleSheet("background-color: lime")

        self.main_widget = QWidget(self)
        # create grid layout
        self.layout = QGridLayout(self.main_widget)
        # create groupbox 1
        self.groupBox1 = QGroupBox('Rollover Models Built by Team 8')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.model0 = QCheckBox(model_list[0],self)
        self.model1 = QCheckBox(model_list[1],self)
        self.model2 = QCheckBox(model_list[2],self)
        self.model3 = QCheckBox(model_list[3],self)

        # add checkbox to layout
        self.groupBox1Layout.addWidget(self.model0,0,0)
        self.groupBox1Layout.addWidget(self.model1,0,1)
        self.groupBox1Layout.addWidget(self.model2,1,0)
        self.groupBox1Layout.addWidget(self.model3,1,1)

        # add test split label input
        self.lblPercentTest = QLabel('Test Dataset Split (%):')
        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setStyleSheet("background-color: white")
        # add Execute button
        self.btnExecute = QPushButton("Run Selected Models")
        self.btnExecute.setStyleSheet("background-color: khaki")
        # add view tree button
        self.btnDTFigure = QPushButton("Display the Accuracy Ranking")
        self.btnDTFigure.setStyleSheet("background-color: khaki")
        #self.btnDTFigure.clicked.connect(self.view_tree)        # update

        # add these labels above
        self.groupBox1Layout.addWidget(self.lblPercentTest,2,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,3,0)
        self.groupBox1Layout.addWidget(self.btnExecute,4,0)
        self.groupBox1Layout.addWidget(self.btnDTFigure,5,0)

        # groupbox layout

        self.layout.addWidget(self.groupBox1, 0, 0)
       # self.layout.addWidget(self.groupBoxG1, 0, 1)
       # self.layout.addWidget(self.groupBox2, 1, 0)
       # self.layout.addWidget(self.groupBoxG2, 1, 1)

        self.setCentralWidget(self.main_widget)  # Creates the window with all the elements
        self.resize(1200, 1000)
        self.show()

class DecisionTree(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree, self).__init__()
        self.Title = 'Decision Tree'
        self.setWindowTitle(self.Title)
        self.resize(1200, 800)  # Resize the window
        self.statusBar().setStyleSheet("background-color: tomato")  # status bar
        self.setStyleSheet("background-color: pink")

        self.main_widget = QWidget(self)
        # create grid layout
        self.layout = QGridLayout(self.main_widget)
        # create groupbox 1
        self.groupBox1 = QGroupBox('Rollover Decision Tree Features')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(feature_list[0], self)
        self.feature1 = QCheckBox(feature_list[1], self)
        self.feature2 = QCheckBox(feature_list[2], self)
        self.feature3 = QCheckBox(feature_list[3], self)
        self.feature4 = QCheckBox(feature_list[4], self)
        self.feature5 = QCheckBox(feature_list[5], self)
        self.feature6 = QCheckBox(feature_list[6], self)
        self.feature7 = QCheckBox(feature_list[7], self)

        # add checkbox to layout
        self.groupBox1Layout.addWidget(self.feature0)
        self.groupBox1Layout.addWidget(self.feature1)
        self.groupBox1Layout.addWidget(self.feature2)
        self.groupBox1Layout.addWidget(self.feature3)
        self.groupBox1Layout.addWidget(self.feature4)
        self.groupBox1Layout.addWidget(self.feature5)
        self.groupBox1Layout.addWidget(self.feature6)
        self.groupBox1Layout.addWidget(self.feature7)

        # add test split label input
        self.lblPercentTest = QLabel('Test Dataset Split (%):')
        self.lblPercentTest.adjustSize()
        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setStyleSheet("background-color: white")
        # add Tree max Depth
        self.lblMaxDepth = QLabel('Max Depth of the Tree:')
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setStyleSheet("background-color: white")
        # add Execute button
        self.btnExecute = QPushButton("Run Selected Features")
        self.btnExecute.setStyleSheet("background-color: aquamarine")
        # add view tree button
        self.btnDTFigure = QPushButton("Display the Tree")
        self.btnDTFigure.setStyleSheet("background-color: aquamarine")
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
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Classification Report:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.txtResults.setStyleSheet("background-color: white")
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()
        self.txtAccuracy.setStyleSheet("background-color: white")

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        #:==========================================
        # Graphic 1 : Confusion Matrix
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas)

        #:==========================================
        # Graphic 2 : ROC Curve
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)
        self.groupBoxG2Layout.addWidget(self.canvas2)

        #:===============================================
        #:===============================================
        #:===============================================

        # groupbox layout

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 1, 0)
        self.layout.addWidget(self.groupBoxG2, 1, 1)

        self.setCentralWidget(self.main_widget)  # Creates the window with all the elements
        self.resize(1200, 1000)
        self.show()


        #self.initUi()
   # def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------
        '''
        self.main_widget = QWidget(self)
        #self.layout = QGridLayout(self.main_widget)

       # self.layout = QVBoxLayout()

        self.groupBox1 = QGroupBox('Rollover Decision Tree Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(feature_list[0], self)
        self.feature1 = QCheckBox(feature_list[1], self)
        self.feature2 = QCheckBox(feature_list[2], self)
        self.feature3 = QCheckBox(feature_list[3], self)
        self.feature4 = QCheckBox(feature_list[4], self)
        self.feature5 = QCheckBox(feature_list[5], self)
        self.feature6 = QCheckBox(feature_list[6], self)
        #self.feature7 = QCheckBox(feature_list[7], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        #self.feature7.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel('Maximun Depth :')
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Run the Data")
       # self.btnExecute.clicked.connect(self.update)

        self.btnDTFigure = QPushButton("Display Tree Plot")
        self.btnDTFigure.clicked.connect(self.view_tree)

        # We create a checkbox for each feature

        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 1, 0)
        self.groupBox1Layout.addWidget(self.feature3, 1, 1)
        self.groupBox1Layout.addWidget(self.feature4, 2, 0)
        self.groupBox1Layout.addWidget(self.feature5, 2, 1)
        self.groupBox1Layout.addWidget(self.feature6, 3, 0)
        #self.groupBox1Layout.addWidget(self.feature7, 3, 1)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 4, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 4, 1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth, 5, 0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth, 5, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 6, 0)
        self.groupBox1Layout.addWidget(self.btnDTFigure, 6, 1)

        self.groupBox2 = QGroupBox('Model Metrics')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()
        self.lblF1 = QLabel('F1-Score:')
        self.txtF1 = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)
        self.groupBox2Layout.addWidget(self.lblF1)
        self.groupBox2Layout.addWidget(self.txtF1)
        self.setCentralWidget(self.main_widget)

        self.show()

        #::-------------------------------------
        # Graphic 1 : Confusion Matrix
        #::------------------------------------
        
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        
        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)
        
        '''
        #::--------------------------------------------
        ## End Graph1
        #::--------------------------------------------

        #::---------------------------------------------
        # Graphic 2 : ROC Curve
        #::---------------------------------------------
        '''
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        ## End of elements of the dashboard

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 0, 2)
        self.layout.addWidget(self.groupBoxG2, 1, 1)
        self.layout.addWidget(self.groupBoxG3, 1, 2)

        self.show()

        #def update(self):
      
        Decision Tree Algorithm
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Decision Tree algorithm
          then the results are presented in graphics and reports in the canvas
        :return: None
        

        # We process the parameters
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[0]]], axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[1]]], axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[2]]], axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[3]]], axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[4]]], axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[5]]], axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[6]]], axis=1)

        if self.feature7.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[7]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[7]]], axis=1)

        vtest_per = float(self.txtPercentTest.text())
        vmax_depth = float(self.txtMaxDepth.text())

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # We assign the values to X and y to run the algorithm

        X_dt = self.list_corr_features
        y_dt = ff_happiness["Happiness.Scale"]

        class_le = LabelEncoder()

        # fit and transform the class

        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100)
        # perform training with entropy.
        # Decision tree with entropy
        self.clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=vmax_depth,
                                                  min_samples_leaf=5)

        # Performing training
        self.clf_entropy.fit(X_train, y_train)

        # predicton on test using entropy
        y_pred_entropy = self.clf_entropy.predict(X_test)

        # confusion matrix for entropy model

        conf_matrix = confusion_matrix(y_test, y_pred_entropy)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred_entropy)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred_entropy) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        #::----------------------------------------------------------------
        # Graph1 -- Confusion Matrix
        #::-----------------------------------------------------------------

        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        class_names1 = ['', 'Happy', 'Med.Happy', 'Low.Happy', 'Not.Happy']

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1, rotation=90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_entropy.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::-----------------------------------------------------
        # End Graph 1 -- Confusioin Matrix
        #::-----------------------------------------------------

        #::-----------------------------------------------------
        # Graph 2 -- ROC Cure
        #::-----------------------------------------------------

        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
        n_classes = y_test_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        lw = 2
        self.ax2.plot(fpr[2], tpr[2], color='darkorange',
                      lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        self.ax2.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')
        self.ax2.set_title('ROC Curve Decision Tree')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        #::--------------------------------
        ### Graph 3 Roc Curve by class
        #::--------------------------------

        str_classes = ['HP', 'MEH', 'LOH', 'NH']
        colors = cycle(['magenta', 'darkorange', 'green', 'blue'])
        for i, color in zip(range(n_classes), colors):
            self.ax3.plot(fpr[i], tpr[i], color=color, lw=lw,
                          label='{0} (area = {1:0.2f})'
                                ''.format(str_classes[i], roc_auc[i]))

        self.ax3.plot([0, 1], [0, 1], 'k--', lw=lw)
        self.ax3.set_xlim([0.0, 1.0])
        self.ax3.set_ylim([0.0, 1.05])
        self.ax3.set_xlabel('False Positive Rate')
        self.ax3.set_ylabel('True Positive Rate')
        self.ax3.set_title('ROC Curve by Class')
        self.ax3.legend(loc="lower right")

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()
'''
    def view_tree(self):
        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_entropy.pdf")
        webbrowser.open_new(r'decision_tree_entropy.pdf')

class Jafari(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(Jafari, self).__init__()
        self.Title = 'Professor Amir Jafari'
        self.setWindowTitle(self.Title)
        self.setStyleSheet("background-color: black")
       # self.layout = QVBoxLayout()
        self.resize(800, 600)  # Resize the window
        self.statusBar().setStyleSheet("background-color: tomato")  # status bar

        self.AJurl = 'https://media-exp1.licdn.com/dms/image/C4E03AQE87ZLXCHD4LA/profile-displayphoto-shrink_800_800/0/1547647710233?e=1624492800&v=beta&t=HtFLZh8Jqxt_7w_DfYCYdVO_ZGyqt039IywF9yz6UU8'
        self.AJimg = QPixmap(self.AJurl)
        self.data = urllib.request.urlopen(self.AJurl).read()

        self.label = QLabel(self)
        self.AJimg.loadFromData(self.data)
        self.label.setPixmap(self.AJimg)
        self.label.setGeometry(200, 20, 500, 400)
        self.label.move(150,80)

class weatherheatmap(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(weatherheatmap, self).__init__()
        self.Title = 'Heatmap: Rollover vs Weather'
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.ax1.imshow(cont_weather_roll, cmap='cool', interpolation='nearest')
        vtitle = "Heatmap: Rollover vs Weather"
        self.ax1.set_title(vtitle)
        self.ax1.set_xlabel("X-axis")
        self.ax1.set_ylabel("Y-axis")

        # sns.heatmap(cont_weather_roll, annot=True, cmap="YlGnBu")

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.layout.addWidget(self.canvas)
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(500, 450)                         # Resize the window
        self.show()

class lightheatmap(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(lightheatmap, self).__init__()
        self.Title = 'Heatmap: Rollover vs Light'
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        self.ax1.imshow(cont_light_roll, cmap='cool', interpolation='nearest')
        vtitle = "Heatmap: Rollover vs Weather"
        self.ax1.set_title(vtitle)
        self.ax1.set_xlabel("X-axis")
        self.ax1.set_ylabel("Y-axis")

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        self.layout.addWidget(self.canvas)

        self.setGeometry(300, 300, 250, 150)

        self.setCentralWidget(self.main_widget)  # Creates the window with all the elements
        self.resize(500, 450)  # Resize the window
        self.show()

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
        self.statusBar().setStyleSheet("background-color: cyan")

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
        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setStatusTip('Warning! Window will be closed!!!')
        exitButton.triggered.connect(self.close)
        aboutMenu.addAction(exitButton)

        # add buttons under 'EDA' menu
        ## weather button
        weatherButton = QAction('Chi-Squared Test: Rollover vs Weather', self)
        weatherButton.setStatusTip('Chi-Squared Test: rollover vs weather')
        weatherButton.triggered.connect(self.weather)
        EDAMenu.addAction(weatherButton)

        ## light button
        lightButton = QAction('Chi-Squared Test: Rollover vs Light', self)
        lightButton.setStatusTip('Chi-Squared Test: rollover vs light')
        lightButton.triggered.connect(self.light)
        EDAMenu.addAction(lightButton)

        ## sns button
        snsButton = QAction('sns test', self)
        snsButton.setStatusTip('heatmap using seaborn')
        snsButton.triggered.connect(self.sns)
        EDAMenu.addAction(snsButton)

        # add buttons under "Models" menu
        ## add decision tree button under 'Models' menu
        treeButton = QAction('Decision Tree', self)
        treeButton.setStatusTip('Decision Tree model of the car rollover dataset')
        treeButton.triggered.connect(self.tree)
        modelMenu.addAction(treeButton)

        ## add random forest button under 'Models' menu
        forestButton = QAction('Random Forest', self)
        forestButton.setStatusTip('Random Forest model of the car rollover dataset')
        forestButton.triggered.connect(self.tree)       #upadate
        modelMenu.addAction(forestButton)

        ## add model comparison button under 'Models' menu
        compareButton = QAction('Model Comparison', self)
        compareButton.setStatusTip('Random Forest model of the car rollover dataset')
        compareButton.triggered.connect(self.modelcompare)      #upadate
        modelMenu.addAction(compareButton)

        self.dialogs = list()

        self.show()

    # under "About" menu
    def team(self):
        QMessageBox.about(self, 'Team Member Introduction', 'Jia-ern Pai' + '\nEthan Litman' + '\nJichong Wu')
        #QMessageBox.setStyleSheet(self, 'QLabel{min-width: 500px;}')

    def AJ(self):
        dialog = Jafari()
        self.dialogs.append(dialog)
        dialog.show()

    # under "EDA" menu
    def weather(self):
        dialog = weatherheatmap()
        self.dialogs.append(dialog)
        dialog.show()
        #plt.figure(figsize=(12, 8))
        #sns.heatmap(cont_weather_roll, annot=True, cmap="YlGnBu")
        #plt.show()

    def light(self):
        dialog = lightheatmap()
        self.dialogs.append(dialog)
        dialog.show()
        #plt.figure(figsize=(12, 8))
        #sns.heatmap(cont_light_roll, annot=True, cmap="YlGnBu")
        #plt.show()

    def sns(self):
        plt.figure(figsize=(12,8))
        sns.heatmap(cont_weather_roll, annot=True, cmap="YlGnBu")
        plt.show()

    # under "Models" menu
    def tree(self):
        dialog = DecisionTree()
        self.dialogs.append(dialog)
        dialog.show()

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

############################################ draft
##############################################

#import matplotlib.pyplot as plt
#import pandas as pd
#project = pd.read_csv('https://raw.githubusercontent.com/JichongWu/Final-Project-Group8/main/Data/project_04142021_J.Pai.csv')

# Chi-squar test: weather conditions and rollovers
#cont_weather_roll= pd.crosstab(project['WEATHER_GROUP'], project['ROLL'])
#plt.imshow(cont_weather_roll, cmap='cool', interpolation='nearest')
#plt.show()


#project['WEATHER_GROUP'], project['ROLL']