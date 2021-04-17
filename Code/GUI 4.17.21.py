import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
import urllib.request
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtWidgets import QSizePolicy

# Figure packages
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure     # Figure
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt  # Control status
from PyQt5.QtWidgets import  QWidget,QLabel, QVBoxLayout, QHBoxLayout, QGridLayout

# models
project = pd.read_csv('https://raw.githubusercontent.com/JichongWu/Final-Project-Group8/main/Data/project_04142021_J.Pai.csv')

# Chi-squar test: weather conditions and rollovers
cont_weather_roll= pd.crosstab(project['WEATHER_GROUP'], project['ROLL'])
# Create a heatmap for weather conditions and rollovers
#plt.figure(figsize=(12,8))
#sns.heatmap(cont_weather_roll, annot=True, cmap="YlGnBu")
#plt.show()

# Chi-squar test: light conditions and rollovers
cont_light_roll= pd.crosstab(project['LIGHT_GROUP'], project['ROLL'])
# Create a heatmap for light conditions and rollovers
#plt.figure(figsize=(12,8))
#sns.heatmap(cont_light_roll, annot=True, cmap="YlGnBu")
#plt.show()

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

        self.fig = Figure(figsize=(15,10))
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

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        self.layout.addWidget(self.canvas)

        self.setGeometry(300, 300, 250, 150)

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

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(500, 450)                         # Resize the window
        self.show()

class CarRollover(QMainWindow):
    def __init__(self):
        super(CarRollover, self).__init__()

        self.setWindowTitle('DATS6103 Team8: Prediction of Vehicle Rollover Event')
        self.gwuurl = 'https://www.alexanderjsingleton.com/wp-content/uploads/2016/06/GWU_Logo.png'
        self.gwulogo = QPixmap()
        self.data = urllib.request.urlopen(self.gwuurl).read()
        self.setStyleSheet("background-color: cyan")
        self.initUI()

    def initUI(self):
        #self.label = QtWidgets.QLabel(self)
        self.setGeometry(800, 200, 800, 800)
        self.statusBar().showMessage('DATS6103 Group Project | Group 8')
        self.statusBar().setStyleSheet("background-color: tomato")

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


        # create main menus
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: greenyellow')
        fileMenu = mainMenu.addMenu('Main')
        heatmapMenu = mainMenu.addMenu('Heatmap')
        modelMenu = mainMenu.addMenu('Models')

        # add exit button to 'Main' menu
        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setStatusTip('Warning! Window will be closed!!!')
        exitButton.triggered.connect(self.close)

        # add team button to 'Main' menu
        teamButton = QAction(QIcon('enter.png'), 'Team8',self)
        teamButton.setStatusTip('This team is awesome!!!')
        teamButton.triggered.connect(self.team)

        fileMenu.addAction(teamButton)
        fileMenu.addAction(exitButton)

        # add buttons under 'Heatmap'
        ## weather button
        weatherButton = QAction('Rollover vs Weather', self)
        weatherButton.setStatusTip('heatmap: rollover vs weather')
        weatherButton.triggered.connect(self.weather)
        heatmapMenu.addAction(weatherButton)

        ## light button
        lightButton = QAction('Rollover vs Light', self)
        lightButton.setStatusTip('heatmap: rollover vs light')
        lightButton.triggered.connect(self.light)
        heatmapMenu.addAction(lightButton)

        self.dialogs = list()

        self.show()

    def team(self):
        QMessageBox.about(self, 'Team Member Introduction', 'Jia-ern Pai' + '\nEthan Litman' + '\nJichong Wu')
        #QMessageBox.setStyleSheet('font-size: 50px;')

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