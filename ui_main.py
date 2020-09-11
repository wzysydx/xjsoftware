# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1293, 886)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName("groupBox")
        self.tableView_2 = QtWidgets.QTableView(self.groupBox)
        self.tableView_2.setGeometry(QtCore.QRect(40, 40, 541, 351))
        self.tableView_2.setObjectName("tableView_2")
        self.horizontalLayout.addWidget(self.groupBox)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.groupBox_4.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_4.setObjectName("groupBox_4")
        self.tableView = QtWidgets.QTableView(self.groupBox_4)
        self.tableView.setGeometry(QtCore.QRect(50, 40, 551, 301))
        self.tableView.setObjectName("tableView")
        self.horizontalLayout_2.addWidget(self.groupBox_4)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_3.setObjectName("groupBox_3")
        self.pb_load = QtWidgets.QPushButton(self.groupBox_3)
        self.pb_load.setGeometry(QtCore.QRect(320, 110, 121, 41))
        self.pb_load.setObjectName("pb_load")
        self.pb_cal = QtWidgets.QPushButton(self.groupBox_3)
        self.pb_cal.setGeometry(QtCore.QRect(320, 160, 121, 41))
        self.pb_cal.setObjectName("pb_cal")
        self.pb_paint = QtWidgets.QPushButton(self.groupBox_3)
        self.pb_paint.setGeometry(QtCore.QRect(320, 210, 121, 41))
        self.pb_paint.setObjectName("pb_paint")
        self.pb_toexcel = QtWidgets.QPushButton(self.groupBox_3)
        self.pb_toexcel.setGeometry(QtCore.QRect(320, 260, 121, 41))
        self.pb_toexcel.setObjectName("pb_toexcel")
        self.pb_topic = QtWidgets.QPushButton(self.groupBox_3)
        self.pb_topic.setGeometry(QtCore.QRect(320, 310, 121, 41))
        self.pb_topic.setObjectName("pb_topic")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setGeometry(QtCore.QRect(30, 40, 54, 12))
        self.label.setObjectName("label")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.doubleSpinBox.setGeometry(QtCore.QRect(70, 30, 62, 22))
        self.doubleSpinBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox.setMaximum(9999.0)
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.label_2.setGeometry(QtCore.QRect(30, 70, 54, 12))
        self.label_2.setObjectName("label_2")
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.doubleSpinBox_2.setGeometry(QtCore.QRect(70, 70, 62, 22))
        self.doubleSpinBox_2.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_2.setMaximum(9999.0)
        self.doubleSpinBox_2.setSingleStep(0.01)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.horizontalLayout_2.addWidget(self.groupBox_3)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1293, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "页岩油储层岩石参数转化软件"))
        self.groupBox.setTitle(_translate("MainWindow", "储层参数计算结果与曲线"))
        self.groupBox_4.setTitle(_translate("MainWindow", "测井资料"))
        self.groupBox_3.setTitle(_translate("MainWindow", "井场参数输入"))
        self.pb_load.setText(_translate("MainWindow", "导入测井资料"))
        self.pb_cal.setText(_translate("MainWindow", "计算岩石力学参数"))
        self.pb_paint.setText(_translate("MainWindow", "绘制力学参数曲线"))
        self.pb_toexcel.setText(_translate("MainWindow", "导出数据到excel文件"))
        self.pb_topic.setText(_translate("MainWindow", "导出图片"))
        self.label.setText(_translate("MainWindow", "顶深："))
        self.label_2.setText(_translate("MainWindow", "底深："))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
