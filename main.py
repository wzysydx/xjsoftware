import os
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import  matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                 NavigationToolbar2QT as NavigationToolbar)
import matplotlib.style as mplStyle

import math

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel, QVBoxLayout

from pd_to_tv import pandasModel
from ui_main import Ui_MainWindow


class MainForm(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        mplStyle.use("classic")  # 使用样式，必须在绘图之前调用,修改字体后才可显示汉字
        mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei']  # 显示汉字为 楷体， 汉字不支持 粗体，斜体等设置
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['axes.unicode_minus'] = False  # 减号unicode编码


        self.__fig = plt.figure()
        self.figCanvas = FigureCanvas(self.__fig)


        self.piclayout = QVBoxLayout(self)
        self.piclayout.addWidget(self.figCanvas)

        self.ui.horizontalLayout.addWidget(self.piclayout)
        self.ax1 = self.__fig.add_subplot(1, 1, 1)


    @pyqtSlot()
    def on_pb_load_pressed(self):
        path = os.getcwd()
        filename, flt = QFileDialog.getOpenFileName(self, "选择一个测井文件",
                                                    path, "常规测井文件(*.txt);;")
        self.filename = filename

        if (filename == ""):
            return

        # file = open(filename, "r", encoding="UTF-8") # 使用pandas直接读取文件，不需要这一行
        self.ori_data = pd.read_table(filename, sep="\s+", skiprows=6)
        # file.close()

        model_ori = pandasModel(self.ori_data)
        self.ui.tableView.setModel(model_ori)

        self.__buildStatusBar()

    def __buildStatusBar(self):
        self.LabCurFile = QLabel("当前文件：" + self.filename, self)
        self.ui.statusbar.addPermanentWidget(self.LabCurFile)

    @pyqtSlot()
    def on_pb_cal_pressed(self):
        data = self.ori_data[["#DEPTH", "AC", "DEN", "GR"]].rename(columns = {'#DEPTH':'深度', 'AC':'纵波时差', 'DEN':'密度', 'GR':'自然伽马值'})
        data = data[(data["深度"] >= 3450) & (data["深度"] <= 3520)]
        data = data.reset_index(drop=True)

        data["Vp0"] = 1 / data["纵波时差"] * 304800
        data["Vp90"] = (0.7109 * (data["Vp0"] / 1000) + 1.6023) * 1000
        data["Vp45"] = (0.9362 * (data["Vp0"] / 1000) + 0.5516) * 1000
        data["Vs0"] = 1947.8 * np.log(data["Vp0"]) - 13746
        data["Vs90"] = (0.6479 * (data["Vs0"] / 1000) + 1.0489) * 1000

        data["C33"] = (data["密度"] * data["Vp0"] * data["Vp0"]) / 1000000
        data["C44"] = (data["密度"] * data["Vs0"] * data["Vs0"]) / 1000000
        data["C11"] = (data["密度"] * data["Vp90"] * data["Vp90"]) / 1000000
        data["C66"] = (data["密度"] * data["Vs90"] * data["Vs90"]) / 1000000
        data["C12"] = data["C11"] - 2 * data["C66"]
        data["C13"] = -data["C44"] + (
                    4 * data['密度'] ** 2 * data["Vp45"] ** 4 / 10 ** 12 - 2 * data['密度'] * data["Vp45"] ** 2 * (
                        data['C11'] + data['C33'] + 2 * data['C44']) / 1000000 + (data['C11'] + data['C44'])
                    * (data['C33'] + data['C44'])) ** 0.5

        data["Ev"] = data["C33"] - 2 * data["C13"] ** 2 / (data['C11'] + data['C12'])
        data["Eh"] = (data["C11"] - data["C12"]) * (
                    data["C11"] * data["C33"] - 2 * data["C13"] ** 2 + data["C12"] * data["C33"]) / (
                                 data["C11"] * data["C33"] - data["C13"] ** 2)
        data["Vv"] = data["C13"] / (data["C11"] + data["C12"])
        data["Vh"] = (data["C12"] * data["C33"] - data["C13"] ** 2) / (data["C11"] * data["C33"] - data["C13"] ** 2)

        den_avg = data["密度"].mean()  # 密度的平均值

        a, b, kH, kh = 1, 1.18, 0.96, 0.35
        data["Pv"] = data["深度"] * 2.406 * 9.8 / 1000
        data['Pp'] = 1.03 * 9.8 * data['深度'] / 1000 * 1.18
        data['Ph'] = data["C13"] / data["C33"] * (data["Pv"] - data['Pp'] * a) + data['Pp'] + (
                    data["C11"] - data["C13"] ** 2 / data["C33"]) * kh + (
                                 data["C12"] - data["C13"] ** 2 / data["C33"]) * kH
        data['PH'] = data["C13"] / data["C33"] * (data["Pv"] - data['Pp'] * a) + data['Pp'] + (
                    data["C11"] - data["C13"] ** 2 / data["C33"]) * kH + (
                                 data["C12"] - data["C13"] ** 2 / data["C33"]) * kh

        data['Vsh'] = (data['自然伽马值'] - 50.667) / 75.043
        data['Stv'] = (0.0045 * data["Ev"] * (1 - data['Vsh']) + 0.008 * data["Ev"] * data['Vsh']) / 12 * 1000
        data['Sth'] = (0.0045 * data["Eh"] * (1 - data['Vsh']) + 0.008 * data["Eh"] * data['Vsh']) / 12 * 1000

        self.layerdata = data
        model_ori = pandasModel(self.layerdata)
        self.ui.tableView_2.setModel(model_ori)

    @pyqtSlot()
    def on_pb_paint_pressed(self):
        self.ax1.plot(self.layerdata['深度'], self.layerdata['Ev'], 'r-o', label='垂向杨氏模量', linewidth=2, markersize=5)
        # self.ax1.set_xlabel('X轴')
        # self.ax1.set_ylabel('y轴')
        self.ax1.set_title('eqweqwwq')
        self.ax1.legend()

    @pyqtSlot()
    def on_pb_toexcel_pressed(self):





if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = MainForm()
    form.show()
    sys.exit(app.exec_())