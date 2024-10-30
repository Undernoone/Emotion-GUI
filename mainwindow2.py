# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow2.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        try:
            MainWindow.setObjectName("Coder729")
            MainWindow.resize(1000, 700)
            MainWindow.setStyleSheet("#MainWindow{background-color: rgb(38,38,40);}")
            MainWindow.setWindowIcon(QtGui.QIcon("D:\\Study_Date\\DeepLearning-Emotion-Classifier-withGUI\\title.jpg"))  # 替换为你的图标路径
            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")

            # 创建字体
            font = QtGui.QFont()
            font.setPointSize(28)

            # 创建按钮
            self.DetectOnBtn = self.create_button(self.centralwidget, "DetectOnBtn", (20, 10, 250, 100), font, "启动识别", "background-color: rgb(0,255,199); border-radius: 10px;")
            self.DetectOffBtn = self.create_button(self.centralwidget, "DetectOffBtn", (280, 10, 250, 100), font, "停止识别", "background-color: rgb(255,40,96); border-radius: 10px;")

            # 创建标签
            self.Cameralabel = self.create_label(self.centralwidget, "Cameralabel", (45, 130, 200, 200)) # 相机
            self.Detectlabel = self.create_label(self.centralwidget, "Detectlabel", (305, 130, 200, 200)) # 识别眼鼻嘴
            self.Face_Label = self.create_label(self.centralwidget, "Face_Label", (45, 350, 200, 200))
            self.EmojiLabel = self.create_label(self.centralwidget, "EmojiLabel", (305, 350, 200, 200))

            # 创建识别结果和状态标签
            # self.shibiejieguo = self.create_text_label(self.centralwidget, "shibiejieguo", (45, 500, 151, 200), "识别结果：", 20)
            # self.emtiontextlabel = self.create_text_label(self.centralwidget, "emtiontextlabel", (45, 550, 151, 200), "Loading........", 20)
            self.shibiejieguo = self.create_text_label(self.centralwidget, "shibiejieguo", (45, 500, 151, 200), "识别结果：", 20)
            self.emtiontextlabel = self.create_text_label(self.centralwidget, "emtiontextlabel", (45, 550, 151, 200), "Loading........",  20)

            self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
            self.gridLayoutWidget.setGeometry(QtCore.QRect(570, 10, 420, 540)) # 柱状图
            self.gridLayoutWidget.setObjectName("gridLayoutWidget")
            self.gridLayoutWidget.setStyleSheet("border: 2px solid black;")
            self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
            self.gridLayout.setContentsMargins(0, 0, 0, 0)
            self.gridLayout.setObjectName("gridLayout")

            MainWindow.setCentralWidget(self.centralwidget)
            self.menuBar = QtWidgets.QMenuBar(MainWindow)
            self.menuBar.setGeometry(QtCore.QRect(0, 0, 936, 24))
            self.menuBar.setObjectName("menuBar")
            MainWindow.setMenuBar(self.menuBar)

            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)

        except Exception as e:
            print(f"设置UI时发生错误: {e}")

    def create_button(self, parent, name, geometry, font, text, style):
        button = QtWidgets.QPushButton(parent)
        button.setGeometry(QtCore.QRect(*geometry))
        button.setFont(font)
        button.setObjectName(name)
        button.setText(text)
        button.setStyleSheet(style)
        return button

    def create_label(self, parent, name, geometry):
        label = QtWidgets.QLabel(parent)
        label.setGeometry(QtCore.QRect(*geometry))
        label.setStyleSheet("border-style: solid;\n"
                            "border-width: 1px;")
        label.setText("")
        label.setObjectName(name)
        return label

    def create_text_label(self, parent, name, geometry, text, font_size):
        font = QtGui.QFont()
        font.setPointSize(font_size)
        label = QtWidgets.QLabel(parent)
        label.setGeometry(QtCore.QRect(*geometry))
        label.setFont(font)
        label.setStyleSheet("color:black")
        label.setText(text)
        label.setObjectName(name)
        return label

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Coder729"))
        # 多语言支持可以放在这里

if __name__ == "__main__":
    import sys
    try:
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"应用程序启动时发生错误: {e}")
