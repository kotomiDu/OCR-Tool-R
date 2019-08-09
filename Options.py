# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'models.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        Dialog.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(100, 70, 271, 27))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setEnabled(True)
        self.label.setGeometry(QtCore.QRect(40, 70, 51, 27))
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Choose Model"))
        self.comboBox.setItemText(0, _translate("Dialog", "  "))
        self.comboBox.setItemText(1, _translate("Dialog", "TensorFlow-CPU"))
        self.comboBox.setItemText(2, _translate("Dialog", "OpenVino-CPU-FP32"))
        self.comboBox.setItemText(3, _translate("Dialog", "OpenVino-CPU-INT8"))
        self.comboBox.setItemText(4, _translate("Dialog", "OpenVino-GPU-FP16"))
        self.comboBox.setItemText(5, _translate("Dialog", "OpenVino-GPU-FP32"))
        self.label.setText(_translate("Dialog", "Model:"))

