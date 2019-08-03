import sys
import cv2
import numpy as np
from sys import platform
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QGraphicsPixmapItem
from TFinference import TFinference
from OVinference import OVinference
import argparse
from Options import *

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-fp","--float_percision", default="FP16", type=str)
ap.add_argument("-tm", "--tfmodel_path", help="Required. Path for tensorflow trained model.",
                default="./east_icdar2015_resnet_v1_50_rbox/", type=str)
ap.add_argument("-om", "--ovmodel", help="Required. Path to an .xml file with a trained model.",
                 default="./east_icdar2015_resnet_v1_50_rbox/FP32/frozen_model_temp.xml",
                  type=str)

ap.add_argument("-l", "--cpu_extension",
                    help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                        "kernels implementations.", type=str, default="")
ap.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
ap.add_argument("-d", "--device",
                    help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                        "acceptable. The demo will look for a suitable plugin for device specified. "
                        "Default value is CPU", default="CPU", type=str)
args = vars(ap.parse_args())

if platform == "linux":
    args["cpu_extension"] = "./openvino/inference_engine/lib/intel64/libcpu_extension.so"
elif platform == "win32":
    args["cpu_extenstion"] = "./openvino/deployment_tools/inference_engine/lib/intel64/Release/cpu_extension.dll"

class win(QDialog):
    def __init__(self):
        #initial img as an array
        
        super().__init__()
        self.initUI()

    def initUI(self):
        self.img = np.ndarray(())
        self.ofileName = ''
        self.t = 0
        self.resize(400, 300)
        self.btnOpen = QPushButton('Open', self)
        self.btnSave = QPushButton('Save', self)
        self.btnProcess = QPushButton('OCR(TF)', self)
        self.btnProcess2 = QPushButton('OCR(OV)', self)
        self.btnQuit = QPushButton('exit', self)
        self.label = QLabel()

        # layout setting
        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 1, 8, 4)
        layout.addWidget(self.btnOpen, 0, 0, 1, 1)
        layout.addWidget(self.btnSave, 1, 0, 1, 1)
        layout.addWidget(self.btnProcess, 2, 0,1, 1)
        layout.addWidget(self.btnProcess2, 3, 0,1, 1)
        layout.addWidget(self.btnQuit, 4, 0, 1, 1)

        # UI interation with function
        self.btnOpen.clicked.connect(self.openSlot)
        self.btnSave.clicked.connect(self.saveSlot)
        self.btnProcess.clicked.connect(self.processSlot)
        #self.btnProcess.clicked.connect(self.processSlot)
        self.btnQuit.clicked.connect(self.close)

        #model load
      
         
        self.tfinfer = TFinference(args["tfmodel_path"])
        self.tfinfer.load_model()
        
        

    def openSlot(self):
        # open file
        self.ofileName,_ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        if self.ofileName is '':
            return
        # read image
        self.img = cv2.imread(self.ofileName, -1)
        if self.img.size == 1:
            return
        self.refreshShow()

    def saveSlot(self):
        # open file
        fileName, _ = QFileDialog.getSaveFileName(self, 'Save Image', 'Image', '*.png *.jpg *.bmp')
        if fileName is '':
            return
        if self.img.size == 1:
            return
        # save image
        cv2.imwrite(fileName, self.img)

    def processSlot(self):
        if self.img.size == 1:
            return
        self.img = cv2.imread(self.ofileName, -1)
        print(args["ovmodel"],args["device"],args["plugin_dir"],args["cpu_extension"])
        self.img, self.dt, self.rt = self.tfinfer.start(self.img)
        Detection_time = self.dt['net'] + self.dt['restore'] + self.dt['nms']
    
        # Show times
        cv2.putText(self.img, 'Tensorflow', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(self.img, 'Detection Time: '+str('%.4f'%Detection_time)+'s', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(self.img, 'Recognition Time: '+str('%.4f'%self.rt)+'s', (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)

        self.refreshShow()


    def processSlot2(self):
        if self.img.size == 1:
            return
        self.img = cv2.imread(self.ofileName, -1)
        print(args["ovmodel"],args["device"],args["plugin_dir"],args["cpu_extension"])
        self.ovinfer = OVinference(args["ovmodel"],args["device"],args["plugin_dir"],args["cpu_extension"])
        self.ovinfer.load_model()

        self.img, self.dt, self.rt = self.ovinfer.start(self.img)
        Detection_time = self.dt['net'] + self.dt['restore'] + self.dt['nms']

        # Show times
        cv2.putText(self.img, 'OpenVino '+ args["device"] + ' ' + args["fp"], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(self.img, 'Detection Time: '+str('%.4f'%Detection_time)+'s', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(self.img, 'Recognition Time: '+str('%.4f'%self.rt)+'s', (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)

        self.refreshShow()


            

    def refreshShow(self):
        # convert opencv image to  Qimage
        height, width, _ = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        # show Qimage
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
        print("refresh...")


class Options(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.child=Ui_Dialog()
        self.child.setupUi(self)
        self.listenRadio()

    def listenRadio(self):
        self.child.radioButton_3.toggled.connect(self.SetPara1)
        self.child.radioButton_4.toggled.connect(self.SetPara2)
        self.child.radioButton_5.toggled.connect(self.SetPara3)
        self.child.radioButton_6.toggled.connect(self.SetPara4)
        self.child.buttonBox.rejected.connect(self.close)

    def SetPara1(self):
        args["device"] = 'GPU'

    def SetPara2(self):
        args["device"] = 'CPU'

    def SetPara3(self):
        args["ovmodel"] = "./east_icdar2015_resnet_v1_50_rbox/FP16/1080_1920/frozen_model_temp.xml"
        args["fp"] = "FP16"

    def SetPara4(self):
        args["ovmodel"] = "./east_icdar2015_resnet_v1_50_rbox/FP32/frozen_model_temp.xml"
        args["fp"] = "FP32"
    
if __name__ == '__main__':
    a = QApplication(sys.argv)
    w = win()
    o = Options()
    w.btnProcess2.clicked.connect(o.show)
    o.child.buttonBox.accepted.connect(w.processSlot2)
    w.show()  

    sys.exit(a.exec_())
