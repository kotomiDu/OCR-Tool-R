import sys
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QGraphicsPixmapItem
from TFinference import TFinference
from OVinference import OVinference
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-tm", "--tfmodel_path", help="Required. Path for tensorflow trained model.",
                default="./east_icdar2015_resnet_v1_50_rbox/", type=str)
ap.add_argument("-om", "--ovmodel", help="Required. Path to an .xml file with a trained model.",
                 default="./east_icdar2015_resnet_v1_50_rbox/FP32/frozen_model_temp.xml",
                  type=str)

ap.add_argument("-l", "--cpu_extension",
                    help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                        "kernels implementations.", type=str, default="./openvino/inference_engine/lib/intel64/libcpu_extension.so")
ap.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
ap.add_argument("-d", "--device",
                    help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                        "acceptable. The demo will look for a suitable plugin for device specified. "
                        "Default value is CPU", default="CPU", type=str)
args = vars(ap.parse_args())

class win(QDialog):
    def __init__(self):
        #initial img as an array
        self.img = np.ndarray(())
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 300)
        self.btnOpen = QPushButton('Open', self)
        self.btnSave = QPushButton('Save', self)
        self.btnProcess = QPushButton('OCR', self)
        self.btnQuit = QPushButton('exit', self)
        self.label = QLabel()

        # layout setting
        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 1, 8, 4)
        layout.addWidget(self.btnOpen, 0, 0, 1, 1)
        layout.addWidget(self.btnSave, 1, 0, 1, 1)
        layout.addWidget(self.btnProcess, 2, 0,1, 1)
        layout.addWidget(self.btnQuit, 3, 0, 1, 1)

        # UI interation with function
        self.btnOpen.clicked.connect(self.openSlot)
        self.btnSave.clicked.connect(self.saveSlot)
        self.btnProcess.clicked.connect(self.processSlot)
        self.btnQuit.clicked.connect(self.close)

        #model load
      
        #ovmode_path = 
        self.tfinfer = TFinference(args["tfmodel_path"])
        self.tfinfer.load_model()
        self.ovinfer = OVinference(args["ovmodel"],args["device"],args["plugin_dir"],args["cpu_extension"])
        self.ovinfer.load_model()
        

    def openSlot(self):
        # open file
        fileName,_ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        if fileName is '':
            return
        # read image
        self.img = cv2.imread(fileName, -1)
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
        mode = "tf"
        if mode == "tf":
            self.img= self.tfinfer.start(self.img)
        else:
            if mode == "ov":
                self.img = self.ovinfer.start(self.img)
            
        self.refreshShow()
            

    def refreshShow(self):
        # convert opencv image to  Qimage
        height, width, _ = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        # show Qimage
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
        print("refresh...")

    
if __name__ == '__main__':
    a = QApplication(sys.argv)
    w = win()
    w.show()
    sys.exit(a.exec_())
