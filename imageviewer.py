
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QListView, QWidget, QHBoxLayout,QVBoxLayout, QLineEdit,QPushButton
import ocr 
import cv2




class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()
    
     
        self.layout = QVBoxLayout()

        self.toplayout = QHBoxLayout()
        self.openButton = QPushButton("open file")
        self.filePath = QLineEdit()
        self.filePath.setPlaceholderText("image file")
        self.ocrButton = QPushButton("OCR")
        self.toplayout.addWidget(self.openButton)
        self.toplayout.addWidget(self.filePath)
        self.toplayout.addWidget(self.ocrButton)

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        
        self.imageLabel.setScaledContents (True)
        

        

        #self.scrollArea = QScrollArea()
        #self.scrollArea.setBackgroundRole(QPalette.Dark)
        #self.scrollArea.setWidget(self.imageLabel)
        #self.setCentralWidget(self.scrollArea)

        self.layout.addLayout(self.toplayout)
        self.layout.addWidget(self.imageLabel)
        #self.layout.addWidget(self.scrollArea)
       

        self.openButton.clicked.connect(self.open)
        self.ocrButton.clicked.connect(self.ocr)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setMenuWidget(widget)

  

        self.setWindowTitle("Image Viewer")
        self.width = 1000
        self.height = 700
        self.resize(self.width, self.height)
        self.sess = ocr.runtesorflow()

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            
            self.filePath.setText(fileName)

            pix = QPixmap.fromImage(image)
            pix = pix.scaled(self.width,self.height - 10)           
            self.imageLabel.setPixmap(pix)
         
            
    def ocr(self):
        imgname =  self.filePath.text()
        ocrimg = ocr.recognition(imgname,self.sess)
        ocrimg = cv2.resize(ocrimg, (self.width,self.height - 10), interpolation=cv2.INTER_CUBIC)
        height, width, channel = ocrimg.shape
        cv2.cvtColor(ocrimg, cv2.COLOR_BGR2RGB, ocrimg)
        QImg = QImage(ocrimg.data, width, height, width*3, QImage.Format_RGB888)
        ocrpix = QPixmap.fromImage(QImg)
        #ocrpix = ocrpix.scaled(self.width,self.height - 10)           
        self.imageLabel.setPixmap(ocrpix)
        







if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
 
    imageViewer = ImageViewer()
    imageViewer.show()
sys.exit(app.exec_())