# OCR-Tool-R
a workbench for OCR




### Installation
1. tensorflow-cpu == 1.5
2. python == 3.6
3. tesseract  == 4.0.0(20190314)or above
3. PyQt  == 5.13.0 or above



### Download
1. download [model file](https://drive.google.com/open?id=1wZG5i1cu-Qf_4hn4W5m9m3fKCNYrvVDK)
2. download OpenVINO lib
   [Linux version](https://drive.google.com/open?id=1g5YamnCw5pY5HfvTzFz1Eyk6dnjmNpmT)
   or [Windows version](https://drive.google.com/open?id=1-xgS_JXjnM-Mf-K6LbZMK79C1XF0SkHT)
3. download [testinput file](https://drive.google.com/open?id=1ZuACWowRZ0PW4Rawi_73c0OWfuzCzk9A)

### set OpenVINO environment
1.edit sg file,change INSTALLDIR to openvino folder (!!! absolute path)
```
cd openvino/bin
vim setupvars.sh
INSTALLDIR=<user_directory>/OCR-Tool-R/openvino
```
2.Open the .bashrc file in <user_directory>: 
```
vi <user_directory>/.bashrc
```
3.Add this line to the end of the file: 
```
source <user_directory>/OCR-Tool-R/openvino/bin/setupvars.sh
```
for windows：the OpenVINO environment can only be activated by the absolute path of the system and then run the project by switching directories under the path.


### set tesseract environment for windows
1.Configuring environment variables,add tesseract to the system path,language packs can be selected during installation（If you don't choose, the default can only parse English）
path = <user_directory>Tesseract-OCR
Language directory = Tesseract-OCR\tessdata
```
2.After the installation is complete, open the command line and type tesseract to check if the installation was successful and type tesseract -v for viewing version.
```
3.install pillow, pytesseract module for python:
Pip install pillow
Pip install pytesseract
```
4.after the installation is complete, modify the pytesseract.py source code:
Find the tesseract_cmd=”tesseract” line and change the following tesseract to tessearct-ocr.exe in the local installation directory, such as: c:\Program Files (x86)\Tesseract-OCR\tesseract.exe.
```

### set PyQt5 environment for windows
1.pip install PyQt5,pip install PyQt5-tools
After the installation is complete, you can see the PyQt5 and pyqt5-tools directories in the Lib\site-packages directory under the Python installation directory.
```
2. Add the installation directory of PyQt5-tools to the system environment variable Path.



### Run video demo
1. linux
```
python3 ocr-video-demo/save_key_events.py -i testinput/1080p.mp4 -m /east_icdar2015_resnet_v1_50_rbox/FP32/96_512/frozen_model_temp.xml -d CPU -l openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.so -o output/
```
2. windows
```
python ocr-video-demo/save_key_events.py -i testinput/1080p.mp4 -m /east_icdar2015_resnet_v1_50_rbox/FP32/96_512/frozen_model_temp.xml -d CPU -l openvino/deployment_tools/inference_engine/lib/intel64/Release/cpu_extension.dll -o output/
```

