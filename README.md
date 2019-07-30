# OCR-Tool-R
a workbench for OCR




### Installation
1. tensorflow-cpu == 1.5
2. python == 3.6


### Download
1. download [model file](https://drive.google.com/open?id=1wZG5i1cu-Qf_4hn4W5m9m3fKCNYrvVDK)
2. download OpenVINO lib
   [Linux version](https://drive.google.com/open?id=1g5YamnCw5pY5HfvTzFz1Eyk6dnjmNpmT)
   or [Windows version](https://drive.google.com/open?id=1-xgS_JXjnM-Mf-K6LbZMK79C1XF0SkHT)
3. download [testinput file](https://drive.google.com/open?id=1ZuACWowRZ0PW4Rawi_73c0OWfuzCzk9A)

### set OpenVINO environment
**Linux
- edit sh file,change INSTALLDIR to openvino folder (!!! absolute path)**
```
cd openvino/bin
vim setupvars.sh
INSTALLDIR=<user_directory>/openvino
```
- option1. temporarily set your environment variables**
```
source <user_directory>/openvino/bin/setupvars.sh
```
- option2. permanently set your environment variables**

1. Open the .bashrc file: 
```
vi .bashrc
```
2. Add this line to the end of the file: 
```
source <user_directory>/openvino/bin/setupvars.sh
```
3. Save and close the file: press the Esc key and type :wq
4. To test your change, open a new terminal. You will see [setupvars.sh] OpenVINO environment initialized.

more guide in [OpenVINO](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)
**Windows
### Run video demo
**Linux
```
python3 ocr-video-demo/save_key_events.py -i testinput/1080p.mp4 -m ./east_icdar2015_resnet_v1_50_rbox/FP32/96_512/frozen_model_temp.xml -d CPU -l ./openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.so -o output/
```
**Windows
```
python3 ocr-video-demo/save_key_events.py -i testinput/1080p.mp4 -m ./east_icdar2015_resnet_v1_50_rbox/FP32/96_512/frozen_model_temp.xml -d CPU -l ./openvino/deployment_tools/inference_engine/lib/intel64/Release/cpu_extension.dll -o output/
```

