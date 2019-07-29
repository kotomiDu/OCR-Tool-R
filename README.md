# OCR-Tool-R
a workbench for OCR




### Installation
1. tensorflow-cpu == 1.5
2. python == 3.6


### Download
1. download [model file](https://drive.google.com/open?id=1wZG5i1cu-Qf_4hn4W5m9m3fKCNYrvVDK)
2. download [OpenVINO lib](https://drive.google.com/open?id=1g5YamnCw5pY5HfvTzFz1Eyk6dnjmNpmT)
3. download [testinput file](https://drive.google.com/open?id=1ZuACWowRZ0PW4Rawi_73c0OWfuzCzk9A）

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

### run video demo
python3 ocr-video-demo/save_key_events.py -i testinput/1080p.mp4 -m /east_icdar2015_resnet_v1_50_rbox/FP32/96_512/frozen_model_temp.xml -d CPU -l openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.so -o output/

