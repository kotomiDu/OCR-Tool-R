# OCR-Tool-R
a workbench for OCR




### Installation
1. tensorflow-cpu == 1.5
2. python == 3.6


### Download
1. downlood [model file](https://drive.google.com/open?id=1wZG5i1cu-Qf_4hn4W5m9m3fKCNYrvVDK)
2. download [OpenVINO lib](https://drive.google.com/open?id=1g5YamnCw5pY5HfvTzFz1Eyk6dnjmNpmT)

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

