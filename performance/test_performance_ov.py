#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function
import sys
import os
import argparse 
import cv2
import time
import logging as log
import numpy as np
import lanms
import pytesseract
import multiprocessing
from functools import partial
from util import *
from OVinference import OVinference

from openvino.inference_engine import IENetwork, IEPlugin


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                    required=True, type=str)

ap.add_argument("-l", "--cpu_extension",
                    help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                        "kernels implementations.", type=str, default=None)
ap.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
ap.add_argument("-d", "--device",
                    help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                        "acceptable. The demo will look for a suitable plugin for device specified. "
                        "Default value is CPU", default="CPU", type=str)
ap.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
ap.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                    default=0.5, type=float)
args = vars(ap.parse_args())


def main():
    cap_time = 0 
    detect_infer_time = 0
    detect_time = 0 
    recog_time = 0
    frame_idx = 0
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    # initialize the video stream and allow the camera sensor to
    # warmup
    print("[INFO] loading video...")
    cap = cv2.VideoCapture(args["input"])
    
    infer = OVinference(args["model"],args["device"],args["plugin_dir"],args["cpu_extension"])
    load_start = time.time()
    infer.load_model()
    load_end = time.time()
    load_time = load_end - load_start

    total_start = time.time()
    while cap.isOpened():
        
        cap_start = time.time()
        ret, frame = cap.read()
        cap_end = time.time()
        cap_time += cap_end - cap_start
        if not ret:
            break
  
        frame_idx += 1
        infer.start(frame)
        detect_infer_time += infer.detect_timer['net']
        detect_time += infer.detect_timer['net'] + infer.detect_timer['restore'] + infer.detect_timer['nms']
        recog_time += infer.recog_timer

        if frame_idx > 10:
            break
    total_end = time.time()
    totaltime = total_end - total_start 
    print("load time:{:.1f}s".format(load_time))
   
    print("total time:{:.1f}s,  fps:{:.3f} fps".format(totaltime, frame_idx/totaltime))
    print("cap_time:{:.1f}s".format(cap_time))
    print("detect time:{:.1f}s,  fps:{:.3f} fps; detect infer time:{:.1f}s,  fps:{:.3f} fps".format(
        detect_time, frame_idx/detect_time, detect_infer_time, frame_idx/detect_infer_time))
    print("recog time:{:.1f}s,  fps:{:.3f} fps".format(recog_time, 0 if recog_time  == 0 else frame_idx/recog_time))

    


if __name__ == '__main__':
    sys.exit(main() or 0)
