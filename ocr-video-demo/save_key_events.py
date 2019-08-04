# import the necessary packages
from pyimagesearch.keyclipwriter import KeyClipWriter
from DEMOinference import Inference
import argparse
import datetime
import time
import cv2
import logging as log
import sys
import os
from threading import Thread
from queue import Queue
from sys import platform
import numpy as np

from openvino.inference_engine import IENetwork, IEPlugin


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", 
	help="path to input video")
ap.add_argument("-o", "--output", 
	help="path to output directory")
ap.add_argument("-w", "--detect_word", type=str, default="KILL",
	help="codec of output video")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-f", "--fps", type=int, default=60,
	help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
	help="codec of output video")
ap.add_argument("-b", "--buffer-size", type=int, default=240,
	help="buffer size of video clip writer")  #save key frame - buffer size, key frame + buffersize
# model args
#ap.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
ap.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                   default="./east_icdar2015_resnet_v1_50_rbox/FP32/96_512/frozen_model_temp.xml",
                  type=str)

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
if platform == "linux":
    args["cpu_extension"] = "./openvino/inference_engine/lib/intel64/libcpu_extension.so"
elif platform == "win32":
    args["cpu_extenstion"] = "./openvino/deployment_tools/inference_engine/lib/intel64/Release/cpu_extension.dll"

fileName = args["input"]
detect_word = args["detect_word"]

def save_key_events(fileName,detect_word):
    # initialize the video stream and allow the camera sensor to
            # warmup
    print("[INFO] loading video...")
    cap = cv2.VideoCapture(fileName)
    print(cap.isOpened())
    #vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
    time.sleep(2.0)


    # initialize key clip writer and the consecutive number of
    # frames that have *not* contained any action

    weapon_usage = []
    cur_weapon = ""
    kill_num = ""
    kcw = KeyClipWriter(bufSize=args["buffer_size"])
    consecFrames = 0
    frame_idx = 0
    frozen_detect = False
    frozen_frame = 0

    infer = Inference(args["model"],args["device"],args["plugin_dir"],args["cpu_extension"],detect_word)
    infer.load_model()
    def model_inference(kcw,Q):
        print("do inference")
        consecFrames = 0
        frame_idx = 0
        frozen_frame = 0
        frozen_detect = False
        while True: 
           
            if not Q.empty():             
                #frame_idx += 1
                updateConsecFrames = True
                
                for i in range(30):              
                    temp = Q.get()
                    frame_idx += 1
                    if frozen_detect and frozen_frame < 9*30:
                        frozen_frame += 1
                        continue
                    kcw.update(temp)
                
                if frozen_detect and frozen_frame < 9*30:
                    continue
    
                cur_frame = temp.copy()
                #time.sleep(3)
                #cur_frame = cv2.imread("test.png")
                detect_flag = infer.start(cur_frame,frame_idx)
                updateConsecFrames = not detect_flag
                frozen_detect = detect_flag
                
                # only proceed if at least one contour was found
                if detect_flag:
                    #cv2.imwrite("output/"+str(frame_idx)+".png",crop_area[:, :, ::-1])     
                    # reset the number of consecutive frames with
                    # *no* action to zero 
                    #froze next frame 
                    consecFrames = 0
                    frozen_frame = 0
                        # if we are not already recording, start recording
                    if not kcw.recording:
                        print("record")
                        detect_flag = False
                        timestamp = datetime.datetime.now()
                        p = "{}/{}.avi".format(args["output"],
                        timestamp.strftime("sparkletime"))
                        kcw.start(p, cv2.VideoWriter_fourcc(*args["codec"]),
                            args["fps"])
                        if kcw.first_flag:
                            kcw.first_flag = False
                # otherwise, no action has taken place in this frame, so
                # increment the number of consecutive frames that contain
                # no action
                if updateConsecFrames:
                    consecFrames += 1
                
                # if we are recording and reached a threshold on consecutive
                # number of frames with no action, stop recording the clip  
                if kcw.recording and consecFrames == 6:
                    print("finish")
                    kcw.finish()
        
        
    TQ = Queue()
    thread = Thread(target=model_inference,args=(kcw,TQ))
    thread.daemon = True
    thread.start()
    # keep looping
    while True:
        # grab the current frame, crop area, resize it, and initialize a
        # boolean used to indicate if the consecutive frames
        # counter should be updated
        frame_idx += 1
        ret, frame = cap.read() 

        if ret is False:
            break
        # update the key frame clip buffer
        
        TQ.put(frame)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(5) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    ###sparkle time summary
    print(infer.weapon_usage)
    kill_num = ""
    cur_weapon = ""
    frame_idx  = 0
    pre_kill_num = ""
    pre_frame_idx = 0
    cal_usage = {}
    weapon = ["punch","M416","M762","M762","Crossbow","Kar98k","Mk14","Grenade"]
    for item in weapon:
        cal_usage[item] = 0
    for item in infer.weapon_usage:
        kill_num,cur_weapon,frame_idx =  item
        if frame_idx - pre_frame_idx > (30*10 + 30) and kill_num != pre_kill_num:
                cal_usage[cur_weapon] += 1
    pre_kill_num = kill_num
    pre_frame_idx = frame_idx
    print(cal_usage)

    im_h,im_w = [1920,1080]
    cal_im =  np.zeros((im_h,im_w,3), np.uint8)
    i = 0
    cv2.putText(cal_im, "FINAL RESULT", (int(im_w/4),int(im_h/4)), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 2)
    for key,value in cal_usage.items():   
        cv2.putText(cal_im, "{}: {}".format(key,str(value)), (int(im_w/3),int(im_h/3) + i*100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
        i  += 1
    cv2.imwrite("output/cal_im.png",cal_im)
    if kcw is not None:
        for i in range(20):
            print("result")
            kcw.writer.write(cal_im)
    # if we are in the middle of recording a clip, wrap it up
    if kcw.recording:
        kcw.finish()

save_key_events(fileName, detect_word)




