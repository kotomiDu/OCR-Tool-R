from threading import Thread
from queue import Queue
import time
import cv2
import logging as log
import sys
import os
from util import *

from openvino.inference_engine import IENetwork, IEPlugin


class OVinference:
    def __init__(self, model_path, device, plugin_dir, cpu_extension):
        self.model_path = model_path
        self.device = device
        self.plugin_dir = plugin_dir
        self.cpu_extension = cpu_extension
        self.input_blob = None
        self.img_info_input_blob = None
        self.exec_net = None
        self.feed_dict = {}
        self.net = None
        self.n, self.c, self.h, self.w = 0, 0 ,0 ,0
        self.recog_timer = 0 
        self.detect_timer = {'net': 0, 'restore': 0, 'nms': 0} 
    
    def load_model(self):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

        model_xml = self.model_path
        model_bin = os.path.splitext(self.model_path)[0] + ".bin"
        # Plugin initialization for specified device and load extensions library if specified
        log.info("Initializing plugin for {} device...".format(self.device))
        plugin = IEPlugin(device=self.device, plugin_dirs=self.plugin_dir)
        plugin.set_config({"CPU_THREADS_NUM":"1"})
        
        if self.cpu_extension and 'CPU' in self.device:
            plugin.add_cpu_extension(self.cpu_extension)
        # Read IR
        log.info("Reading IR...")

        self.net = IENetwork(model=model_xml, weights=model_bin)

        if plugin.device == "CPU":
            supported_layers = plugin.get_supported_layers(self.net)
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                            format(plugin.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                            "or --cpu_extension command line argument")
                sys.exit(1)

       
        for blob_name in self.net.inputs:
            if len(self.net.inputs[blob_name].shape) == 4:
                self.input_blob = blob_name
            elif len(self.net.inputs[blob_name].shape) == 2:
                self.img_info_input_blob = blob_name
            else:
                raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                                    .format(len(self.net.inputs[blob_name].shape), blob_name))

        log.info("Loading IR to the plugin...")
        self.exec_net = plugin.load(network=self.net, num_requests=2)
        # Read and pre-process input image
        self.n, self.c, self.h, self.w = self.net.inputs[self.input_blob].shape
        if self.img_info_input_blob:
            self.feed_dict[self.img_info_input_blob] = [self.h, self.w, 1]
    
    def start(self,frame):
        self.recog_timer = 0 
        cur_request_id = 0  
        autopad = 0        
        im = frame.copy()       
        im = im[:, :, ::-1] #BGR - RGB
        in_frame = cv2.resize(im, (self.w, self.h))
        
        fh, fw, _ = im.shape
        ratio_h = self.h / float(fh)
        ratio_w = self.w / float(fw)
        

        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))


        inf_start = time.time()
        self.feed_dict[self.input_blob] = in_frame
        self.exec_net.start_async(request_id=cur_request_id, inputs=self.feed_dict)
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            self.detect_timer['net'] = inf_end - inf_start

            # Parse detection results of the current request
            out_blob = iter(self.net.outputs)
            out_blob1 = next(out_blob)
            out_blob2 = next(out_blob)
            res_score = self.exec_net.requests[cur_request_id].outputs[out_blob1]   
            res_geometry = self.exec_net.requests[cur_request_id].outputs[out_blob2]
            score = res_score[0]
            geometry = res_geometry[0]  
            
            boxes, self.detect_timer = self.detect(score_map=score, geo_map=geometry, timer=self.detect_timer)

            if boxes is not None:         
                boxes = boxes[:, :8].reshape((-1, 4, 2))                   
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h
                start = time.time()                       
                for idx, box in enumerate(boxes):
                    box = sort_poly(box.astype(np.int32))
               
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                       continue
                    #fix bug: when text is close to boundary, the box will out of image
                    box[box < 0] = 0                   
                    box[box[:,1] > fh] = fh
                    box[box[:,0] > fw] = fw
                    coor_min = np.min(box, axis=0)
                    coor_max = np.max(box, axis=0)
                
                    number_flag = False
                    box = (coor_min[0]-autopad,coor_min[1]-autopad,coor_max[0]+ autopad,coor_max[1]+autopad)
                    inf_start = time.time()
                    res_text = recog_words(im[box[1]-autopad:box[3]+autopad,box[0]-autopad:box[2]+autopad, ::-1],number_flag)  
                    inf_end = time.time()
                    self.recog_timer += inf_end - inf_start             
       
                    cv2.rectangle(frame[:,:,:], (box[0],box[1]),(box[2],box[3]), (255,255,0), 2)
                    
                    if abs(box[1]-box[3]) < 25:
                        cv2.putText(frame[:, :, :], res_text, (box[0], box[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame[:, :, :], res_text, (box[0], box[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)

        return frame, self.detect_timer, self.recog_timer
    
    def detect(self,score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
        '''
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param timer:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        '''
        if len(score_map.shape) == 3:
            score_map = score_map[0, :, :]
            #chw->hcw
            geo_map = np.transpose(geo_map, (1, 2, 0))
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)

        # sort the te
        # xt boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        start = time.time()
        text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2 #[start:end:step] *4 to map the origin size
        #print('{} text boxes before nms'.format(text_box_restored.shape[0]))
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        timer['restore'] = time.time() - start
        # nms part
        start = time.time()

        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
        timer['nms'] = time.time() - start

        if boxes.shape[0] == 0:
            return None, timer

        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]

        return boxes, timer
    
    
   