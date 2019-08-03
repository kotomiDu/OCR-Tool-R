import tensorflow as tf
import model
import numpy as np
from  util import *
import time
import os


class TFinference:
    def __init__(self,path):
        self.checkpoint_path = path
        self.sess = None
        self.recog_timer = 0 
        self.detect_timer = {'net': 0, 'restore': 0, 'nms': 0} 
    

    def load_model(self):
        global f_geometry,f_score,input_images
    
        #with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        self.sess =  tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        ckpt_state = tf.train.get_checkpoint_state(self.checkpoint_path)
        model_path = os.path.join(self.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(self.sess, model_path)
    
        return self.sess

    def start(self,frame):
        autopad = 0
        self.recog_timer = 0 
        if frame is not None:
            im = frame.copy()
            im = im[:, :, ::-1]
           
            im_resized, (ratio_h, ratio_w) = resize_image(im)
            h,w,_ = frame.shape
    
            
            start = time.time()
            score, geometry = self.sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
            self.detect_timer['net'] = time.time() - start

            boxes, self.detect_timer = self.detect(score_map=score, geo_map=geometry, timer=self.detect_timer)
           

            if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h
                        
                for idx, box in enumerate(boxes):
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                            continue
        
                    box[box < 0] = 0            
                    box[box[:,1] > h] = h
                    box[box[:,0] > w] = w
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
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)
        # sort the text boxes via the y axis
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