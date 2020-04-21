from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import sys
import detect_face

cap = cv2.VideoCapture(0)
# Create the haar cascade
#faceCascade = cv2.CascadeClassifier(".\haarcascade_frontalface_alt2.xml") 

frame_interval = 3
c = 0

# Gray Image to RGB
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

# detect net
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor


print('Creating networks and loading parameters')
gpu_memory_fraction=0.6
with tf.Graph().as_default():
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if (c%frame_interval == 0):

        # Our operations on the frame come here
        #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        find_results=[]
        
        #print(gray.ndim)
        if gray.ndim == 2:
            img = to_rgb(gray)

        img = img[:, :, 0:3]
        #sys.exit(0)
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        # faces = faceCascade.detectMultiScale(
        #     gray,
        #     scaleFactor=1.15,
        #     minNeighbors=5,
        #     minSize=(5,5),
        #     flags = cv2.CASCADE_SCALE_IMAGE
        # ) #4
        #for i in range(bounding_boxes[0].size):
        #print()    
        for i in range(len(bounding_boxes)):
            bbox = bounding_boxes[i][:4]
            score = bounding_boxes[i][-1]

            print(bounding_boxes[i])
            #(x, y, w, h) = bounding_boxes[i]
            cv2.rectangle(frame,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255, 0, 255),thickness=2)
            cv2.putText(frame, str(round(score*100, 2)) + "%",(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))

        cv2.imshow('Human Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()