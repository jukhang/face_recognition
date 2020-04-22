from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
from sklearn import metrics 
from sklearn.externals import joblib
import align.detect_face as detect_face
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# mtcnn detect net
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

# mtcnn
gpu_memory_fraction=0.8
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './align/')

# images find human face
# bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

# facenet embdding
#tf.Graph().as_default()
#sess = tf.Session()
#facenet.load_model("/data/wxh/workspace/facenet/models/20180408-102900")

# Get input and output tensors
#images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

# images your dataset
#feed_dict = { images_placeholder: images, phase_train_placeholder:False }
#emb = sess.run(embeddings, feed_dict=feed_dict)

# Dataset

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

from os.path import join as pjoin
def read_img(person_dir,f):
    img=cv2.imread(pjoin(person_dir, f))
                        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                    
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img

data_dir = "/data/wxh/workspace/facenet/src/data"
def load_data(data_dir):
    data = {}
    pics_ctr = 0
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)
        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]
        data[guy] = curr_pics
                                                                                                                             
    return data

data=load_data(data_dir)

print(data.keys())
# load data
keys=[]
for key in data.keys():
    keys.append(key)
    print('foler:{},image numbers：{}'.format(key,len(data[key])))

# knn classfiy dataset
train_x=[]
train_y=[]

#print(data[keys[0]])
print(len(data.items()))


#for x in data[keys[0]]:
#    print(x.shape)
#import sys
#sys.exit(1)
"""
训练每一个data[[0]]
得到一个 emb_data list，一类图像的特征向量列表
"""
with tf.Graph().as_default():
    with tf.Session() as sess:
        facenet.load_model("/data/wxh/workspace/facenet/models/20180408-102900")

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        
        for i in range(len(data.items())):
            for x in data[keys[i]]:
                bounding_boxes, _ = detect_face.detect_face(x, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]    #number of faces

                for face_position in bounding_boxes:
                    face_position=face_position.astype(int)
                    cv2.rectangle(x, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
                    crop=x[face_position[1]:face_position[3],face_position[0]:face_position[2],]
                    
                    crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC )
                    crop_data=crop.reshape(1, 160, 160,3)
                    feed_dict = { images_placeholder: np.array(crop_data), phase_train_placeholder:False }
                    emb_data = sess.run(embeddings, feed_dict=feed_dict)
                    train_x.append(emb_data)
                    train_y.append(i)
 
"""
for x in data[keys[0]]:
    bounding_boxes, _ = detect_face.detect_face(x, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]    #number of faces

    for face_position in bounding_boxes:
        face_position=face_position.astype(int)
        cv2.rectangle(x, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        crop=x[face_position[1]:face_position[3],face_position[0]:face_position[2],]

        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )
        crop_data=crop.reshape(-1,96,96,3)

        feed_dict = { images_placeholder: np.array(crop_data), phase_train_placeholder:False }
        emb_data = sess.run(embeddings, feed_dict=feed_dict)
        
        train_x.append(emb_data)
        train_y.append(0)


for y in data[keys[1]]:
    bounding_boxes, _ = detect_face.detect_face(y, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]#number of faces

    for face_position in bounding_boxes:
        face_position=face_position.astype(int)
        #print(face_position[0:4])
        cv2.rectangle(y, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        crop=y[face_position[1]:face_position[3],
        face_position[0]:face_position[2],]
                                                 
        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )

        crop_data=crop.reshape(-1,96,96,3)
        
        feed_dict = { images_placeholder: np.array(crop_data), phase_train_placeholder:False }
        emb_data = sess.run(embeddings, feed_dict=feed_dict) 
                                                    
        train_x.append(emb_data)
        train_y.append(1)
"""
#train/test split
train_x=np.array(train_x)
#print(train_x.shape)
#exit(1)
train_x=train_x.reshape(35,-1)
train_y=np.array(train_y)
print(train_x.shape)
print(train_y.shape)


X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.3, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
#sys.exit(1)
# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model

classifiers = knn_classifier 
model = classifiers(X_train,y_train)  
predict = model.predict(X_test)  

accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy)  ) 
  
      
#save model
joblib.dump(model, './knn_classifier.model')

# Test model
model = joblib.load('./knn_classifier.model')
predict = model.predict(X_test) 
accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy)  ) 
