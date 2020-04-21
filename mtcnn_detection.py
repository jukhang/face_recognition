from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import detect_face
import cv2

class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return detect_face.create_mtcnn(sess, None)
    
    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        
        for i in range(len(bounding_boxes)):
            bbox = bounding_boxes[i][:4]
            score = bounding_boxes[i][-1]

            print(bounding_boxes[i])
            cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255, 0, 255),thickness=2)
            cv2.putText(image, str(round(score*100, 2)) + "%",(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        # for bb in bounding_boxes:
        #     face = Face()
        #     face.container_image = image
        #     face.bounding_box = np.zeros(4, dtype=np.int32)

        #     img_size = np.asarray(image.shape)[0:2]
        #     face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
        #     face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
        #     face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
        #     face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
        #     cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
        #     face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

        #     faces.append(face)

        # return faces