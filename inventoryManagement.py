#STEP 1: Import everything
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import cv2



#STEP 2: Setup video feed
cap = cv2.VideoCapture(0) #Set videofeed to webacm

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")





#STEP 3: Model preparation 
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# STEP 4: Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)





#STEP 5: Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


#STEP 6: Detection
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})




    

        #banana = 52, apple = 53, orange = 55

        #boxes[0] = arrays of all the boxes around all objects. ex. people, oranges, dog boxes
        #boxes[0][i] = coordinantes of box at i, where i is iterate of the all the boxes made. Ex. when i = 3 the coordinate at boxes[0][i] would be the coordinantes for the 3 box made in the video
        #boxes[0][i][0] = ymin value. coordinates of the boxes [ymin, xmin, ymax, xmax]
     
      for i,b in enumerate(boxes[0]):       #for i in range of all the boxes made in the video  
          if classes[0][i] == 1 and scores[0][i] > 0.5:            #if the classification for one of the boxes is 1 AKA a person. And if accuracy is above 50%
                  vase_location = boxes[0][i]
                  vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes[0]), np.squeeze(classes[0]).astype(np.int32), np.squeeze(scores[0]), category_index, use_normalized_coordinates=True, line_thickness=8)  #Visualizes boxes of all objects
                  print(classes[0][i],": ",vase_location)     #prints the object being detected and coordinantes of object [ymin, xmin, ymax, xmax] at the i'th box
                  
                  
         

      
    #Makes the letter q the quit button
      cv2.imshow('window',cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break














        
