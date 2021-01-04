## Copyright 2020-2021 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# EfficientDetObjectDetector.py
#
# 

import os
import sys
import time
import numpy as np
import glob
import cv2
import tensorflow
import traceback
import tensorflow as tf
import tensorflow
import matplotlib.pyplot as plt
import json

from tensorflow.keras import backend as K
from tensorflow import keras

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Dropout

from model import efficientdet
from utils import preprocess_image, postprocess_boxes

from utils.draw_boxes_with_filters import draw_boxes_with_filters



class EfficientDetObjectDetector:
  # Constructor
  #   
  def __init__(self, dataset_dir, output_dir):
    BEST_MODEL_DIR  = "best_model"
    
    best_model_dir  = os.path.join(dataset_dir, BEST_MODEL_DIR)
    print("=== best_model_dir {}".format(best_model_dir))
    
    phi      = 0
    
    weight_h5 = ""
    for i in [6,5,4,3,2,1,0]:
      weight_h5  = best_model_dir +"/" + "weight_d" + str(i) + ".h5"
      #print("=== weight_h5 {}".format(weight_h5))
      
      if  os.path.exists(weight_h5):
        phi = i
        print("=== EfficientDet phi {}".format(phi))
        print("=== Found weight file {}".format(weight_h5))
        break
        
    if not os.path.exists(weight_h5):
      raise Exception("No found weight file {}".format(weight_h5))
      
    
    
    #self.dataset_dir = dataset_dir
    self.annotation_file = self.find_annotation_file(dataset_dir)
    
    self.output_dir      = output_dir
    print("=== EfficienetDetObjectDetector weight {}".format(weight_h5))
    
    weighted_bifpn = True
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    print("=== phi {}".format(phi))
    
    self.image_size = image_sizes[phi]
    
    print("=== Image size {}".format(self.image_size))
    self.score_threshold = 0.4
    
    print("=== Score threshold {}".format(self.score_threshold))
    
    self.classes = self.get_classes(self.annotation_file)
    print("=== classes {}".format(self.classes))
    
    self.num_classes = len(self.classes)
    print("=== num_classes {}",format(self.num_classes))

    self._, self.model = efficientdet(phi=phi,
                            weighted_bifpn=True, 
                            num_classes=self.num_classes,
                            freeze_bn=True,
                            detect_quadrangle=False,
                            score_threshold=self.score_threshold)
                            
    self.model.load_weights(weight_h5, by_name=True)
    print("=== Loaded weight file {}".format(weight_h5))
    self.colors = [np.random.randint(0, 256, 3).tolist() for _ in range(self.num_classes)]


  def find_annotation_file(self, dir):
    annotation_file = ""
    
    pattern = dir + "/train/*.json"
    json_files = glob.glob(pattern)
    print("=== find_annotation_file {}".format(json_files))
    
    if len(json_files) == 1:
      annotation_file = json_files[0]
    else:
      raise Exception("Not found a json annotation file {}".format(json_files))
      
    return annotation_file


  def get_classes(self, json_file):
    print("=== get_classes")
    classes = []
    
    if json_file is not None:
        with open(json_file,'r') as f:
         js = json.loads(f.read())
         categories =js['categories']
         for values in categories:
           cname = values['name']
           id    = values['id']
           if id>0:
             classes.append(cname)
             
    print("classes {}".format(classes))
    return classes


  def detect_all(self, image_dir, filters=None):
    print("=== detect all {}".format(image_dir))
    if not os.path.exists(image_dir):
      raise Exception("Not found an image dir {}".format(image_dir))
    if not image_dir.endswith(".jpg"):
      image_dir = image_dir + "/*.jpg"
    files = glob.glob(image_dir)
    for image_path in files:
      self.detect(image_path)


  def detect(self, image_path, filters=None):
    print("=== detect {}".format(image_path))
         
    #image_name = image_path.split('/')[-1]
    pos = image_path.rfind('/')
    p   = image_path.rfind('\\')
    if (p> pos):
      pos = p
   
    image_name = image_path[pos+1:]
    print("=== image_name {}".format(image_name))

    image = cv2.imread(image_path)
    #image = Image.open(image_path)
    #image = image.convert("RGB")
    
    #image = np.asarray(Image.open(path).convert('RGB'))
   
    src_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w, _ = image.shape
    
    image = np.array(image)
    image, scale = preprocess_image(image, image_size=self.image_size)
    
    # run network
    start = time.time()
    boxes, scores, labels = self.model.predict_on_batch([np.expand_dims(image, axis=0)])
    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
    elapsed = time.time() - start
    print("=== elapsed time {}".format(elapsed))
    
    boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
    #print("=== boxes{}".format(boxes))
   
    # select indices which have a score above the threshold
    indices = np.where(scores[:] > self.score_threshold)[0]
    #print("=== indices{}".format(indices))

    # select those detections
    boxes  = boxes[indices]
    labels = labels[indices]
    #print("=== boxes{}".format(boxes))
    #print("=== labels {}".format(labels))
    
    detected_objects = []
    objects_stats    = {}
    draw_boxes_with_filters(src_image, boxes, scores, labels, self.colors, self.classes, detected_objects, objects_stats, filters)

    self.SEP = ", "
    self.NL  = "\n"

    self.save_detected_image(image_name, src_image, filters)
    self.save_detected_objects(image_name, detected_objects, filters)
    self.save_objects_stats(image_name, objects_stats, filters)
    

  def save_detected_image(self, image_name, src_image, filters):
    output_image_file = os.path.join(self.output_dir, image_name)
    
    cv2.imwrite(output_image_file, src_image)
    print("=== saved a detected image file {}".format(output_image_file))


  def save_detected_objects(self, image_name, detected_objects, filters):
    detected_objects_csv = os.path.join(self.output_dir, image_name) + "_objects.csv"
    with open(detected_objects_csv, mode='w') as f:
      #2020/09/15 Write a header(title) line of csv.
      header = "id, class, score, x, y, w, h" + self.NL
      f.write(header)

      for item in detected_objects:
        line = str(item).strip("()").replace("'", "") + self.NL
        f.write(line)
   
    print("==== Saved detected_objects {}".format(detected_objects_csv))


  def save_objects_stats(self, image_name, objects_stats, filters):
    objects_stats_csv = os.path.join(self.output_dir, image_name) + "_stats.csv"
    with open(objects_stats_csv, mode='w') as s:
       header = "id, class, count" + self.NL
       s.write(header)
       
       for (k,v) in enumerate(objects_stats.items()):
         (name, value) = v
         line = str(k +1) + self.SEP + str(name) + self.SEP + str(value) + self.NL
         s.write(line)
    print("==== Saved objects_stats {}".format(objects_stats_csv))
  


# python EfficientDetObjectDetector.py  image_file_or_dir dataset_dir 
#
# python EfficientDetObjectDetector.py   ../projects/demo/BloodCells/test ../projects/demo/BloodCells
#
#
if __name__ == "__main__":
  try:
    image_file_or_dir = ""
    dataset_dir         = ""
    output_dir        = ""
    filters           = None
    
    if len(sys.argv) >= 2:
      image_file_or_path = sys.argv[1]
      if not os.path.exists(image_file_or_path):
        raise Exception("Not found image_path {}".format(image_file_or_path))

    if len(sys.argv) >=3:
      dataset_dir = sys.argv[2]
      
      if not os.path.exists(dataset_dir):
        raise Exception("Not found annotation_file_path {}".format(dataset_dir))
    #Default output_dir    
    output_dir = dataset_dir + "/output/"
    
    if len(sys.argv) >=4:
      output_dir = sys.argv[3]
      
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
      if not os.path.exists(output_dir):
        raise Exception("Not found output_dir {}".format(output_dir))
        
    detector = EfficientDetObjectDetector(dataset_dir, output_dir)
 
    if os.path.isdir(image_file_or_path):
      detector.detect_all(image_file_or_path, filters)
    else:
      detector.detect(image_file_or_path, filters)

  except Exception as ex:
    traceback.print_exc()



