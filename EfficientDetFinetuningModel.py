# Copyright 2020-2021 antillia.com Toshiyuki Arai
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

# This is a based on Keras-Efficient https://github.com/xuannianz/EfficientDet
 
# EfficientDetFinetuningModel.py
# 
# See also: https://www.kaggle.com/ateplyuk/gwd-starter-efficientdet-keras-train
#          https://www.kaggle.com/savanmorya/efficientdet-keras-train-and-test-offline
#          https://www.kaggle.com/nakajima/xuannianz-efficientdet
#        
import os
import sys
import traceback
import math
import numpy as np
import random
import time
import glob

import tensorflow as tf

from tensorflow.keras.optimizers import Adam
import tensorflow.keras.optimizers

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from augmentor.color import VisualEffect
from augmentor.misc import MiscEffect

from model import efficientdet
from losses import smooth_l1, focal
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

#from generators.coco import CocoGenerator

#pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' -q

from CocoGenerator import CocoGenerator

from EpochChangeCallback import EpochChangeCallback
from TrainingResultWriter import TrainingResultWriter
from TrainConfigParser import TrainConfigParser

#from EvaluationCallback import EvaluationCallback


# EfficientDetFinetuningModel
#

class EfficientDetFinetuningModel:
  #
  # Constructor
  #
  def __init__(self, train_config_path):
    #image_sizes
    #(512, 640, 768, 896, 1024, 1280, 1408)
    
    
    self.TRAIN_CONFIG  = train_config_path
    
    self.score_threshold = 0.4

    self.config = TrainConfigParser(train_config_path)
    self.config.dump_all()

    # The following parameters should be read from a configuration file.
    
    self.PHI           =  self.config.efficient_det()
    
    self.PROJECT_NAME  = self.config.project_name()
    
    self.DATASET_NAME  = self.config.dataset_name()
       
    self.BATCH_SIZE    = self.config.batch_size()
 
    self.NUM_CLASSES   = self.config.num_classes()
    
    self.EPOCHS        = self.config.epochs()

    #self.EPOCHS        = 3  #For quick test.

    self.LEARNING_RATE = self.config.learning_rate()
    

    self.EARLY_STOPPING_ENABLED = self.config.earlystopping_enabled()
    
    self.PATIENCE      = self.config.earlystopping_patience()
    
    self.MODELS_DIR     = self.config.models()
    print("=== models_dir {}".format(self.MODELS_DIR))

    # Extract a train_config filename from the train_config_path 
    #without .config extention part

    BEST_MODEL = "/best_model"
    PROJECTS   = "../projects/"
    
    self.BEST_MODEL_DIR  = PROJECTS + self.PROJECT_NAME + "/" + self.DATASET_NAME + BEST_MODEL
    print("=== BEST_MODEL_DIR {}".format(self.BEST_MODEL_DIR))
    
    config_filename    = train_config_path
    #if '\\' in train_config:
    train_config = train_config_path.replace('\\', '/')
     
    pos = train_config.rfind('/')
    lps = train_config.rfind('.')
    if pos > 0 and pos<lps:
      config_filename = train_config[pos+1:lps]
      
    self.TRAIN_CONFIG_FILENAME = config_filename
    print("=== train_config_filename {}".format(self.TRAIN_CONFIG_FILENAME))
    
    if not os.path.exists(self.MODELS_DIR):
       os.makedirs(self.MODELS_DIR)
       
    # Create a model saving directory
    self.MODEL_SAVE_DIR = self.MODELS_DIR + "/" + self.TRAIN_CONFIG_FILENAME
    print("=== MODEL_SAVE_DIR {}".format(self.MODEL_SAVE_DIR))
    
    if not os.path.exists(self.MODEL_SAVE_DIR):
      os.makedirs(self.MODEL_SAVE_DIR)


    self.TRAIN_DIR     =  self.config.train_data_path()
    self.TRAIN_ANNOTATION = self.find_annotation_file(self.TRAIN_DIR)

    self.VALID_DIR     =  self.config.valid_data_path()
    self.VALID_ANNOTATION = self.find_annotation_file(self.VALID_DIR)
       
    if not os.path.exists(self.TRAIN_DIR):
       raise Exception("Not founc data_dir {}".format(self.TRAIN_DIR))

    if not os.path.exists(self.VALID_DIR):
       raise Exception("Not founc data_dir {}".format(self.VALID_DIR))
       
    print("=== train dir             {}".format(self.TRAIN_DIR))
    print("=== train annotation file {}".format(self.TRAIN_ANNOTATION))

    print("=== valid dir             {}".format(self.VALID_DIR))
    print("=== valid annotation file {}".format(self.VALID_ANNOTATION ))
       
    misc_effect   = MiscEffect()
    visual_effect = VisualEffect()
    

    # Create the CocoGenerators
    # See also: https://github.com/xuannianz/EfficientDet/blob/master/train.py
    self.train_generator = CocoGenerator(data_dir      = self.TRAIN_DIR, 
                                         annotation    = self.TRAIN_ANNOTATION, 
                                         group_method  = 'random',
                                         misc_effect   = misc_effect,
                                         visual_effect = visual_effect,
                                         batch_size    = self.BATCH_SIZE,
                                         phi           = self.PHI)

    self.valid_generator = CocoGenerator(data_dir       = self.VALID_DIR, 
                                         annotation     = self.VALID_ANNOTATION, 
                                         shuffle_groups = False,
                                         batch_size     = self.BATCH_SIZE, 
                                         phi            = self.PHI)
    
    num_classes = self.train_generator.num_classes()
    #num_anchors = self.train_generator.num_anchors
    print("=== num_classes {}".format(num_classes))
    #print("=== num_anchors {}".format(num_anchors))
    #score_threshold = 0.4
    
    self.model, self.prediction_model = efficientdet(self.PHI,
                                       num_classes     = self.NUM_CLASSES,
                                       #num_anchors    = num_anchors,
                                       weighted_bifpn  = True,
                                       freeze_bn       = True,
                                       detect_quadrangle=False,
                                       #score_threshold  =score_threshold,
                                       )
                                       
    model_name = 'efficientnet-b{}'.format(self.PHI)
    file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
    file_hash = WEIGHTS_HASHES[model_name][1]
    weights_path = tf.keras.utils.get_file(file_name,
                                    BASE_WEIGHTS_PATH + file_name,
                                    cache_subdir='models',
                                    file_hash=file_hash)
    print("=== {}".format(weights_path))
    
    self.model.load_weights(weights_path, by_name=True)


    for i in range(1, [227, 329, 329, 374, 464, 566, 656][self.PHI]):
        self.model.layers[i].trainable = False
    
    self.model.compile(optimizer=Adam(lr=self.LEARNING_RATE), 
               metrics=['acc'], 
               loss={'regression': smooth_l1(), 'classification': focal() },
               run_eagerly=True, ) 


  def find_annotation_file(self, dir):
    annotation_file = ""
    
    pattern = dir + "/*.json"
    json_files = glob.glob(pattern)
    #print("=== {}".format(json_files))
    
    if len(json_files) == 1:
      annotation_file = json_files[0]
    return annotation_file


  def train(self):
    print("=== Started a training")
    start_time = time.time()

    train_dataset_size = self.train_generator.size()
    valid_dataset_size   = self.valid_generator.size()
    print("=== train_dataset_size {}".format(train_dataset_size))
    print("=== valid_dataset_size {}".format(valid_dataset_size))

    earlystopping_callback = EarlyStopping(monitor='val_loss', patience=self.PATIENCE, verbose=2, mode='auto')
    # 
    #monitor    = 'mAP',
    #https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/train.py
    #earlystopping_callback = EarlyStopping(monitor='mAP',patience   = self.PATIENCE, mode = 'max', min_delta  = 0.01)
    
    TRAININIG_CSV = "training.csv"
    self.TRAININIG_CSV_PATH = self.MODEL_SAVE_DIR + "/" + TRAININIG_CSV
    print("=== training_csv_path {}".format(self.TRAININIG_CSV_PATH))
    
    epoch_change_callback = EpochChangeCallback(self.TRAININIG_CSV_PATH, ipaddress="127.0.0.1", port=9999, epochs=self.EPOCHS)

    WEIGHT_FILE_NAME     = "weight_d" + str(self.PHI) + ".h5"
    self.WEIGHT_FILEPATH = self.MODEL_SAVE_DIR + "/" +  WEIGHT_FILE_NAME
    
    print("=== WEIGHT_FILEPATH{}".format(self.WEIGHT_FILEPATH))

    #evaluation_callback = EvaluationCallback(self.valid_generator, self.MODELS_DIR, self.prediction_model)
    #print("=== evaluation_callbacks")     
    steps_per_epoch  = train_dataset_size //self.BATCH_SIZE
    validation_steps = valid_dataset_size //self.BATCH_SIZE
    
    history = self.model.fit(
          self.train_generator,
          steps_per_epoch  = steps_per_epoch,
          initial_epoch    = 0,
          epochs           = self.EPOCHS,
          validation_data  = self.valid_generator,
          validation_steps = validation_steps,
          #callbacks       = [evaluation_callback, earlystopping_callback, epoch_change_callback],
          callbacks        = [earlystopping_callback, epoch_change_callback],
          #workers         = workers,
          #use_multiprocessing=.multiprocessing,
          #max_queue_size  = max_queue_size,
          verbose          = 2
    )
    
    end_time = time.time()
    elapsed  = end_time - start_time
    print("=== Elapsed time {}".format(elapsed))

    # Save the weight file
    
    self.model.save_weights(self.WEIGHT_FILEPATH)
    print("=== Saved weights as a file {}".format(self.WEIGHT_FILEPATH))
    

    resultwriter = TrainingResultWriter(self)
    resultwriter.write(elapsed, history)


# Usage python EfficientDetFineTuningModel.py train_config
# Example: python EfficientDetFineTuningModel.py ../projects/demo/BloodCells/config/1*.config
if __name__ == '__main__':
  try:
    if len(sys.argv) < 2:
      raise Exception("Invalid arguments: python EfficientDetFineTuningModel.py train_config")
      
    train_config_path  = ""
    
    if len(sys.argv) >= 2:
      train_config_path = sys.argv[1]
    
    
    if not os.path.exists(train_config_path):
      raise Exception("Not found train_config_path {}".format(train_config_path))
      
          
    model = EfficientDetFinetuningModel(train_config_path)
    model.train()
    
  except:
    traceback.print_exc()
    
