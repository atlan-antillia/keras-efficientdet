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
#
# TrainConfig.py
# 2020/12/17 atlan-antillia

import os
import sys
import glob
import json

import configparser
import traceback


class TrainConfig:
  #
  # class variables  
  PROJECT           = "project"
  NAME              = "name"
  OWNER             = "owner"
  
  MODEL             = "model"
  DATASET           = "dataset"
  DATASET_NAME      = "dataset_name"
  CONFIG            = "config"

  NUM_CLASSES       = "num_classes"
  CLASSES           = "classes"
  
  EFFICIENT_DET     = "efficient_det"
  #Trained models folder
  MODELS            = "models"
  BEST_MODEL        = "best_model"

  
  TRAIN_CONFIG      = "train_config"
  BATCH_SIZE        = "batch_size"
  EPOCHS            = "epochs"
  LEARING_RATE      = "learning_rate"

  EARLY_STOPPING    = "early_stopping"
  EARLY_STOPPING_ENABLED  = "early_stopping_enabled"
  
  PATIENCE          = "patience"
    
  TRAIN             = "train"
  TRAIN_DATA_PATH   = "train_data_path"

  VALID             = "valid"
  VALID_DATA_PATH   = "valid_data_path"
  
  TEST              = "test"
  TEST_DATA_PATH    = "test_data_path"


  #
  # Constructor
  def __init__(self):
     pass
     
     
  # We assume that the format of the annotation file is a coco.json format
  # which may be under the "dataset/{dataset_name}/train, 
  # The file name may be something like "_annotations.coco.json"
  
  def find_annotation_file(self, dir):
    annotation_file = ""
    
    pattern = dir + "/train/*.json"
    print("=== pattern {}".format(pattern))
    
    json_files = glob.glob(pattern)
    print("=== find_annotation_file {}".format(json_files))
    
    if len(json_files) == 1:
      annotation_file = json_files[0]
    else:
      raise Exception("Not found a json annotation file {}".format(json_files))
      
    return annotation_file
    
    #annotation_file will be the name like as "../dataset/{dataset_name}/train/_annotations.coco.json"    #json


  # To get a list of class-names from the json_annotation file.
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

