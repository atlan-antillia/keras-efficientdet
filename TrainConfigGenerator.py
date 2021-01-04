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
# TrainConfigGenerator.py

import os
import sys
import glob
import json

import configparser
import traceback
from TrainConfig import TrainConfig

class TrainConfigGenerator(TrainConfig):
  #
  # Constructor
  def __init__(self, config_template, batch_size, efficient_det=0):
    
    self.config_template   = config_template
    self.batch_size        = batch_size
    self.efficient_det     = efficient_det
   
    #self.learning_rates   = [0.001, 0.0001]
    self.learning_rates    = [0.0001]
    self.epochs            = 300
    #self.patiences        = [10, 15]
    self.patiences         = [15]
    self.config = configparser.ConfigParser()
    self.config.read(self.config_template)


  def generate(self, project_name, dataset_name):
    #Fixed folder names
    PROJECTS_DIR   = "../projects"  # Fixed projects folder name
    CONFIG_DIR     = "/config"       # Fixed folder name to save config files  
    MODELS_DIR     = "/models"       # Fixed folder name to save trained weights files.
    BEST_MODEL_DIR = "/best_model"   # Fixed folder name to save the best model.
    self.project_name   = project_name
        
    
    #self.dataset  =  str(self.config[self.MODEL][self.DATASET]) 
    dataset_dir = PROJECTS_DIR + "/" + self.project_name + "/" + dataset_name
   
    print("=== dataset_dir {}".format(dataset_dir))
    
    # Test the dataset_dir is already existing.
    if not os.path.exists(dataset_dir):
       raise Exception("Not found dataset_dir {}".format(dataset_dir))
       
    print("== dataset_dir {}".format(dataset_dir))
    
    # Get an annotation file of the format coco.json.
    annofile = self.find_annotation_file(dataset_dir)
    
    print("=== annotation file {}".format(annofile))
    
    self.classes     = self.get_classes(annofile)
    self.num_classes = len(self.classes)
    print("=== num classes {}".format(self.num_classes))
    
    output_config_dir = dataset_dir +  CONFIG_DIR
    
    print("=== output_config_dir {}".format(output_config_dir))

    output_trained_models_dir    = dataset_dir + MODELS_DIR
                              
    output_trained_best_model_dir= dataset_dir + BEST_MODEL_DIR
                             
    print("=== output_trained_models_dir    {}".format(output_trained_models_dir))
    print("=== output_trained_best_model_dir{}".format(output_trained_best_model_dir))
    
    if not os.path.exists(output_config_dir):
       os.makedirs(output_config_dir)
    if not os.path.exists(output_config_dir):
       raise Exception("Not found output_config_dir {}".format(output_config_dir))
    
    index = 1
    for lr in self.learning_rates :
      for pa in self.patiences :
        filename = "{}_B_{}_L_{}_P_{}_E_{}_D_{}.config".format(index, self.batch_size, lr,pa, self.epochs, self.efficient_det )
        print("=== filename {}".format(filename))
        index += 1
        config_file_path = output_config_dir + "/" + filename
        self.config[self.PROJECT][self.NAME]              = str(project_name)  
        self.config[self.MODEL][self.NUM_CLASSES]         = str(self.num_classes)
        self.config[self.MODEL][self.DATASET_NAME]        = str(dataset_name)

        self.config[self.MODEL][self.EFFICIENT_DET]       = str(self.efficient_det)
        self.config[self.MODEL][self.CLASSES]             = str(self.classes)

        self.config[self.MODEL][self.MODELS]      = output_trained_models_dir
        self.config[self.MODEL][self.BEST_MODEL]  = output_trained_best_model_dir
 
        self.config[self.TRAIN_CONFIG][self.BATCH_SIZE]   = str(self.batch_size)
        self.config[self.TRAIN_CONFIG][self.LEARING_RATE] = str(lr)
        self.config[self.TRAIN_CONFIG][self.EPOCHS]       = str(self.epochs)

        self.config[self.EARLY_STOPPING][self.PATIENCE] = str(pa)

        #The following fixed folder-names, each folder contains a set of images and a coco.json annotation file.
       
        train_data_path = dataset_dir + "/" +  self.TRAIN
        if not os.path.exists(train_data_path):
          raise Exception("Not found {}".format(train_data_path))
          
        self.config[self.TRAIN][self.TRAIN_DATA_PATH]= train_data_path
        
        valid_data_path = dataset_dir + "/" + self.VALID
        if not os.path.exists(valid_data_path):
          raise Exception("Not found {}".format(valid_data_path))
        
        self.config[self.VALID][self.VALID_DATA_PATH]= valid_data_path

        test_data_path = dataset_dir + "/" + self.TEST
        if not os.path.exists(test_data_path):
          raise Exception("Not found {}".format(test_data_path))
        
        self.config[self.TEST][self.TEST_DATA_PATH]= test_data_path

        with open(config_file_path, "w") as f:
           self.config.write(f)          
           print("=== Saved config file {}".format(config_file_path))
           

# Run the following command in the keras-efficientdet-object-detection/EfficientDet 
#
# python TrainConfigGenerator.py projects_dir project_name dataset_name [batch_size] [efficient_det]
# project_name : Your own project_name
# dataset_name : Your dataset_name
# optional batch_size : default value = 8
# optional efficient_det: default value = 0
#
# Example Usage:
# python TrainConfigGenerator.py myproject BloodCells

if __name__=="__main__":

  template_config = "./train_template.config"
  try:

     project_name     = ""
     dataset_name     = ""
     batch_size       = 8  #default batch size
     efficient_det    = 0  #default efficient_det model id

     if len(sys.argv) <3:
       raise Exception("Invalid arguments: python TrainConfigGenerator.py project_name dataset_name [batch_size] [efficient_det] ")

     if len(sys.argv) >= 2:
        project_name = str(sys.argv[1])

     if len(sys.argv) >= 3:
        dataset_name = str(sys.argv[2])
         
     if len(sys.argv) >= 4:
        batch_size = int(sys.argv[3])
        print("============= batch_size {}".format(batch_size))
        
     if batch_size <8 or batch_size >48:
        org_batch  = batch_size
        batch_size = 8
        print("=== Warning: resized batch_size from {} to {}".format(org_batch, batch_size))

     if len(sys.argv) >= 5:
        efficient_det = int(sys.argv[4])
        print("============= efficient_det {}".format(efficient_det))

     #         0,    1,   2,   3,    4,    5,    6     
     sizes = [512, 640, 768, 896, 1024, 1280, 1408]
       
     if efficient_det <0 or efficient_det >len(sizes):
        org_det = efficient_det
        efficient_det = 0
        print("=== Warning: resized efficienit_det from {} to {}".format(org_det, efficient_det))
     print("=== project_name {} dataset_name {}".format(project_name, dataset_name))
     
     generator = TrainConfigGenerator(template_config, batch_size, efficient_det)
     generator.generate(project_name, dataset_name)

  except:
   traceback.print_exc()
