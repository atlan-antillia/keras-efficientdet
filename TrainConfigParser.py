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
# TrainConfigParser.py
#
import os
import sys
import glob
import json
from collections import OrderedDict
import pprint
import configparser 
import traceback

from TrainConfig import TrainConfig


class TrainConfigParser(TrainConfig):

  # Constructor
  # 
  def __init__(self, train_config_path):
    print("==== TrainConfigParser {}".format(train_config_path))
    if not os.path.exists(train_config_path):
      raise Exception("Not found train_config_path {}".format(train_config_path))

    try:
      self.parse(train_config_path)
    except Exception as ex:
      traceback.print_exc()


  def parse(self, train_config_path):
    self.config = configparser.ConfigParser()
    self.config.read(train_config_path)
    self.dump_all()


  def project_name(self):
    return self.config[self.PROJECT][self.NAME]

  def dataset_name(self):
    return self.config[self.MODEL][self.DATASET_NAME]

  def num_classes(self):
    return int( self.config[self.MODEL][self.NUM_CLASSES] )

  def classes(self):
    return self.config[self.MODEL][self.CLASSES]

  def efficient_det(self):
    return int( self.config[self.MODEL][self.EFFICIENT_DET] )

  #A folder to save weight_files
  def models(self):
    return self.config[self.MODEL][self.MODELS]
    
  #A folder to save best model.
  def best_model(self):
    return self.config[self.MODEL][self.BEST_MODEL]

  def batch_size(self):    
    return int( self.config[self.TRAIN_CONFIG][self.BATCH_SIZE] )

  def epochs(self):
    return int( self.config[self.TRAIN_CONFIG][self.EPOCHS] )

  def learning_rate(self):
    return float( self.config[self.TRAIN_CONFIG][self.LEARING_RATE] )

  def earlystopping_patience(self):
    return int( self.config[self.EARLY_STOPPING][self.PATIENCE] )

  def earlystopping_enabled(self):    
    patience     = int( self.config[self.EARLY_STOPPING][self.PATIENCE] )
    MIN_PATIENCE = 5
    MAX_PATIENCE = 20
    rc = True
    if patience >=MIN_PATIENCE and patience <=MAX_PATIENCE:
      rc = True
    else: 
      rc = False
    return rc
    

  def train_data_path(self):
    return self.config[self.TRAIN][self.TRAIN_DATA_PATH]

  def valid_data_path(self):
    return self.config[self.VALID][self.VALID_DATA_PATH]

  def test_data_path(self):
    return self.config[self.TEST][self.TEST_DATA_PATH]
    
     
  def dump_all(self):
    print("==== TrainConfig  dump_all")
    
    print("project_name           {}".format(self.project_name()) )
    print("dataset_name           {}".format(self.dataset_name()) )
    print("num_classes            {}".format(self.num_classes()) )
    print("classes                {}".format(self.classes()) )
    print("models                 {}".format(self.models()) )
    print("best_model             {}".format(self.best_model()) )
    
    print("batch_size             {}".format(self.batch_size()) )
    print("epochs                 {}".format(self.epochs()) )
    print("learning_rate          {}".format(self.learning_rate()) )
    
 
    print("earlystopping_enabled  {}".format(self.earlystopping_enabled()) )
    print("earlystopping_patience {}".format(self.earlystopping_patience()) )

    print("train_data_path        {}".format(self.train_data_path()) )
    print("valid_data_path        {}".format(self.valid_data_path()) )
    print("test_data_path         {}".format(self.test_data_path()) )

##
##
##
if __name__ == "__main__":
  config_file = ""

  try:

    if len(sys.argv) >=2:
       config_file = sys.argv[1]
    if not os.path.exists(config_file):
        raise Exception("Not found {}".format(config_file))

    print("{}".format(config_file))
    train_config = TrainConfigParser(config_file)
    
    train_config.dump_all()
        
  except Exception as ex:
    traceback.print_exc()
    
     