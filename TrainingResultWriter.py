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
# TrainingResultWriter.py
#

import os
import sys
import traceback
import time
import csv
import shutil

from METRICS import *

##
##
class TrainingResultWriter:

  def __init__(self, model, stdout=True):
    self.model     = model
    self.stdout    = stdout
    

  def elapsed_time(self, elapsed):    
    s = elapsed
    hours = s // 3600 
    # remaining seconds
    s = s - (hours * 3600)
    # minutes
    minutes = s // 60
    # remaining seconds
    seconds = s - (minutes * 60)
    # total time
    elapced_hh_mm_ss = '{}h:{:02}m:{:02}s'.format(int(hours), int(minutes), int(seconds))
    return  elapced_hh_mm_ss

  
  def write(self, elapsed, history):
    #print("=== write_training_result history {}".format(history.history))
    history = history.history
    
    loss         = history[LOSS]
    val_loss     = history[VAL_LOSS]
    
    last_epoch   = len(loss)
    epoch        = last_epoch -1

    print("=== last_epoch {}".format(last_epoch))
    
    last_loss     = loss    [epoch]
    last_val_loss = val_loss[epoch]

    classification_acc     = 0.0
    val_classification_acc = 0.0
    
    regression_acc         = 0.0
    val_regression_acc     = 0.0

    classification_accs    = []

    if CLASSIFICATION_ACC in history:
      classification_accs     = history[CLASSIFICATION_ACC]

    if FT_CLASSIFICATION_ACC in history:
      classification_accs     = history[FT_CLASSIFICATION_ACC]
    
    classification_acc     =  classification_accs[epoch]

    val_classification_accs = []
    
    if VAL_CLASSIFICATION_ACC in history:
      val_classification_accs     = history[VAL_CLASSIFICATION_ACC]

    if VAL_FT_CLASSIFICATION_ACC in history:
      val_classification_accs     = history[VAL_FT_CLASSIFICATION_ACC]   
    
    val_classification_acc = val_classification_accs[epoch]

    regression_accs         = []
    if REGRESSION_ACC in history:
      regression_accs         = history[REGRESSION_ACC]
    regression_acc          = regression_accs[epoch]
    
    val_regression_accs     = []
    if VAL_REGRESSION_ACC in history:
      val_regression_accs   = history[VAL_REGRESSION_ACC]
    
    val_regression_acc      = val_regression_accs[epoch]
     
           
    print("=== last_loss:{},last_val_loss:{} ".format(last_loss, last_val_loss))

    TRAINING_RESULT_CSV      = "training_result.csv"
    self.TRAINING_RESULT_CSV_PATH = self.model.MODEL_SAVE_DIR + "/" + TRAINING_RESULT_CSV
    print("=== TRAINING_RESULT_CSV_PATH {}".format(self.TRAINING_RESULT_CSV_PATH))
    
    elapced_hh_mm_ss = self.elapsed_time(elapsed)    
    
    if os.path.exists(self.TRAINING_RESULT_CSV_PATH):
      try:
        os.remove(self.TRAINING_RESULT_CSV_PATH)
      except:
        traceback_print_exc()
        
    NL = "\n"
    print("=== last_loss:{}, last_val_loss:{} ".format(last_loss, last_val_loss))
    
    with open(self.TRAINING_RESULT_CSV_PATH, "w") as f:
        line = "name,   value" + NL
        f.write(line)
        line = "elapsed, {}".format(elapced_hh_mm_ss) + NL
        f.write(line)
 
        line = "last_epoch, " + str(last_epoch) + NL
        f.write(line)
        

        line = "last_val_loss, {:.4f}".format(last_val_loss) + NL
        f.write(line)
        
        line = "last_loss, {:.4f}".format(last_loss) + NL
        f.write(line)
        
        line = "classification_acc, {:.4f}".format(classification_acc) + NL
        f.write(line)
        
        line = "val_classification_acc, {:.4f}".format(val_classification_acc) + NL
        f.write(line)
        
        line = "regression_acc, {:.4f}".format(regression_acc) + NL
        f.write(line)
        
        line = "val_regression_acc, {:.4f}".format(val_regression_acc) + NL
        f.write(line)
    
        
        line = "batch_size, {}".format(self.model.BATCH_SIZE) + NL
        f.write(line)

        line = "learning_rate, {:.4f}".format(self.model.LEARNING_RATE) + NL
        f.write(line)
        
        line = "patience, {}".format(self.model.PATIENCE) + NL
        f.write(line)
        
        line = "epochs, {}".format(self.model.EPOCHS) + NL
        f.write(line)

    print("=== Saved training_result_csv file {}".format(self.TRAINING_RESULT_CSV_PATH))
    
    last_val_loss      = float(last_val_loss)
    best_last_val_loss = float(self.get_last_val_loss_from_best_model())
    print("==== best_last_val_loss {}  last_val_loss {}".format(best_last_val_loss, last_val_loss))

    if best_last_val_loss > last_val_loss:
       print("==== best_last_val_loss > last_val_loss")
       
       print("=== Try to copy to {}".format(self.model.BEST_MODEL_DIR))
       try:
         if not os.path.exists(self.model.BEST_MODEL_DIR):
           os.makedirs(self.model.BEST_MODEL_DIR)
       except Exception as ex:
          print("=== Failed to create a directory {}".format(self.model.BEST_MODEL_DIR))
          raise Exception("Failed to create a directory {}".format(self.model.BEST_MODEL_DIR))
          
       try:
         shutil.copy(self.TRAINING_RESULT_CSV_PATH, self.model.BEST_MODEL_DIR)
         shutil.copy(self.model.TRAININIG_CSV_PATH, self.model.BEST_MODEL_DIR)
         shutil.copy(self.model.WEIGHT_FILEPATH,    self.model.BEST_MODEL_DIR)
         print("=== Copied files {} {} {}".format(self.TRAINING_RESULT_CSV_PATH, self.model.TRAININIG_CSV_PATH, self.model.WEIGHT_FILEPATH))
         
       except Exception as ex:
         print("=== Failed to copy files {}".format(ex))
    else:
      print("best_last_val_loss <= last_val_loss, so do nothing")


  def get_last_val_loss_from_best_model(self):
    best_last_val_loss = 100.0
    result_csv = self.model.BEST_MODEL_DIR + "/" + "training_result.csv"
    if not os.path.exists(result_csv):
      print("=== Not found result_csv {}".format(result_csv))
      return best_last_val_loss
      
    with open(result_csv) as f:
      reader = csv.reader(f)
      for row in reader:
         name, value = row
         if name == 'last_val_loss':
           best_last_val_loss = value
           print("=== Found an existing best_last_val_loss {}".format(best_last_val_loss))
           break
    
    return best_last_val_loss
    
