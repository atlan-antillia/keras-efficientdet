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

# EpochChangeCallback.py
#


import os
import sys
import glob
import socket

import time
import traceback
import tensorflow

from METRICS import *


class EpochChangeCallback(tensorflow.keras.callbacks.Callback):
  ##
  # Constructor
  def __init__(self, training_csv_path, ipaddress, port, epochs=100):
    self.TRAININIG_CSV_PATH  = training_csv_path
    self.epochs              = epochs
         
    self.remove_training_csv_file()
    self.write_training_log_header("epoch", "loss", "val_loss", 
                                   "regression_acc", "val_regression_acc", 
                                   "classification_acc", "val_classification_acc")
                                   #"ft_classification_acc", "val_ft_classification_acc")

    self.sock     = None
    self.notifier = ""
    self.epochs   = epochs

    # Create a DATGRAM socket
    try:
      self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      self.server_address = (ipaddress, port)
      print(self.server_address)
    except:
      print(formatted_traceback())


  def remove_training_csv_file(self):
    try:
      if os.path.exists(self.TRAININIG_CSV_PATH):
        os.remove(self.TRAININIG_CSV_PATH)
    except:
      traceback.print_exc()

  def write_training_log_header(self, epoch, loss, val_loss, 
                 regression_acc, val_regression_acc,
                 ft_classification_acc, val_ft_classification_acc):
     NL  = "\n"
     try:
       with open(self.TRAININIG_CSV_PATH, "a") as f:
         # epoch loss, acc, val_loss, val_acc
         line = "{}, {}, {}, {}, {}, {}, {}".format(epoch, loss, val_loss, regression_acc, val_regression_acc, ft_classification_acc, val_ft_classification_acc)
         #print("=== {}".format(line))
         
         line = line + NL
         f.write(line)
         
     except:
      traceback.print_exc()


  def write_training_log(self, epoch, loss, val_loss, regression_acc, val_regression_acc, ft_classification_acc, val_ft_classification_acc):
     NL  = "\n"
     try:
       with open(self.TRAININIG_CSV_PATH, "a") as f:
         # epoch loss, acc, val_loss, val_acc
         line = "{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(epoch, loss, val_loss, regression_acc, val_regression_acc, ft_classification_acc, val_ft_classification_acc)
         #print("=== {}".format(line))
         
         line = line + NL
         f.write(line)
         
     except:
      traceback.print_exc()
       

  def on_train_begin(self, logs={}):
    print("on_train_begin")
    self.send("on_train_begin:" + self.notifier + ":" + str(self.epochs) )


  def send(self, message):
    text = str(message)
    # You should send a "utf-8" encoded data
    data = text.encode("utf-8")
    self.sock.sendto(data, self.server_address)


  def on_epoch_end(self, epoch, logs):
    #print("=========== on_epoch_end {}".format(logs))
    """
    LOSS                      = "loss"
    VAL_LOSS                  = "val_loss"
    FT_CLASSIFICATION_ACC     = "ft_classification_acc"
    CLASSIFICATION_ACC        = "classification_acc"
    
    REGRESSION_ACC            = "regression_acc"
    VAL_FT_CLASSIFICATION_ACC = "val_ft_classification_acc"
    VAL_CLASSIFICATION_ACC    = "val_classification_acc"
    
    VAL_REGRESSION_ACC        = "val_regression_acc"
    """
    
    loss                      = 0.0
    val_loss                  = 0.0
    classification_acc        = 0.0
    val_classification_acc    = 0.0
    
    regression_acc            = 0.0
    val_regression_acc        = 0.0

    ft_classification_acc     = 0.0
    val_ft_classification_acc = 0.0

    if LOSS in logs:
      loss     = logs.get(LOSS)

    if VAL_LOSS in logs:
      val_loss = logs.get(VAL_LOSS)
    
    if REGRESSION_ACC in logs:
      regression_acc = logs.get(REGRESSION_ACC)

    if VAL_REGRESSION_ACC in logs:
      val_regression_acc = logs.get(VAL_REGRESSION_ACC)

    if FT_CLASSIFICATION_ACC in logs:
      ft_classification_acc = logs.get(FT_CLASSIFICATION_ACC)

    if CLASSIFICATION_ACC in logs:
      ft_classification_acc = logs.get(CLASSIFICATION_ACC)

    if VAL_FT_CLASSIFICATION_ACC in logs:
      val_ft_classification_acc = logs.get(VAL_FT_CLASSIFICATION_ACC)

    if VAL_CLASSIFICATION_ACC in logs:
      val_ft_classification_acc = logs.get(VAL_CLASSIFICATION_ACC)

    message = "{},  {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f} ".format(epoch, loss, regression_acc, val_loss, val_regression_acc, ft_classification_acc, val_ft_classification_acc)
    self.send(message)
    print("===== send message {}".format(message))
    self.write_training_log(epoch, loss, val_loss, regression_acc, val_regression_acc, ft_classification_acc, val_ft_classification_acc)
            
    print("=== on_epoch_end {}".format(message))
    

    
  def close(self):
    if self.sock != None:
      self.sock.close()
      self.sock = None
        
