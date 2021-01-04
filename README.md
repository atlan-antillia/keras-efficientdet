# Keras-EfficientDet
This is a simple extension of the implementation of [EfficientDet](https://github.com/xuannianz/EfficientDet) for object detection on Keras and Tensorflow. 
On detail, please see: [EfficientDet](https://github.com/xuannianz/EfficientDet) 

## About pretrained weights
* The pretrained EfficientNet weights on imagenet are downloaded from [Callidior/keras-applications/releases](https://github.com/Callidior/keras-applications/releases)
* The pretrained EfficientDet weights on coco are converted from the official release [google/automl](https://github.com/google/automl).

Thanks for their hard work.
This project is released under the Apache License. Please take their licenses into consideration too when use this project.

**NEW Python Scripts [01/05/2021]**
- EfficientDetFinetuningMode.py
- EfficientDetObjectDetector.py
- EpochChangeCallback.py
- TrainConfig.py
- TrainConfigGenerator.py
- TrainConfigParser.py
- TrainingResultWriter.py

### Create projects folder

 Please create <i>projects</i> folder on your local work folder
    work/
      + keras-efficientdet/
      + projects/


### Create a project folder and deploy a dataset 

 Please create your own <i>demo</i> folder under the <i>projects</i> folder,
   and a <i> dataset</i> folder which contains <i>train</i>, <i>valid</i>, <i>test</i> folders.
   Those folders contain coco-annotation files and jpg image files.

    work/
      + keras-efficientdet/
      + projects/
        + demo/
          + dataset/
            + train/
               + _annotation.coco.json
               + something.jpg

            + valid/
               + _annotation.coco.json
               + something.jpg

            + test/
               + _annotation.coco.json
               + something.jpg

### Generate a configuration file

 In keras-efficientdet folder, please run the follwing command in terminal console 
 to create configuration files.
<br>
<b>
keras-efficientdet>python TrainConfigGenerator.py project dataset 
</b>
<br>
Example:
<br>
keras-efficientdet>python TrainConfigGenerator.py demo BloodCells 


By running the above command, configration file will be generated in <i>config</i> folder under the dataset folder.

    work/
      + keras-efficientdet/
      + projects/
        + demo/
          + BloodCells/
            + config/
               1_B_8_L_0.0001_P_15_E_300_D_0.config


### Train the dataset by EfficientDetFinetuningModel.py
 Run the following command to train your own dataset on the configuration file
generated above.
<br>
<b>
keras-efficientdet>python EfficientDetFinetuningModel.py configurationfile 
</b>

Example:
<br>
keras-efficientdet>python EfficientDetFinetuningModel.py ../projects/demo/BloodCells/config/1_B_8_L_0.0001_P_15_E_300_D_0.config


### Detect objects by EfficientDetObjectDetector.py

 Run the following command to detect objects in an image by using a model
trained by EfficientDetFinetuningModel.
<br>
<b>
keras-efficientdet>python EfficientDetObjectDetector.py image_file_or_dir dataset_dir 
# 
</b>

Example:
<br>
keras-efficientdet>python EfficientDetObjectDetector.py ../projects/demo/BloodCells/test ../projects/demo/BloodCells


