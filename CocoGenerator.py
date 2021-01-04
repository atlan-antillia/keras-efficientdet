#
# CocoGenerator.py
#
# 
import os
import sys
import cv2
import numpy as np

#from model import efficientdet
#from losses import smooth_l1, focal

from generators.common import Generator


from pycocotools.coco import COCO


class CocoGenerator(Generator):
  # Constructor

  def __init__(self, data_dir, annotation, **kwargs):
      self.data_dir = data_dir
      self.annotation = annotation
      print("==== CocoGenerator.data_dir {}".format(data_dir))
      print("==== CocoGenerator.annotation {}".format(data_dir))
                                          
      self.coco = COCO(self.annotation)                
      self.image_ids = self.coco.getImgIds()
      self.load_classes()

      super(CocoGenerator, self).__init__(**kwargs)


  def load_classes(self):
      categories = self.coco.loadCats(self.coco.getCatIds())
      categories.sort(key=lambda x: x['id'])

      self.classes = {}
      self.coco_labels = {}
      self.coco_labels_inverse = {}
      for c in categories:
          self.coco_labels[len(self.classes)] = c['id']
          self.coco_labels_inverse[c['id']] = len(self.classes)
          self.classes[c['name']] = len(self.classes)
      print("=== {}".format(self.classes))
      
      self.labels = {}
      for key, value in self.classes.items():
          self.labels[value] = key
      print("=== {}".format(self.labels))


  def size(self):
      return len(self.image_ids)


  def num_classes(self):
      num = len(self.classes)
      return num-1
      #return 1

  def has_label(self, label):
      return label in self.labels

  def has_name(self, name):
      return name in self.classes

  def name_to_label(self, name):
      return self.classes[name]

  def label_to_name(self, label):
      return self.labels[label]

  def coco_label_to_label(self, coco_label):
      return self.coco_labels_inverse[coco_label]

  def coco_label_to_name(self, coco_label):
      return self.label_to_name(self.coco_label_to_label(coco_label))

  def label_to_coco_label(self, label):
      return self.coco_labels[label]


  def image_aspect_ratio(self, image_index):
      image = self.coco.loadImgs(self.image_ids[image_index])[0]
      return float(image['width']) / float(image['height'])


  def load_image(self, image_index):        
      image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
      
      path = os.path.join(self.data_dir, image_info['file_name'])        
      #print("=== CocoGenrator. load_image {}".format(path))

      image = cv2.imread(path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     
     
      return image


  def load_annotations(self, image_index):
      annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
      annotations = {'labels': np.empty((0,), dtype=np.float32), 'bboxes': np.empty((0, 4), dtype=np.float32)}

      if len(annotations_ids) == 0:
          return annotations
      #print("=== load_annotations")
      
      coco_annotations = self.coco.loadAnns(annotations_ids)
      for idx, a in enumerate(coco_annotations):
          # some annotations have basically no width / height, skip them
          if a['bbox'][2] < 1 or a['bbox'][3] < 1:
              continue

          annotations['labels'] = np.concatenate(
              [annotations['labels'], [a['category_id'] - 1]], axis=0)
          annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
              a['bbox'][0],
              a['bbox'][1],
              a['bbox'][0] + a['bbox'][2],
              a['bbox'][1] + a['bbox'][3],
          ]]], axis=0)          
      #print("=== annotations{}".format(annotations))
      return annotations    

      
  def compute_resize_scale(self, image_shape, min_side=800, max_side=1333):
      """
      Compute an image scale such that the image size is constrained to min_side and max_side.

      Args
          min_side: The image's min side will be equal to min_side after resizing.
          max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

      Returns
          A resizing scale.
      """
      (rows, cols, _) = image_shape

      smallest_side = min(rows, cols)

      # rescale the image so the smallest side is min_side
      scale = min_side / smallest_side

      # check if the largest side is now greater than max_side, which can happen
      # when images have a large aspect ratio
      largest_side = max(rows, cols)
      if largest_side * scale > max_side:
          scale = max_side / largest_side

      return scale


  def resize_image(self, img, min_side=800, max_side=1333):
      """
      Resize an image such that the size is constrained to min_side and max_side.

      Args
          min_side: The image's min side will be equal to min_side after resizing.
          max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

      Returns
          A resized image.
      """
      # compute scale to resize the image
      scale = self.compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

      # resize the image with the computed scale
      img = cv2.resize(img, None, fx=scale, fy=scale)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

      return img, scale

