#Copyright 2020-2021 antillia.com Toshiyuki Arai
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

# draw_boxes_with_filters.py

# This is based on draw_boxes.py

import cv2
      
def draw_boxes_with_filters(image, boxes, scores, labels, colors, classes, detected_objects, objects_stats, filters=None):
    for b, l, s in zip(boxes, labels, scores):
        class_id = int(l)
        class_name = classes[class_id]
    
        xmin, ymin, xmax, ymax = list(map(int, b))
        score = '{:.4f}'.format(s)
        color = colors[class_id]
        label = '-'.join([class_name, score])
    
        x = int(xmin)
        y = int(ymin)
        w = int(xmax - xmin)
        h = int(ymax - ymin)
        
        if filters is None:
          ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
          cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
          cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
          cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

          id = len(detected_objects) +1
          # id, class, score, x, y, w, h
          
          print("{},  {}, {}, {}, {}, {}, {}".format(id, class_name, score, x, y, w, h))
          detected_objects.append((id, class_name, score, x, y, w, h))
        
          if class_name not in objects_stats:
             objects_stats[class_name] = 1
          else:
            count = int(objects_stats[class_name]) 
            objects_stats.update({class_name: count+1})
        
        else:
         if class_name in filters:
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            id = len(detected_objects) +1
            # id, class, score, x, y, w, h
            detected_objects.append((id, class_name, score, x, y, w, h))

            # 2020/08/15
            if class_name not in objects_stats:
               objects_stats[class_name] = 1
            else:
              count = int(objects_stats[class_name]) 
              objects_stats.update({class_name: count+1})

