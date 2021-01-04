#https://raw.githubusercontent.com/fizyr/keras-retinanet/master/keras_retinanet/callbacks/eval.py
#

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os

import numpy as np

from tensorflow import keras
from utils import eval 
#import evaluate
from tqdm import tqdm
from tqdm import trange
import progressbar

import tensorflow as tf
from tensorflow.keras.models import load_model

class EvaluationCallback(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(self, generator, model_dir, model, tensorboard=None, threshold=0.01):
        """ Evaluate callback initializer.
        Args
            generator : The generator used for creating validation data.
            model: prediction model
            tensorboard : If given, the results will be written to tensorboard.
            threshold : The score threshold to use.
        """
        self.model_dir = model_dir
        self.model     = model    #prediction model
        self.generator = generator
        self.model     = None
        self.threshold = threshold
        self.tensorboard = tensorboard

        super(EvaluationCallback, self).__init__()


    def evaluate(self, generator, model, threshold=0.01):
        """
        Use the pycocotools to evaluate a COCO model on a dataset.
        Args
            generator: The generator for generating the evaluation data.
            model: The model to evaluate.
            threshold: The score threshold to use.
        """
        # start collecting results
        results = []
        image_ids = []
        for index in progressbar.progressbar(range(generator.size()), prefix='COCO evaluation: '):
        #for index in trange(generator.size(), desc='COCO evaluation: '):
            image = generator.load_image(index)
            src_image = image.copy()
            h, w = image.shape[:2]

            image, scale = generator.preprocess_image(image)

            # run network
            
            boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
            #boxes, scores,  = model.predict_on_batch([np.expand_dims(image, axis=0)])
            preds = model.predict_on_batch([np.expand_dims(image, axis=0)])
            print("len preds {}".format(len(preds)))
            print(preds)
            
            boxes /= scale
            boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w - 1)
            boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h - 1)
            boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, w - 1)
            #boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, h - 1)

            # change to (x, y, w, h) (MS COCO standard)
            boxes[:, :, 2] -= boxes[:, :, 0]
            #boxes[:, :, 3] -= boxes[:, :, 1]

            # select indices which have a score above the threshold
            indices = np.where(scores[0, :] > threshold)[0]
            print("===== {}".format(indices))
            
            boxes = boxes[0, indices]
            scores = scores[0, indices]
            #class_ids = labels[0, indices]

            # compute predicted labels and scores
            for box, score, class_id in zip(boxes, scores, class_ids):
                # append detection for each positively labeled class
                image_result = {
                    'image_id': generator.image_ids[index],
                    'category_id': int(class_id) + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }
                # append detection to results
                results.append(image_result)

            #     box = np.round(box).astype(np.int32)
            #     class_name = generator.label_to_name(generator.coco_label_to_label(class_id + 1))
            #     ret, baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            #     cv2.rectangle(src_image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 1)
            #     cv2.putText(src_image, class_name, (box[0], box[1] + box[3] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                 (0, 0, 0), 1)
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.imshow('image', src_image)
            # cv2.waitKey(0)

            # append image to list of processed images
            image_ids.append(generator.image_ids[index])

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
        json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

        # # load results in COCO evaluation tool
        # coco_true = generator.coco
        # coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(generator.set_name))
        #
        # # run COCO evaluation
        # coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        # coco_eval.params.imgIds = image_ids
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()
        # return coco_eval.stats

    #https://github.com/xuannianz/EfficientDet/blob/master/eval/coco.py
    def on_epoch_end(self, epoch, logs=None):
        
        num = epoch //4
        print("===EvaluationCallback.on_epoch_end {} num {}".format(epoch, num))

        if epoch == 0 or num != 0:
          return
        
        weight_h5 = os.path.join(self.model_dir, "weight.h5")
        print("=== EvaluationCallback on_epoch_end {}",format(weight_h5))
        
        if not os.path.exists(weight_h5):
          print("=== Not found weight file {}".format(weight_h5))
          
          return
        print("=== EvaluationCallback load_weight")
        
        self.model.load_weights(weight_h5)


        logs = logs or {}

        coco_tag = ['AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50      | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.75      | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]']
        coco_eval_stats = self.evaluate_coco(self.generator, self.model, self.threshold)
        #coco_eval_stats = self.evaluate(self.generator, self.model, self.threshold)
        print("=====coco_eval_stats {}".format(coco_eval_stats))
        if coco_eval_stats is not None and self.tensorboard is not None:
            print("===== coco_eval_stats is not None")

            if tf.version.VERSION < '2.0.0' and self.tensorboard.writer is not None:
                print("===== tf.version < 2.0.0")

                summary = tf.Summary()
                for index, result in enumerate(coco_eval_stats):
                    summary_value = summary.value.add()
                    summary_value.simple_value = result
                    summary_value.tag = '{}. {}'.format(index + 1, coco_tag[index])
                    self.tensorboard.writer.add_summary(summary, epoch)
                    logs[coco_tag[index]] = result
            else:
                print("===== tf.version >=2.0.0")
                for index, result in enumerate(coco_eval_stats):
                    tag = '{}. {}'.format(index + 1, coco_tag[index])
                    tf.summary.scalar(tag, result, epoch)


    def evaluate_coco(self, generator, model, threshold=0.05):
        """
        Use the pycocotools to evaluate a COCO model on a dataset.

        Args
            generator: The generator for generating the evaluation data.
            model: The model to evaluate.
            threshold: The score threshold to use.
        """
        # start collecting results
        results = []
        image_ids = []
        for index in trange(generator.size(), desc='COCO evaluation: '):
            image = generator.load_image(index)
            src_image = image.copy()
            image_shape = image.shape[:2]
            image_shape = np.array(image_shape)
            image = generator.preprocess_image(image)

            # run network
            print("=== Run network ")
            ls = [np.expand_dims(image, axis=0), np.expand_dims(image_shape, axis=0)]
            #ls = tf.convert_to_tensor(ls, dtype=np.float32)
            
            print("=== ls {}".format(ls))
            #ls = ls[0]
            print("=== ls {}".format(ls))
            #ls = tf.convert_to_tensor(ls, dtype=np.float32)
         
            #detections = model.predict_on_batch([np.expand_dims(image, axis=0), np.expand_dims(image_shape, axis=0)])[0]
            dets = model.predict_on_batch(ls)
            #detections = model.predict_on_batch(ls)[0]
            detections = dets[0]

            print("=== detections{}".format(detections))
            
            # change to (x, y, w, h) (MS COCO standard)
            boxes = np.zeros((detections.shape[0], 4), dtype=np.int32)
            # xmin
            boxes[:, 0] = np.maximum(np.round(detections[:, 1]).astype(np.int32), 0)
            # ymin
            boxes[:, 1] = np.maximum(np.round(detections[:, 0]).astype(np.int32), 0)
            # w
            boxes[:, 2] = np.minimum(np.round(detections[:, 3] - detections[:, 1]).astype(np.int32), image_shape[1])
            # h
            boxes[:, 3] = np.minimum(np.round(detections[:, 2] - detections[:, 0]).astype(np.int32), image_shape[0])
            scores = detections[:, 4]
            class_ids = detections[:, 5].astype(np.int32)
            # compute predicted labels and scores
            for box, score, class_id in zip(boxes, scores, class_ids):
                # scores are sorted, so we can break
                if score < threshold:
                    break

                # append detection for each positively labeled class
                image_result = {
                    'image_id': generator.image_ids[index],
                    'category_id': generator.label_to_coco_label(class_id),
                    'score': float(score),
                    'bbox': box.tolist(),
                }
                # append detection to results
                results.append(image_result)
                class_name = generator.label_to_name(class_id)
                ret, baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(src_image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 1)
                cv2.putText(src_image, class_name, (box[0], box[1] + box[3] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', src_image)
            cv2.waitKey(0)
            # append image to list of processed images
            image_ids.append(generator.image_ids[index])

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
        json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = generator.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(generator.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats


    def xxevaluate_coco(self, generator, model, threshold=0.05):
        """ Use the pycocotools to evaluate a COCO model on a dataset.

        Args
            generator : The generator for generating the evaluation data.
            model     : The model to evaluate.
            threshold : The score threshold to use.
        """
        # start collecting results
        results = []
        image_ids = []
        for index in progressbar.progressbar(range(generator.size()), prefix='COCO evaluation: '):
            image = generator.load_image(index)
            #image = generator.preprocess_image(image)
            image, scale = generator.resize_image(image)

            if keras.backend.image_data_format() == 'channels_first':
                image = image.transpose((2, 0, 1))

            input = np.expand_dims(image, axis=0)
            #input = input.astype(int)
            input = tf.convert_to_tensor(input)
            
            # run network
            #boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes, scores, labels = model.predict_on_batch(input)

            # correct boxes for image scale
            boxes /= scale

            # change to (x, y, w, h) (MS COCO standard)
            boxes[:, :, 2] -= boxes[:, :, 0]
            boxes[:, :, 3] -= boxes[:, :, 1]

            # compute predicted labels and scores
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted, so we can break
                if score < threshold:
                    break

                # append detection for each positively labeled class
                image_result = {
                    'image_id'    : generator.image_ids[index],
                    'category_id' : generator.label_to_coco_label(label),
                    'score'       : float(score),
                    'bbox'        : box.tolist(),
                }

                # append detection to results
                results.append(image_result)

            # append image to list of processed images
            image_ids.append(generator.image_ids[index])

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
        json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = generator.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(generator.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

    def xxxon_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print("==== on_epoch_end {}".format(epoch))
        
        # run evaluation
        average_precisions, _ = eval.evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        # compute per class average precision
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        if self.tensorboard:
            import tensorflow as tf
            writer = tf.summary.create_file_writer(self.tensorboard.log_dir)
            with writer.as_default():
                tf.summary.scalar("mAP", self.mean_ap, step=epoch)
                if self.verbose == 1:
                    for label, (average_precision, num_annotations) in average_precisions.items():
                        tf.summary.scalar("AP_" + self.generator.label_to_name(label), average_precision, step=epoch)
                writer.flush()

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))
            
            