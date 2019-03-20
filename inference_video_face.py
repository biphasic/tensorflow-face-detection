#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import os
import argparse

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Face Detection using SSD')
  parser.add_argument('--video_folder', dest='video_folder', help='Folder of video')
  parser.add_argument('--video_name', dest='video_name', help='file name of video')
  parser.add_argument('--conf_thresh', dest='conf_thresh', help='Confidence threshold for the detections, float from 0 to 1', default=0.7, type=float)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './protos/face_label_map.pbtxt'

    NUM_CLASSES = 2

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    width = 304
    height = 240
    frame_rate = 25.0

    args = parse_args()
    path = os.path.normpath(args.video_folder).split(os.sep)
    recording_number = path[-1]

    video_path = os.path.join(args.video_folder, args.video_name)

    print(video_path)
    if not os.path.exists(video_path):
        raise IOError('Video does not exist.')

    video = cv2.VideoCapture(video_path)
    out = None

    csv_file = os.path.join(args.video_folder, '%s-ssd-annotations.csv' % recording_number)
    print(csv_file)
    fid_csv = open(csv_file, 'w')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with tf.Session(graph=detection_graph, config=config) as sess:
        frame_num = 1;
        while True:
          ret, image = video.read()
          if ret != True:
              break

          if out is None:
              out_path =  args.video_folder + '/%s-ssd.avi' % recording_number
              fourcc = cv2.cv.CV_FOURCC(*'XVID')
              out = cv2.VideoWriter(out_path, fourcc, frame_rate, (width, height))

          image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          start_time = time.time()
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          elapsed_time = time.time() - start_time
          print('inference time cost: {}'.format(elapsed_time))
          #print(boxes.shape, boxes)
          #print(boxes.shape[0])
          #print(scores.shape,scores)
          #print(classes.shape,classes)
          #print(num_detections)
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=4,
              min_score_thresh=args.conf_thresh)

          for i, box in enumerate(np.squeeze(boxes)):
              if np.squeeze(scores)[i] > args.conf_thresh:
                  print("frame={}, ymin={}, xmin={}, ymax={}, xmax={}".format(frame_num, box[0]*height, box[1]*width, box[2]*height, box[3]*width))
                  fid_csv.write(str(frame_num*1000000/frame_rate) + ', %f, %f, %f, %f\n' % (box[0]*height, box[1]*width, box[2]*height, box[3]*width))
          out.write(image)
          frame_num += 1

        video.release()
        fid_csv.close()
        out.release()
