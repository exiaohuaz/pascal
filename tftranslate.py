import os

import tensorflow as tf

from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import dataset_util

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

from google.protobuf import text_format

import glob
import re
import xml.etree.ElementTree as ET
import os

LABEL_MAP_NAME = 'data/label_map.pbtxt'

#labels
labels = []
label_map = string_int_label_map_pb2.StringIntLabelMap()

def create_tf_example(data):
  print('Start create tf example...')
  size = data.find('size')
  height = int(size.find('height').text) # Image height   #get from actual file, don't hardcode
  width = int(size.find('width').text) # Image width
  print("Height: ", height, " Width: ", width)
  image_format = b'jpg'

  xmins = []
  xmaxs = [] 
  ymins = [] 
  ymaxs = [] 
  classes_text = [] 
  classes = [] 
  filename = 'VOCdevkit/VOC2012/JPEGImages/' + data.find('filename').text
  
  with open(filename, "rb") as f2:
    encoded_image_data = f2.read() #solved by adding "rb" to open. specifies that it's binary?
    for item in data.findall("object"):
      print("Start object...")
      #labels
      label = item.find('name').text
      if label not in labels:
        #print("new label: " + label)
        labels.append(label)
        newlabel = string_int_label_map_pb2.StringIntLabelMapItem()
        newlabel.name = label
        newlabel.id = len(labels)
        label_map.item.append(newlabel)
        lid = newlabel.id
      else: 
        lid = labels.index(label)
      #end labels

      print("Label: ", label, " ID: ", lid)
      classes_text.append(label.encode())
      classes.append(lid)
      bndbox = item.find('bndbox')
      xmins.append(float(bndbox.find('xmin').text)/width)
      ymins.append(float(bndbox.find('ymin').text)/height)
      xmaxs.append(float(bndbox.find('xmax').text)/width)
      ymaxs.append(float(bndbox.find('ymax').text)/height)
      print("xmin: ", xmins[-1], " ymin: ", ymins[-1], " xmax: ", xmaxs[-1], " ymax: ", ymaxs[-1])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(filename.encode()),
    'image/source_id': dataset_util.bytes_feature(filename.encode()),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
  return tf_example

def main():
  index = 0
  for annotation in os.listdir("VOCdevkit/VOC2012/Annotations/"):
    if annotation.startswith("2012"):
      tree = ET.parse('VOCdevkit/VOC2012/Annotations/' + annotation)
      data = tree.getroot()

      num_shards = 3
      output_filebase = 'data/file.tfrecord'
      with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
          tf_record_close_stack, output_filebase, num_shards)
        
        tf_example = create_tf_example(data)
        #import pudb; pu.db #a breakpoint
        print("Index: ", index)
        output_shard_index = index % num_shards
        print("Output shard index: ", output_shard_index)
        output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
        index = index + 1

  with open(LABEL_MAP_NAME, 'w') as f:
    f.write(text_format.MessageToString(label_map))

if __name__ == '__main__':
  main()