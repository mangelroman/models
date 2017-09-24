# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Bosch Small Traffic Lights dataset to TFRecord for object_detection.

     https://hci.iwr.uni-heidelberg.de/node/6132

Example usage:
    ./create_bosch_tf_record --data_dir=/home/user/bosch \
        --output_dir=/home/user/bosch/output \
        --label_map_path=/home/user/bosch/label_map.pbtxt
"""

import hashlib
import io
import logging
import os
import random

import PIL.Image
import tensorflow as tf
import yaml

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
FLAGS = flags.FLAGS

def create_tf_example(data, label_map_dict, image_dir, ignore_occluded=False):
  """Convert YAML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding YAML fields for a single image
    label_map_dict: A map from string label names to integers ids.
    image_dir: String specifying the directory holding the actual image data.
    ignore_occluded: Whether to skip occluded instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  filename = data['path']
  img_path = os.path.join(image_dir, filename)

  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = PIL.Image.open(encoded_png_io)
  if image.format != 'PNG':
    raise ValueError('Image format not PNG')
  key = hashlib.sha256(encoded_png).hexdigest()

  width = 1280
  height = 720

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []

  for obj in data['boxes']:

    if ignore_occluded and obj['occluded']:
      continue

    xmin.append(float(obj['x_min']) / width)
    ymin.append(float(obj['y_min']) / height)
    xmax.append(float(obj['x_max']) / width)
    ymax.append(float(obj['y_max']) / height)

    class_name = obj['label']
    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          filename.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     data_dir,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    data_dir: Directory where data files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))

    tf_example = create_tf_example(example, label_map_dict, data_dir, ignore_occluded=True)
    writer.write(tf_example.SerializeToString())

  writer.close()

def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from Bosch dataset.')
  examples_path = os.path.join(data_dir, 'train.yaml')

  examples_list = yaml.load(open(examples_path, 'rb').read())

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.7 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'bosch_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'bosch_val.record')
  create_tf_record(train_output_path, label_map_dict, data_dir, train_examples)
  create_tf_record(val_output_path, label_map_dict, data_dir, val_examples)

if __name__ == '__main__':
  tf.app.run()
