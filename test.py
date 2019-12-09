# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import operator
import os
import time
import numpy as np
from pkg_resources import parse_version
from PIL import Image, ImageDraw
import tensorflow as tf
import tflite_runtime.interpreter as tflite

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'


Class = collections.namedtuple('Class', ['id', 'score'])


def input_size(interpreter):
  """Returns input image size as (width, height) tuple."""
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  return width, height


def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = interpreter.get_input_details()[0]['index']
  return interpreter.tensor(tensor_index)()[0]


def output_tensor(interpreter):
  """Returns dequantized output tensor."""
  output_details = interpreter.get_output_details()[0]
  output_data = np.squeeze(interpreter.tensor(output_details['index'])())
  scale, zero_point = output_details['quantization']
  return scale * (output_data - zero_point)


def set_input(interpreter, data):
  """Copies data to input tensor."""
  input_tensor(interpreter)[:, :] = data


def get_output(interpreter, top_k=1, score_threshold=0.0):
  """Returns no more than top_k classes with score >= score_threshold."""
  scores = output_tensor(interpreter)
  classes = [
      Class(i, scores[i])
      for i in np.argpartition(scores, -top_k)[-top_k:]
      if scores[i] >= score_threshold
  ]
  return sorted(classes, key=operator.itemgetter(1), reverse=True)


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])


cifar10 = tf.keras.datasets.cifar10
x_train = cifar10.load_data()[0][0][:1000]

times = dict()

#models = ['MobileNet', 'ResNet50', 'ResNet101', 'VGG16', 'VGG19']
models = ["Demo"]
for m in models:
    print(m)
#    interpreter = make_interpreter('models/' + m + '/edge.tflite')
    interpreter = make_interpreter('./cifar10_post_quant_model_io_edgetpu.tflite')
    interpreter.allocate_tensors()
    size = input_size(interpreter)

    times[m] = list()
    for img in x_train:
        start = time.monotonic()
        set_input(interpreter, img)
        interpreter.invoke()
        get_output(interpreter)
        inference_time = time.monotonic() - start
        times[m].append(inference_time * 1000)
    runtimes = np.array(times[m])
    print("TPU |", m, "| Mean Runtime:", np.mean(runtimes))

for m in models:
    print(m)
#    interpreter = make_interpreter('models/' + m + '/cpu.tflite')
    interpreter = make_interpreter('./saved_model/cifar10_post_quant_model_io.tflite')
    interpreter.allocate_tensors()
    size = input_size(interpreter)

    times[m] = list()
    for img in x_train:
        start = time.monotonic()
        set_input(interpreter, img)
        interpreter.invoke()
        get_output(interpreter)
        inference_time = time.monotonic() - start
        times[m].append(inference_time * 1000)
    runtimes = np.array(times[m])
    print("CPU |", m, "| Mean Runtime:", np.mean(runtimes))