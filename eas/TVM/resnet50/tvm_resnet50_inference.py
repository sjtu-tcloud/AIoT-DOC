# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile Tensorflow Models
=========================
This article is an introductory tutorial to deploy tensorflow models with TVM.

For us to begin with, tensorflow python module is required to be installed.

Please refer to https://www.tensorflow.org/install
"""

# tvm, relay
import tvm
from tvm import te
from tvm import relay

# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf
tf_compat_v1 = tf

#for count time
from datetime import datetime

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

# Base location for model related files.
repo_base = '.'

# Test image
img_name = 'dog.png'
img_path = os.path.join(repo_base, img_name)
model_path = os.path.join(repo_base, 'resnet50_v1.pb')
# Human readable text for labels
label_path = os.path.join(repo_base, 'label.txt')

# Target settings
# Use these commented settings to build for cuda.
#target = 'cuda'
#target_host = 'llvm'
#layout = "NCHW"
#ctx = tvm.gpu(0)
target = 'llvm'
target_host = 'llvm'
#layout = None
layout = "NHWC"
ctx = tvm.cpu(0)

def load_label():
    label=['Background']
    with open(label_path,'r',encoding='utf-8') as r:
        lines = r.readlines()
        for l in lines:
            l = l.strip()
            arr = l.split(',')
            label.append(arr[1])
    return label

label = load_label()

from tvm.contrib import graph_runtime

# load the module back.
loaded_json = open("deploy_graph.json").read()
loaded_lib = tvm.runtime.load_module("deploy_lib.tar")
loaded_params = bytearray(open("deploy_param.params", "rb").read())

m = graph_runtime.create(loaded_json, loaded_lib, ctx)
m.load_params(loaded_params)

######################################################################
#Load Img and pre-process
from PIL import Image
image = Image.open(img_path).resize((224, 224))

x = np.array(image)
x = x.astype('float32')
#print(x.shape)
#print(x)
x[ :, :, 0] -= 103.939
x[ :, :, 1] -= 116.779
x[ :, :, 2] -= 123.68
x = np.expand_dims(x,axis=0)
#print(x)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------

# set inputs
m.set_input('input_tensor', tvm.nd.array(x.astype('float32')))
# execute
a=datetime.now()
m.run()
b=datetime.now()
print ("===== TVM RESULTS =======")
print("%d.%ds" %((b-a).seconds, (b-a).microseconds))

# get outputs
tvm_output = m.get_output(0, tvm.nd.empty(((1, 1001)), 'float32'))

######################################################################
# Process the output
# ------------------
# Process the model output to human readable text
predictions = tvm_output.asnumpy()
predictions = np.squeeze(predictions)

# Print top 5 predictions from TVM output.
top_k = predictions.argsort()[-5:][::-1]
for node_id in top_k:
    score = predictions[node_id]
    print('[%4d]%s: %.5f' % (node_id, label[node_id], score))

