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

model_name = 'resnet50_v1.pb'
model_path = os.path.join(repo_base, model_name)

# Human readable text for labels
label_map = 'label.txt'
label_path = os.path.join(repo_base, label_map)

print(img_path)
print(model_path)
print(label_path)

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

######################################################################
# Import model
# ------------
# Creates tensorflow graph definition from protobuf file.

with tf_compat_v1.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf_compat_v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax_tensor')

######################################################################
#Load Img and pre-process
from PIL import Image
image = Image.open(img_path).resize((224, 224))

x = np.array(image)
x = x.astype('float32')
print(x.shape)
#print(x)
x[ :, :, 0] -= 103.939
x[ :, :, 1] -= 116.779
x[ :, :, 2] -= 123.68
x = np.expand_dims(x,axis=0)
#print(x)

######################################################################
# Import the graph to Relay
# -------------------------
# Import tensorflow graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
shape_dict = {'input_tensor': (1,224,224,3)}
mod, params = relay.frontend.from_tensorflow(graph_def,
                                             layout=layout,
                                             shape=shape_dict)

print("Tensorflow protobuf imported to relay frontend.")
######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
#
# Results:
#   graph: Final graph after compilation.
#   params: final params after compilation.
#   lib: target library which can be deployed on target with TVM runtime.

with tvm.transform.PassContext(opt_level=3):
#    lib = relay.build(mod, target=target, target_host=target_host, params=params)
    graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)

# save the graph, lib and params into separate files
from tvm.contrib import util

lib.export_library("deploy_lib.tar")
with open("deploy_graph.json", "w") as fo:
    fo.write(graph)
with open("deploy_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.

from tvm.contrib import graph_runtime
#m = graph_runtime.GraphModule(lib['default'](ctx))

# load the module back.
loaded_json = open("deploy_graph.json").read()
loaded_lib = tvm.runtime.load_module("deploy_lib.tar")
loaded_params = bytearray(open("deploy_param.params", "rb").read())

m = graph_runtime.create(loaded_json, loaded_lib, ctx)
m.load_params(loaded_params)

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
# Process the model output to human readable text for InceptionV1.
predictions = tvm_output.asnumpy()
predictions = np.squeeze(predictions)

# Print top 5 predictions from TVM output.
top_k = predictions.argsort()[-5:][::-1]
for node_id in top_k:
    score = predictions[node_id]
    print('[%4d]%s: %.5f' % (node_id, label[node_id], score))
######################################################################
# Inference on tensorflow
# -----------------------
# Run the corresponding model on tensorflow

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf_compat_v1.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

def run_inference_on_image(image):
    """Runs inference on an image.

    Parameters
    ----------
    image: String
        Image file name.

    Returns
    -------
        Nothing
    """
#    if not tf_compat_v1.gfile.Exists(image):
#        tf.logging.fatal('File does not exist %s', image)
#    image_data = tf_compat_v1.gfile.GFile(image, 'rb').read()
    image_data = x

    # Creates graph from saved GraphDef.
    create_graph()

    with tf_compat_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax_tensor:0')
        a=datetime.now()
        predictions = sess.run(softmax_tensor,
                               {'input_tensor:0': image_data})
        b=datetime.now()
        predictions = np.squeeze(predictions)

        # Print top 5 predictions from tensorflow.
        top_k = predictions.argsort()[-5:][::-1]
        print ("===== TENSORFLOW RESULTS =======")
        print("%d.%ds" %((b-a).seconds, (b-a).microseconds))
        for node_id in top_k:
            score = predictions[node_id]
            print('[%4d]%s: %.5f' % (node_id, label[node_id], score))

run_inference_on_image(img_path)
