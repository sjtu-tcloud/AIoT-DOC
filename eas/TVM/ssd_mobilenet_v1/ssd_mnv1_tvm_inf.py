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

from datetime import datetime

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

# Base location for model related files.
repo_base = '.'

label_path = os.path.join(repo_base, 'coco_labels91.txt')

def load_label():
    label=['Background']
    with open(label_path,'r',encoding='utf-8') as r:
        lines = r.readlines()
        for l in lines:
            l = l.strip()
            label.append(l)
    return label

label = load_label()

# Target settings
# Use these commented settings to build for cuda.
#target = 'cuda'
#target_host = 'llvm'
#layout = "NCHW"
#ctx = tvm.gpu(0)
target = 'llvm'
target_host = 'llvm'
layout = None
ctx = tvm.cpu(0)

# from PIL import Image

# image = Image.open(img_path).resize((300, 300))
# x = np.array(image)
# x = x[:, :, [2, 1, 0]]  # BGR2RGB
# x = (x/255.0-0.5)*2.0
# x = np.expand_dims(x,axis=0)
#print(x.shape)
#print(x)

import cv2 as cv
image = cv.imread('timg.jpg')
x = cv.resize(image, (300, 300))
x = x[:, :, [2, 1, 0]]  # BGR2RGB
x = (x/255.0-0.5)*2.0

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

concat = tvm.nd.empty((1917,4), ctx=ctx)
concat_1 = tvm.nd.empty((1917,91), ctx=ctx)

# set inputs
m.set_input('image_tensor_float32', tvm.nd.array(x.astype('float32')))
# execute
a=datetime.now()
m.run()
b=datetime.now()
print ("===== TVM RESULTS =======")
print("%d.%ds" %((b-a).seconds, (b-a).microseconds))
# print(m.get_num_outputs())

concat = m.get_output(0).asnumpy()
concat = np.squeeze(concat)
# print(concat.shape)
# print(concat)

concat_1 = m.get_output(1).asnumpy()
concat_1 = np.squeeze(concat_1)
# print(concat_1.shape)
# print(concat_1)

###################################################
img = image

import ctypes
from ctypes import *

lib = ctypes.cdll.LoadLibrary("./libpostprocess.so")

img = img.transpose((2, 0, 1)) #HWC2CHW
# img = img[:, :, [2, 1, 0]]  # BGR2RGB
img = (img/255.0-0.5)*2.0
if not img.flags['C_CONTIGUOUS']:
    img = np.ascontiguousarray(img, dtype=np.float32)
img_ctypes_ptr = cast(img.ctypes.data, POINTER(c_float))

if not concat.flags['C_CONTIGUOUS']:
    concat = np.ascontiguousarray(concat, dtype=np.float32)
loc_ctypes_ptr = cast(concat.ctypes.data, POINTER(c_float))

if not concat_1.flags['C_CONTIGUOUS']:
    concat_1 = np.ascontiguousarray(concat_1, dtype=np.float32)
cls_ctypes_ptr = cast(concat_1.ctypes.data, POINTER(c_float))

# print(img.shape, type(img), img.dtype, img.flags['C_CONTIGUOUS'])
# print(concat.shape, type(concat), concat.dtype, concat.flags['C_CONTIGUOUS'])
# print(concat_1.shape, type(concat_1), concat_1.dtype, concat_1.flags['C_CONTIGUOUS'])

lib.post_process.argtypes = [POINTER(c_float), ctypes.c_int, ctypes.c_int, 
                             POINTER(c_float), POINTER(c_float)]
lib.post_process(img_ctypes_ptr, c_int(img.shape[2]), c_int(img.shape[1]), loc_ctypes_ptr, cls_ctypes_ptr)

# print(img.shape, type(img), img.dtype, img.flags['C_CONTIGUOUS'])
# img = img[:, :, [2, 1, 0]]  #RGB2BGR
img = img.transpose((1, 2, 0)) #CHW2HWC
img = (img + 1.0)*255.0/2.0
img = img.astype(np.uint8)
# print(img.shape, type(img), img.dtype, img.flags['C_CONTIGUOUS'])
if not img.flags['C_CONTIGUOUS']:
    img = np.ascontiguousarray(img, dtype=img.dtype)
# print(img.shape, type(img), img.dtype, img.flags['C_CONTIGUOUS'])

# import matplotlib.pyplot as plt
# img = img[:, :, [2, 1, 0]]  # BGR2RGB
# plt.imshow(img)

cv.imshow('TensorFlow MobileNet-SSD', img)
cv.waitKey(5000)



