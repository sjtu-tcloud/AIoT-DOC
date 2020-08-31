import numpy as np
import tensorflow as tf
import cv2 as cv
# from tensorflow.tools.graph_transforms import TransformGraph
from datetime import datetime

# inputs = ['image_tensor_float32']
# outputs = ['concat', 'concat_1']
# converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph('ssd_mnv1_const_test_2.pb', inputs, outputs,
#             input_shapes={"image_tensor": [1, 300, 300, 3]})
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)

######################################################################################

interpreter = tf.contrib.lite.Interpreter("converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(input_details)
# print(output_details)

img = cv.imread('timg.jpg')
# print(img.shape, type(img), img.dtype, img.flags['C_CONTIGUOUS'])
rows = img.shape[0]
cols = img.shape[1]
inp = cv.resize(img, (300, 300))
inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
inp = (inp/255.0-0.5)*2.0
inp = np.expand_dims(inp,axis=0)
inp = inp.astype(np.float32)
if not inp.flags['C_CONTIGUOUS']:
    inp = np.ascontiguousarray(inp, dtype=inp.dtype)
print(inp.shape, type(inp), inp.dtype, inp.flags['C_CONTIGUOUS'])

interpreter.set_tensor(input_details[0]['index'], inp)
a=datetime.now()
interpreter.invoke()
b=datetime.now()
print ("===== Tensorflow RESULTS =======")
print("%d.%ds" %((b-a).seconds, (b-a).microseconds))

concat = interpreter.get_tensor(output_details[0]['index'])
concat_1 = interpreter.get_tensor(output_details[1]['index'])
# print(concat.shape, type(concat), concat.dtype, concat.flags['C_CONTIGUOUS'])
# print(concat_1.shape, type(concat_1), concat_1.dtype, concat_1.flags['C_CONTIGUOUS'])

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
    concat = np.ascontiguousarray(concat, dtype=concat.dtype)
loc_ctypes_ptr = cast(concat.ctypes.data, POINTER(c_float))

if not concat_1.flags['C_CONTIGUOUS']:
    concat_1 = np.ascontiguousarray(concat_1, dtype=concat_1.dtype)
cls_ctypes_ptr = cast(concat_1.ctypes.data, POINTER(c_float))

lib.post_process.argtypes = [POINTER(c_float), ctypes.c_int, ctypes.c_int, 
                             POINTER(c_float), POINTER(c_float)]
lib.post_process(img_ctypes_ptr, c_int(cols), c_int(rows), loc_ctypes_ptr, cls_ctypes_ptr)

# print(img.shape, type(img), img.dtype, img.flags['C_CONTIGUOUS'])
# img = img[:, :, [2, 1, 0]]  #RGB2BGR
img = img.transpose((1, 2, 0)) #CHW2HWC
img = (img + 1.0)*255.0/2.0
img = img.astype(np.uint8)
# print(img.shape, type(img), img.dtype, img.flags['C_CONTIGUOUS'])
if not img.flags['C_CONTIGUOUS']:
    img = np.ascontiguousarray(img, dtype=img.dtype)

# print(img.shape, type(img), img.dtype, img.flags['C_CONTIGUOUS'])
cv.imshow('TensorFlow MobileNet-SSD', img)
cv.waitKey(5000)

