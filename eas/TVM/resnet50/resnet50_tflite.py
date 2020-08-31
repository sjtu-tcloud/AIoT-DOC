# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf

#for count time
from datetime import datetime

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


inputs = ['input_tensor']
outputs = ['softmax_tensor']
converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(model_name, inputs, outputs,
             input_shapes={"input_tensor": [1, 224, 224, 3]})
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

######################################################################################

#print(img_path)
#print(model_path)
#print(label_path)

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
x = x.astype(np.float32)
if not x.flags['C_CONTIGUOUS']:
    x = np.ascontiguousarray(x, dtype=x.dtype)
print(x.shape, type(x), x.dtype, x.flags['C_CONTIGUOUS'])

######################################################################
# Inference on tensorflow
# -----------------------
# Run the corresponding model on tensorflow lite

def run_inference_on_image(image):

    image_data = x

    interpreter = tf.contrib.lite.Interpreter("converted_model.tflite")
    interpreter.allocate_tensors()

# Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], x)
    a=datetime.now()
    interpreter.invoke()
    b=datetime.now()
    print ("===== Tensorflow RESULTS =======")
    print("%d.%ds" %((b-a).seconds, (b-a).microseconds))

    predictions = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.squeeze(predictions)

# Print top 5 predictions from tensorflow.
    top_k = predictions.argsort()[-5:][::-1]
    for node_id in top_k:
        score = predictions[node_id]
        print('[%4d]%s: %.5f' % (node_id, label[node_id], score))

run_inference_on_image(img_path)
