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
#print(x)

######################################################################
# Inference on tensorflow
# -----------------------
# Run the corresponding model on tensorflow

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')

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
#    if not tf.gfile.Exists(image):
#        tf.logging.fatal('File does not exist %s', image)
#    image_data = tf.gfile.GFile(image, 'rb').read()
    image_data = x

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
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
