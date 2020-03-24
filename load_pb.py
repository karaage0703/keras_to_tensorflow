#!/usr/bin/env python
# reference
# https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/

import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='load pb')
parser.add_argument('-m', '--model', default='./model.pb')
parser.add_argument('--file', default='test.jpg')

args = parser.parse_args()

with tf.Session() as sess:
    with gfile.FastGFile(args.model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

        X = []
        img = cv2.imread(args.file)
        img = cv2.resize(img, (64, 64))
        img = img_to_array(img)
        img = img/255
        X.append(img)
        X = np.asarray(X)

        output_tensor = sess.graph.get_tensor_by_name('import/activation_3/Softmax:0')
        output = sess.run(output_tensor, feed_dict = {'import/conv2d_input:0': X})
        print(output)