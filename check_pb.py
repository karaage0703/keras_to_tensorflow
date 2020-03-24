#!/usr/bin/env python
# reference
# https://toxweblog.toxbe.com/2018/06/07/tensorboard-open-pb-file/
# https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/

import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util

parser = argparse.ArgumentParser(description='check pb')
parser.add_argument('-m', '--model', default='./model.pb')
parser.add_argument('--logdir', default='LOGDIR')

args = parser.parse_args()

with tf.Session() as sess:
    with gfile.FastGFile(args.model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

    train_writer = tf.summary.FileWriter(args.logdir)
    train_writer.add_graph(sess.graph)

    print('Check out the input placeholders:')
    nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
    for node in nodes:
        print(node)

    print('Get layer names')
    layers = [op.name for op in sess.graph.get_operations()]
    for layer in layers:
        print(layer)

    print('Check out the weights of the nodes')
    weight_nodes = [n for n in graph_def.node if n.op == 'Const']
    for n in weight_nodes:
        print("Name of the node - %s" % n.name)
        print("Value - " )
        print(tensor_util.MakeNdarray(n.attr['value'].tensor))

