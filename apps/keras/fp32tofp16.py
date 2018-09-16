import re
import argparse
import tensorflow as tf
import numpy as np

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.platform import gfile

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util

def ConvertFP32ToFP16(graphdef):
  sess = tf.Session(graph=tf.import_graph_def(graphdef))
  output_graph_def = graph_pb2.GraphDef()
  dummy_tensor = sess.run(tf.constant([0.1]))
  dummy_tensor_proto = tensor_util.make_tensor_proto(dummy_tensor, \
      dtype=tf.float16, shape=dummy_tensor.shape)
  dummy_tensor32 = sess.run(tf.constant([0.1]))
  dummy_tensor_proto32 = tensor_util.make_tensor_proto(dummy_tensor, \
      dtype=tf.float32, shape=dummy_tensor.shape)
  dt_float_type_attr = attr_value_pb2.AttrValue(type=dummy_tensor_proto32.dtype)
  dt_half_type_attr = attr_value_pb2.AttrValue(type=dummy_tensor_proto.dtype)
  for node in graphdef.node:
    output_node = node_def_pb2.NodeDef()
    output_node.CopyFrom(node)
    if (node.op == "Const"):
      if (node.attr["dtype"] == dt_float_type_attr):
        a = tensor_util.MakeNdarray(node.attr["value"].tensor)
        a = tf.cast(a, tf.float16)
        a = sess.run(a)
        tensor=tensor_util.make_tensor_proto(a, dtype=tf.float16, shape=a.shape)
        output_node.attr["dtype"].CopyFrom(
            attr_value_pb2.AttrValue(type=tensor.dtype))
        output_node.attr["value"].CopyFrom(
            attr_value_pb2.AttrValue(
              tensor=tensor_util.make_tensor_proto(a,\
                dtype=tf.float16, shape=a.shape)))
    else:
      if ("T" in node.attr.keys()):
        if (output_node.attr["T"] == dt_float_type_attr):
          output_node.attr["T"].CopyFrom(
            attr_value_pb2.AttrValue(type=dummy_tensor_proto.dtype))
      if ("dtype" in node.attr.keys()):
        if (node.attr["dtype"] == dt_float_type_attr):
          output_node.attr["dtype"].CopyFrom(dt_half_type_attr)
      if ("SrcT" in node.attr.keys()):
        if (node.attr["SrcT"] == dt_float_type_attr):
          output_node.attr["SrcT"].CopyFrom(dt_half_type_attr)
      if ("DstT" in node.attr.keys()):
        if (node.attr["DstT"] == dt_float_type_attr):
          output_node.attr["DstT"].CopyFrom(dt_half_type_attr)
    output_graph_def.node.extend([output_node])
  return output_graph_def


def load_graph(model_file):
#  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
#  with graph.as_default():
#    tf.import_graph_def(graph_def)

  return graph_def

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--out_graph", help="graph/model to be generated")
  args = parser.parse_args()
  
  graph_f32 = load_graph(args.graph)
  graph_f16 = ConvertFP32ToFP16(graph_f32)
  output_xformed_graph_name = args.out_graph
  with gfile.GFile(output_xformed_graph_name, "wb") as f:
    f.write(graph_f16.SerializeToString())
  with gfile.GFile(output_xformed_graph_name+"txt", 'w') as f:
    f.write(text_format.MessageToString(graph_f16))