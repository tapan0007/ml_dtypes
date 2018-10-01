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

def ConvertFP32ToOther(graphdef, cast_type, graph_type, zero_lower_bytes=False):
  """Converts an FP32 network by casting all constants (weights) to a lower
     precision floating point type (FP16 or BFLOAT16) and updating the dtypes
     everywhere.

     If the cast_type is FP32, the zero_lower_bytes argument can be set to zero
     out the lower two bytes of every weight value. The resulting network is
     still FP32 but the weights have the precision of a BFloat16 value.

     In addition, this supports a hack where the actual numbers are cast to one
     type but the dtype stored in the graph is another (specified via
     graph_type). This is used to generate a BFLOAT16 graph that 'pretends' to
     be an FP16 graph (to work around limitations in tooling)."""
  assert not zero_lower_bytes or cast_type == tf.float32, \
    "zero_lower_bytes only allowed for FP32->FP32"

  sess = tf.Session(graph=tf.import_graph_def(graphdef))
  output_graph_def = graph_pb2.GraphDef()
  dummy_tensor = sess.run(tf.constant([0.1]))
  dummy_tensor_proto = tensor_util.make_tensor_proto(dummy_tensor, \
      dtype=graph_type, shape=dummy_tensor.shape)
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
        if zero_lower_bytes:
          raw = bytearray(a.tobytes())
          assert len(raw) % 4 == 0, "Input size not a multiple of 4?"
          for i in range(0, len(raw)//4):
            raw[i*4] = 0
            raw[i*4+1] = 0
          a = np.frombuffer(raw, dtype=np.float32).reshape(a.shape)
        else:
          a = tf.cast(a, cast_type)
          a = sess.run(a)
        output_node.attr["dtype"].CopyFrom(dt_half_type_attr)
        output_node.attr["value"].CopyFrom(
            attr_value_pb2.AttrValue(
              tensor=tensor_util.make_tensor_proto(a,\
                dtype=cast_type, shape=a.shape)))
        # Hack: Override the type stored in the graph
        if cast_type != graph_type:
          output_node.attr["value"].tensor.dtype = dummy_tensor_proto.dtype
    else:
      if ("T" in node.attr.keys()):
        if (output_node.attr["T"] == dt_float_type_attr):
          output_node.attr["T"].CopyFrom(dt_half_type_attr)
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
  parser.add_argument("--graph", help="graph/model to be executed",
      required=True)
  parser.add_argument("--out_graph", help="graph/model to be generated",
      required=True)
  parser.add_argument("--cast_type", help="type to cast to (float16 or "+
     "bfloat16)", required=True)
  parser.add_argument("--graph_type", help="dtype to set in the graph "+
      "(optional, normally equal to cast_type unless you want to hack it)")
  parser.add_argument("--zero_lower_bytes", help="zero out the lower two "+
      "bytes of every FP32 weight", action='store_true')
  args = parser.parse_args()

  types = {"float16": tf.float16, "bfloat16": tf.bfloat16,
      "float32": tf.float32}
  cast_type = types[args.cast_type]
  if not args.graph_type:
    graph_type = cast_type
  else:
    graph_type = types[args.graph_type]

  if not args.zero_lower_bytes:
    zero_lower_bytes = False
  else:
    zero_lower_bytes = True

  graph_f32 = load_graph(args.graph)
  graph_f16 = ConvertFP32ToOther(graph_f32, cast_type, graph_type, zero_lower_bytes)
  output_xformed_graph_name = args.out_graph
  with gfile.GFile(output_xformed_graph_name, "wb") as f:
    f.write(graph_f16.SerializeToString())
  with gfile.GFile(output_xformed_graph_name+"txt", 'w') as f:
    f.write(text_format.MessageToString(graph_f16))
