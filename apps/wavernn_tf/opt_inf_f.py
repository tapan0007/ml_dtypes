



import re
import argparse
import tensorflow as tf
import numpy as np


from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph

model_name = "wave_rnn_ts0_cb_f"
if(1) : 
    # read back frozen graph
    frozen_graph_def = graph_pb2.GraphDef()
    with gfile.FastGFile("waver_rnn_tf_ts0_cb.pb", "rb") as f:
        frozen_graph_def.ParseFromString(f.read())
    # write out in text for debugging
    #with gfile.GFile(frozen_file+"txt", 'w') as f:
    #    f.write(text_format.MessageToString(frozen_graph_def))

    # fold batch normalization and constants and remove unused nodes
    transformed_graph_def = graph_pb2.GraphDef()
    transformed_graph_def = TransformGraph (
             frozen_graph_def,
             ['cond','prev'],
             ['multinomial/Multinomial'],
             [
                'add_default_attributes',
                'remove_nodes(op=Identity, op=CheckNumerics)',
                'fold_constants(ignore_errors=true)',
                'fold_batch_norms',
                'fold_batch_norms',
                'fold_old_batch_norms',
                'strip_unused_nodes',
                'sort_by_execution_order',
                
             ])

    output_xformed_graph_name = model_name + "_opt.pb"
    with gfile.GFile(output_xformed_graph_name, "wb") as f:
        f.write(transformed_graph_def.SerializeToString())
    #with gfile.GFile(output_xformed_graph_name+"txt", 'w') as f:
    #    f.write(text_format.MessageToString(transformed_graph_def))


