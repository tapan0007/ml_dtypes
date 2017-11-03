#!/usr/bin/env python3

# Top level Tensorflow to Tonga compiler interface
#
# Examples
#
#   set tfpb =  ../../../Kaena-external-opensource/apps/tf/resnet_v2_152/pb/resnet_v2_152_fp32.pb
#
#   python3 tffe.py --tfpb $tfpb --depth 5
#
#   python3 tffe.py --tfpb $tfpb --depth 5 --weights
#
#   python3 tffe.py --tfpb $tfpb --depth 5 --focus block3/unit_18
#
#   set img = ../../../Kaena-external-opensource/apps/tf/resnet_v2_152/dog.jpg
#   ./tffe.py --tfpb $tfpb --depth 5 --images $img

#   set net=jdr_v1
#   python3 $net.py
#   python /home/ubuntu/src/tensorflow_p3/tensorflow/python/tools/freeze_graph.py --input_graph out_$net.pb --input_checkpoint out_$net.data --output_graph out_"$net"_freeze.pb --output_node_names $net/output
#   ./tffe.py --tfpb out_"$net"_freeze.pb --depth 5 --images linear --width 30
#   python3 -c "import numpy as np; x=np.load('out_"$net"__output:0.npy'); print('Output OFMAP\n', x)"


import argparse
import os.path
import TfFrontEnd

parser = argparse.ArgumentParser()
parser.add_argument('--tfpb', help='TensorFlow freeze graph file', default="f.pb")
parser.add_argument('--out_prefix', help='Prefix for output files', default="out_")
parser.add_argument('--weights', help='Generate weight files', dest="weights",
                    action="store_true",  default=False)
parser.add_argument('--images', help='Generate images (IFMAP and OFMAP files) for an input image', dest="images", default=None)
parser.add_argument('--depth', help='Depth of layer name hierarchy to show in the dot output',
                    default=5)
parser.add_argument('--focus', help='Regular expression to filter a subset of nodes',
                    default=".*")
parser.add_argument('--width', help='Highlight data paths wider than the width',
                    default=1000)

args = parser.parse_args()
inputTensorName = "input"

file = args.tfpb
if not os.path.isfile(file):
  raise("ERROR: missing --tfpb " + file)
if args.images != None and args.focus != ".*":
  raise("ERROR: Unsupported --images with --focus")

tffe = TfFrontEnd.TfFe(int(args.width))
tffe.loadPb(file, args.focus)
tffe.writeDot(int(args.depth), args.out_prefix + "graph.dot", "svg")
if args.weights:
  tffe.writeWeights(args.out_prefix)
kog = tffe.getKaenaOpGraph()
if args.images != None:
  tffe.writeImages(args.out_prefix, args.images, inputTensorName)
  kog.identifyMainFlowEdges(inputTensorName)
  tffe.writeOpsCsv(args.out_prefix + "ops.csv")
  tffe.writeDot(int(args.depth), args.out_prefix + "graph_ann.dot", "svg")
kog.genCompilerPy(args.out_prefix + "compiler.py")

