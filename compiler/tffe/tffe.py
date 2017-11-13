#!/usr/bin/env python3

# Top level Tensorflow to Tonga compiler interface
#
# Examples
#
#   set tfpb =  $KAENA_PATH/../Kaena-external-opensource/apps/tf/resnet_v2_152/pb/resnet_v2_152_fp32.pb
#
#   python3 tffe.py --tfpb $tfpb --depth 5
#
#   python3 tffe.py --tfpb $tfpb --depth 5 --weights
#
#   python3 tffe.py --tfpb $tfpb --depth 5 --focus block3/unit_18
#
#   set img = $KAENA_PATH/../Kaena-external-opensource/apps/tf/resnet_v2_152/dog.jpg
#   $KAENA_PATH/compiler/tffe//tffe.py --tfpb $tfpb --depth 5 --images $img
#     firefox out_graph_ann.dot.svg
#     soffice  out_ops.csv  
#
#   See the Makefile for examples of running units tests


import argparse
import os.path
import TfFrontEnd
import sys

print("\nINFO: started as  ", " ".join(sys.argv))

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
parser.add_argument('--verbose', help='Verbosity level, 0 default, 1 shows in/outputs, 2 TBD',
                    default=0)

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
  fileList = kog.genCompilerPy(args.out_prefix + "compiler.py", int(args.verbose))
  fileList += [args.out_prefix + "graph_ann.dot.svg"]
  kog.genCompilertgz(args.out_prefix + "compiler.tgz", fileList)

