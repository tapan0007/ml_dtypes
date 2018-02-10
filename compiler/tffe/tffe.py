#!/usr/bin/env python3

# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Top level Tensorflow to Tonga compiler interface
#
# Examples
#
#   set tfpb =  $KAENA_PATH/../Kaena-external-opensource/apps/tf/resnet_v2_152/pb/resnet_v2_152_fp32.pb
#
#   python3 $KAENA_PATH/compiler/tffe/tffe.py --tfpb $tfpb --depth 5
#
#   python3 $KAENA_PATH/compiler/tffe/tffe.py --tfpb $tfpb --depth 5 --weights
#
#   python3 $KAENA_PATH/compiler/tffe/tffe.py --tfpb $tfpb --depth 5 --focus block3/unit_18
#
#   set img = $KAENA_PATH/../Kaena-external-opensource/apps/tf/resnet_v2_152/dog.jpg
#   $KAENA_PATH/compiler/tffe//tffe.py --tfpb $tfpb --depth 5 --images $img
#     firefox out_graph_ann.dot.svg
#     soffice  out_ops.csv  
#
#   See the Makefile for examples of running units tests


import argparse
import os.path
import sys
import TfFrontEnd
import KgraphPartitions

print("\nINFO: started as  ", " ".join(sys.argv))

parser = argparse.ArgumentParser()
parser.add_argument('--tfpb', help='TensorFlow freeze graph file', default="f.pb")
parser.add_argument('--out_prefix', help='Prefix for output files', default="out_")
parser.add_argument('--weights', help='Generate weight files', dest="weights",
                    action="store_true",  default=False)
parser.add_argument('--images', help='Generate images (IFMAP and OFMAP files) for an input image', dest="images", default=None)
parser.add_argument('--depth', help='Depth of layer name hierarchy to show in the dot output',
                    type=int, default=5)
parser.add_argument('--debug', help='Debug level, 1 minimal, 3 detailed op values ',
                    type=int, default=0)
parser.add_argument('--focus', help='Regular expression to filter a subset of nodes',
                    default=".*")
parser.add_argument('--width', help='Highlight data paths wider than the width',
                    type=int, default=1000)
parser.add_argument('--verbose', help='Verbosity level, 0 default, 1 shows in/outputs, 2 TBD',
                    type=int, default=0)
parser.add_argument('--input_node', help='Input node in the neural network graph (where --images should be injected during calibration)',
                    default="input")
parser.add_argument('--dot_timeout', help='Timeout for planarization of op and flow graphs in Graphviz, default 60 sec ',
                    type=int, default=60)
parser.add_argument('--scheduler', help='Select scheduler method tcc or wave, default is tcc',
                    default='tcc')
parser.add_argument('--batch', help='Batch override for late-binding networks',
                    type=int, default=1)
parser.add_argument('--partition', help='Partition into subgraphs; use fromOpRe toOpRe or auto; the default is none',
                    nargs='+', default=["none"])

args = parser.parse_args()
inputTensorName = args.input_node

file = args.tfpb
if not os.path.isfile(file):
  raise("ERROR: missing --tfpb " + file)
if args.images != None and args.focus != ".*":
  raise("ERROR: Unsupported --images with --focus")

debugLevel = args.debug
dotTimeout = args.dot_timeout
tffe = TfFrontEnd.TfFe(args.width, debugLevel, dotTimeout, args.scheduler, args.batch)
tffe.loadPb(file, args.focus)
tffe.writeDot(args.depth, args.out_prefix + "graph.dot", "svg")
if args.weights:
  tffe.writeWeights(args.out_prefix)
kog = tffe.getKaenaOpGraph()
if args.images != None:
  tffe.writeImages(args.out_prefix, args.images, inputTensorName)
  kog.identifyMainFlowEdges(inputTensorName)
  tffe.writeOpsCsv(args.out_prefix + "ops.csv")
  tffe.writeDot(args.depth, args.out_prefix + "graph_ann.dot", "svg")
  if args.partition[0] == "none":
    fileList = []
    (refOutNpyFile, fileListJson) = kog.genCompilerJson(args.out_prefix + "compiler.json", args.verbose)
    fileList += fileListJson
    jsonFile = {"tcc" : "compiler.json", "wave" : "wavegraph.json"}
    fileList += kog.genKgraphSetupFiles(args.out_prefix + "compiler.py", args.out_prefix + jsonFile[args.scheduler], refOutNpyFile)
    fileList += [args.out_prefix + "graph_ann.dot.svg"]
    fileList += tffe.runScheduler(args.out_prefix)
    kog.genCompilertgz(args.out_prefix + "compiler.tgz", list(set(fileList)))
  else:
    kp = KgraphPartitions.KgraphPart(kog, debugLevel)
    if args.partition[0] == "auto":
      kp.autoColorNodes()
      kp.partitionByColor()
      kp.print()
      
