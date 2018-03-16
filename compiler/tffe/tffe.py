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
import sys, json, re
import TfFrontEnd
import KgraphPartitions
import shutil
from MiscUtil import remove_write_permissions

kPath = os.environ.get('KAENA_PATH')
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
                    default=None)
parser.add_argument('--dot_timeout', help='Timeout for planarization of op and flow graphs in Graphviz, default 60 sec ',
                    type=int, default=60)
parser.add_argument('--scheduler', help='Select scheduler method tcc or wave, default is tcc',
                    default='tcc')
parser.add_argument('--batch', help='Batch override for late-binding networks',
                    type=int, default=1)
parser.add_argument('--partition', help='Partition into subgraphs; use from, meauto, or auto; the default is none',
                    nargs='+', default=["suppauto"])
parser.add_argument('--executors', help='Specifies executors per subgraph, e.g., tcc 1 2 3 (implies rest on host, host 0 4 5), default ""',
                    nargs='+', default=[])
parser.add_argument('--parallel_streams', help='run execution engines in parallel', action='store_true', default=False)
parser.add_argument('--preprocessor', help='Specify preprocessor script to be part of inference run', default="")
parser.add_argument('--postprocessor', help='Specify postprocessor script to be part of inference run', default="")

args = parser.parse_args()
inputNodeName = args.input_node

tfpbFile = args.tfpb
if not os.path.isfile(tfpbFile):
  raise("ERROR: missing --tfpb " + tfpbFile)
if args.images != None and args.focus != ".*":
  raise("ERROR: Unsupported --images with --focus")

def writeBackEndFiles(kGraph, outPrefix, verbose):
  fileList = []
  (refOutNpyFile, fileListJson) = kGraph.genCompilerJson("compiler.json", verbose)
  fileList += fileListJson
  fileList += [outPrefix + "graph_ann.dot.svg"]
  #kGraph.genCompilertgz(outPrefix + "compiler.tgz", list(set(fileList)))


debugLevel = args.debug
dotTimeout = args.dot_timeout
tffe = TfFrontEnd.TfFe(args.width, debugLevel, dotTimeout, args.batch)
tffe.loadPb(tfpbFile, args.focus)
kog = tffe.getKaenaOpGraph()
kog.setSchedulerMode(args.scheduler)
kog.writeDot(args.depth, args.out_prefix + "graph.dot", "svg")
ret = 0
if args.weights:
  tffe.writeWeights(args.out_prefix)
  
if args.images != None:
  tffe.writeImages(args.out_prefix, args.images, inputNodeName)

kog.identifyMainFlowEdges()
tffe.writeOpsCsv(args.out_prefix + "ops.csv")
kog.writeDot(args.depth, args.out_prefix + "graph_ann.dot", "svg")


kp = KgraphPartitions.KgraphPart(kog, debugLevel)
sgJsonList = []
kp.colorNodes(args.partition)
kp.partitionByColor()
kp.calcExecutorMap(args.executors)

sgId = 0
jsonFileOption = {"tcc" : "--json compiler.json",
                  "wave" : "--wavegraph wavegraph.json"}
parallelStreamsOption= ""
if args.parallel_streams:
    parallelStreamsOption = "--parallel_streams"
for sg in kp.getSubgraphs():
  sgDir = "sg%02d" % sgId;
  print("\nINFO: processing subgraph %s" % sgDir)
  sg.graph.print()
  os.makedirs(sgDir)
  os.chdir(sgDir)
  sg.addSideNodes(kog)
  sg.graph.levelize()
  sg.relinkNpFiles("..")
  sg.graph.setSchedulerMode(args.scheduler)
  sg.graph.print()
  sg.graph.writeDot(args.depth, args.out_prefix + "graph_ann.dot", "svg")
  
  executor = kp.getExecutorById(sgId)
  if not executor == 'host':
    writeBackEndFiles(sg.graph, args.out_prefix, args.verbose)
    meOk, fileList = sg.graph.runScheduler(args.out_prefix)
    if not executor == 'waveopt':
      if meOk:
        cmd = "%s/compiler/be/compiler/compiler.exe %s %s> log-be.txt 2>&1" % (kPath, jsonFileOption[args.scheduler], parallelStreamsOption)
        print("INFO: executing %s" % cmd, flush=True)
        ret = os.system(cmd)
      else:
        print("ERROR: skipping backend compilation since the mid scheduler failed", flush=True)

  os.chdir("..")
  sgJson = sg.genExecutorGraphJson(sgDir)
  sgJson["executor"] = executor
  sgJson["parallel_streams"] = False
  if args.scheduler == 'wave':
      sgJson["parallel_streams"] = args.parallel_streams
  sgJsonList.append(sgJson)
  sgId += 1

KgraphPartitions.attachPrePost(sgJsonList, args.preprocessor, args.postprocessor)

nnGraphFile = "nn_graph.json"
with open(nnGraphFile, "w") as f:
  s = json.dumps({"SubGraphs" : sgJsonList}, indent=2, sort_keys=True)
  f.write(s)

# K-elf compilation is done. Make output directory read/execute only
if debugLevel > 0:
  remove_write_permissions(".")

print("INFO: Kaena Compiler status %s" % ("PASS" if ret == 0 else "FAIL"))
sys.exit(0 if ret == 0 else 1)


