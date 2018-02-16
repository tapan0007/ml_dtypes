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
import sys, json
import TfFrontEnd
import KgraphPartitions

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
parser.add_argument('--partition', help='Partition into subgraphs; use fromOpRe toOpRe or auto; the default is none',
                    nargs='+', default=["none"])
parser.add_argument('--executors', help='Specifies executors per subgraph, e.g., tcc 1 2 3 (implies rest on host, host 0 4 5), default ""',
                    nargs='+', default=[])

args = parser.parse_args()
inputNodeName = args.input_node

tfpbFile = args.tfpb
if not os.path.isfile(tfpbFile):
  raise("ERROR: missing --tfpb " + tfpbFile)
if args.images != None and args.focus != ".*":
  raise("ERROR: Unsupported --images with --focus")

def writeBackEndFiles(kGraph, outPrefix, verbose, scheduler):
  fileList = []
  (refOutNpyFile, fileListJson) = kGraph.genCompilerJson(outPrefix + "compiler.json", verbose)
  fileList += fileListJson
  jsonFile = {"tcc" : "compiler.json", "wave" : "wavegraph.json"}
  fileList += kGraph.genKgraphSetupFiles(outPrefix + "compiler.py", outPrefix + jsonFile[scheduler], refOutNpyFile)
  fileList += [outPrefix + "graph_ann.dot.svg"]
  fileList += kGraph.runScheduler(outPrefix)
  kGraph.genCompilertgz(outPrefix + "compiler.tgz", list(set(fileList)))
  

debugLevel = args.debug
dotTimeout = args.dot_timeout
tffe = TfFrontEnd.TfFe(args.width, debugLevel, dotTimeout, args.batch)
tffe.loadPb(tfpbFile, args.focus)
kog = tffe.getKaenaOpGraph()
kog.setSchedulerMode(args.scheduler)
kog.writeDot(args.depth, args.out_prefix + "graph.dot", "svg")
ret = 1
if args.weights:
  tffe.writeWeights(args.out_prefix)
  
if args.images != None:
  tffe.writeImages(args.out_prefix, args.images, inputNodeName)
  kog.identifyMainFlowEdges()
  tffe.writeOpsCsv(args.out_prefix + "ops.csv")
  kog.writeDot(args.depth, args.out_prefix + "graph_ann.dot", "svg")
  if args.partition[0] == "none":
    writeBackEndFiles(kog, args.out_prefix, args.verbose, args.scheduler)
    cmd = "bash %s/compiler/be/test/RunOne.sh *tgz" % kPath
    print("INFO: executing %s" % cmd, flush=True)
    ret = os.system(cmd)
  else:
    kp = KgraphPartitions.KgraphPart(kog, debugLevel)
    executorsStr = " ".join(args.executors)
    sgJsonList = []
    kp.colorNodes(args.partition)
    kp.partitionByColor()
    #kp.print()
    sgId = 0
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
      try:
        writeBackEndFiles(sg.graph, args.out_prefix, args.verbose, args.scheduler)
      except:
        executorsStr += " host %d" % sgId
      os.chdir("..")
      sgJsonList.append(sg.genExecutorGraphJson(sgDir))
      sgId += 1
    nnGraphFile = "nn_graph.json"
    with open(nnGraphFile, "w") as f:
      s = json.dumps({"SubGraphs" : sgJsonList}, indent=2, sort_keys=True)
      f.write(s)
    cmd = "%s/runtime/util/nn_executor --nn_graph %s --tfpb %s --executors %s" % (
          kPath, nnGraphFile, tfpbFile, executorsStr)
    print("INFO: executing  %s" % cmd, flush=True)
    ret = os.system(cmd)
  print("INFO: Kaena flow status %s" % ("PASS" if ret == 0 else "FAIL"))
  sys.exit(0 if ret == 0 else 1)
