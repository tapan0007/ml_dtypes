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


import argparse
import os.path
import TfFrontEnd

parser = argparse.ArgumentParser()
parser.add_argument('--tfpb', help='TensorFlow freeze graph file', default="f.pb")
parser.add_argument('--out_prefix', help='Prefix for output files', default="out_")
parser.add_argument('--weights', help='Generate weight files', dest="weights",
                    action="store_true",  default=False)
parser.add_argument('--depth', help='Depth of layer name hierarchy to show in the dot output',
                    default=5)
parser.add_argument('--focus', help='Regular expression to filter a subset of nodes',
                    default=".*")

args = parser.parse_args()

file = args.tfpb
if not os.path.isfile(file):
  raise("ERROR: missing --tfpb " + file)

tffe = TfFrontEnd.TfFe()
tffe.loadPb(file, args.focus)
tffe.writeDot(int(args.depth), args.out_prefix + "graph.dot", "svg")
if args.weights:
  tffe.writeWeights(args.out_prefix)



