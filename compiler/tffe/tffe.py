# Top level Tensorflow to Tonga compiler interface

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

args = parser.parse_args()

file = args.tfpb
if not os.path.isfile(file):
  raise("ERROR: missing --tfpb " + file)

tffe = TfFrontEnd.TfFe()
tffe.loadPb(file)
tffe.writeDot(int(args.depth), args.out_prefix + "graph", "svg")
if args.weights:
  tffe.writeWeights(args.out_prefix)



