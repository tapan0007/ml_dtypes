#!/usr/bin/env python3

import json
import os
import re
import argparse
from layeropt_utils import CircbufPtrs
from layeropt import KGraph, KNode

# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavegraph", help="Wave-graph Json file to read")
    parser.add_argument("--typefilter", default=".*", help="Filter on waveop type")
    parser.add_argument("--fieldfilter", default=".*", help="Filter on waveop field name")
    args = parser.parse_args()

    # test by reading it back
    try:
        print("Test by loading Wave-graph %s"%args.wavegraph)
        wavegraph_json = json.load(open(args.wavegraph))
    except Exception as e:
        print(e)
        exit(-1)

    # create graph from JSON file        
    wavegraph = KGraph()
    wavegraph.populate_from_kgraph_json(wavegraph_json)

    # check for SBAtomFile nodes with no input
    for i in wavegraph.node_dict:
        entry = wavegraph.node_dict[i]
        if 'waveop_type' in entry.data:
            if re.search(args.typefilter, entry.data['waveop_type']):
                print("%d: %s"%(entry.node_number, entry.data['waveop_name']), end=" ")
                for j in entry.data:
                    if re.search(args.fieldfilter, j):
                        print(j + "=", end="")
                        print(entry.data[j], end=" ")
                print("")
