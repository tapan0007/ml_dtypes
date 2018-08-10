#!/usr/bin/env python3

# Run this script within test/e2e directory after a regression run.
# It looks into all test directories for compiler.json files and summarizes them.

import os, json
import yaml
#from sets import Set

def get_dirs():
    return [f for f in os.listdir("./") if os.path.isdir(f)]

merged_struct = {}
merged_data = {}
knode_common = {}
knode_dict = {}

struct_descr = {
        'layers' : 'A list of KNodes.',
        'data_type'  : 'Data type for the network (float16, bfloat16, uint8, uint16).',
        'net_name'  : 'Name of the network.',
        }

field_descr = {
        '#comment' : 'Comment describing the KNode type, not to be interpreted downstream.',
        'layer_type' : 'Type of the KNode, see list of Knode types.',
        'layer_name' : 'Name of the KNode, which will be used in previous_layers.',
        'previous_layers': 'A list of KNode names for KNodes that feed this KNode.',
        'ofmap_format': 'The shape format of the output tensor.',
        'ofmap_shape': 'The shape of the output tensor, specified as an array of dimensions.',
        'ref_file': 'Name of the numpy file that contains output tensor values.',
        'kernel_file': 'Name of the numpy file that contains weights tensor values.',
        'kernel_format': 'The shape format of the weights tensor.',
        'kernel_shape': 'The shape of the weights tensor, specified as an array of dimensions.',
        'padding': 'Padding for each dimension, an array of 2-elem arrays, each 2-elem array specify lower and upper padding.',
        'stride': 'Stride of convolution or pooling, specified as an array of strides for all dimensions.',
        'mul_scalar': 'Scalar for the multiplication operation.',
        'channel_slice': 'The bounds for a slice in the channel dimension, specified as a 2-elem array.',
        'next_layer_order': 'A list of 2-elem arrays, each 2-elem array specifies the order of unstack followed by the KNode name.',
        'unstack_axis': 'The axis of the unstack operation.',
        }

knode_descr = {
        }

def merge_data(json_data):
    global merged_data
    for m in json_data:
        merged_struct[m] = struct_descr[m]
    for l in json_data['layers']:
        if l['layer_type'] not in merged_data:
            merged_data[l['layer_type']] = l
            knode_dict[l['layer_type']] = l['#comment']
            if knode_common == {}:
                for f in l:
                    if f in field_descr:
                        descr = field_descr[f]
                    else:
                        raise RuntimeError(f + " doesn't have a description.")
                    knode_common[f] = descr + " Ex. " + str(l[f])
            else:
                for f in knode_common:
                    if f not in l:
                        del knode_common[f]

def remove_common():
    global merged_data
    for l in merged_data:
        if merged_data[l] is not knode_common:
            for f in knode_common:
                if f in merged_data[l] and f != '#comment':
                    del merged_data[l][f]

testdirs = get_dirs()

for d in testdirs:
    os.chdir(d)
    sgdirs = get_dirs()
    for d in sgdirs:
        compiler_json_fn = d+"/compiler.json"
        if os.path.exists(compiler_json_fn):
            try:
                json_data = json.load(open(compiler_json_fn))
            except Exception as e:               
                print(e)
                exit(-1)
            merge_data(json_data)
    os.chdir("../")

remove_common()

key_list = list(merged_data.keys())

for l in merged_data:
    for f in merged_data[l]:
        if f in field_descr:
            descr = field_descr[f]
        else:
            raise RuntimeError(f + " doesn't have a description.")
        if f != '#comment':
            merged_data[l][f] = descr + " Ex. " + str(merged_data[l][f])

print("--- # Top structure is a dictionary with following fields:")
print(yaml.dump(struct_descr, default_flow_style=False))
#print("--- # The following are a list of KNode types:")
#print(yaml.dump(knode_dict, default_flow_style=False))
print("--- # The following are fields common to all KNodes:")
print(yaml.dump(knode_common, default_flow_style=False))
print("--- # The KNodes with fields that are not in common list:")
print(yaml.dump(merged_data, default_flow_style=False))
