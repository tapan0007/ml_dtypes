"""
Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
"""

"""Middle Scheduler (aka Middle-End) v2
"""

import json
import os
import re
import numpy as np
import argparse
from me_utils import *
from me_graph import *
from me_models import *
from me_batch import *
from graphviz import Digraph

DEBUG_LEVEL_DEFAULT=2

#np.set_printoptions(precision=14)

"""State buffer memory manager
"""
class StateBuffer:

    SB_NUM_PARTITIONS = 128
    #SB_ATOM_SZ = 1024
    SB_ATOM_SZ = 2048   # For FP32, use this to guarantee gapless spaces for 28x28 (without using skip-atoms), when folding is involved
    #SB_ATOM_SZ = 4096
    SB_PARTITION_SZ = 96*1024# 96KB per partition
    SB_NUM_1K_ATOMS = SB_PARTITION_SZ//SB_ATOM_SZ
    SB_NUM_64B_MORSELS = SB_PARTITION_SZ // 64

    def __init__(self, batcher):
        self.batcher = batcher
        self.file_mapper = FileMapper(self.SB_PARTITION_SZ, self.batcher.data_type)
        self.zero_bias_file_params = None
        self.next_bias_file_start = 0
        self.next_nonbias_file_start = 0
        self.next_weights_file_start = 0
        self.printed_map_trace_header = False

    """Create constants file:
    kaena-452: create a file of zeros for use with Activation instruction without BiasAdd
    Format matches existing bias formats, but it should be more like CRSM to match weights
    """
    def create_constants_file(self, op, args):
        bias_shape_dims = ShapeDims("NCHW", [1, PEArray.NUM_ROWS, 1, 1])           
        bias_file_params = FileParams(
                                    file_name       = op.data['ref_file'].replace(".npy", "-constants.npy"),
                                    file_dims       = bias_shape_dims, 
                                    data_type       = self.batcher.data_type,
                                    op_params       = op,
                                    args            = args,
                                    contain_weights = True)
        bias_file_params.layer_name = op.data['layer_name']
        bias_file_params.load_file()
        bias_file_start_addr = self.next_bias_file_start
        bias_file_sz = align_addr_8B(self.batcher.item_sz)
        self.file_mapper.map_file(bias_file_params, bias_file_start_addr, wrap_around=False, region_sz=bias_file_sz)
        self.next_bias_file_start = align_addr_8B(self.next_bias_file_start + bias_file_sz)
        self.zero_bias_file_params = bias_file_params


"""Stream of waveops: consist of list of waveops that are fused (communicate through PSUM buffers)
"""
class WaveopStream(list):

    def __init__(self):
        self.last_main_waveop = None    # main stream waveop (PEArray resource)
        self.last_main_using_psum_bank = 0    # last main waveop using PSUM bank
        self.last_psum_waveop = [None for i in range(PEArray.PSUM_NUM_BANKS)]   # PSUM streams (PSUM resouce)
        self.waveop_name_set = set()
        self.waveop_count = 0
        self.nonload_waveop_count = 0
        self.nonload_waveop_list = []

    def append_check(self, item):
        item_name = item['waveop_name']
        i = 0
        if args.abstract_mem and item_name in self.waveop_name_set:
            return
        else:
            new_name = item_name
            while (new_name in self.waveop_name_set):
                new_name = item_name + "__" + str(i)
                print("WARNING: waveop_name %s exists; so modifying name to %s before adding waveop to stream"%(item_name, new_name))
                i += 1
            item_name = new_name
        item['waveop_name'] = item_name
        self.waveop_name_set.add(item['waveop_name'])                
        self.append(item)
        self.waveop_count += 1
        if (item['waveop_type'] != 'SBAtomLoad'):
            if (args.debug > 3): print("INFO: Adding nonload waveop %s ID %d"%(item['waveop_name'], self.nonload_waveop_count))
            self.nonload_waveop_list.append(item)
            self.nonload_waveop_count += 1

    def add_linked(self, waveop, side_waveops, psum_bank):
        input_list = []
        for i in side_waveops:
            self.append_check(i)
            input_list.append(i['waveop_name'])
        if (psum_bank < 0):
            if (self.last_main_waveop != None):
                input_list.append(self.last_main_waveop['waveop_name'])
        else:                
            if (self.last_psum_waveop[psum_bank] != None and waveop['waveop_type'] != "MatMul"):
                input_list.append(self.last_psum_waveop[psum_bank]['waveop_name'])
            elif (self.last_main_waveop != None):
                input_list.append(self.last_main_waveop['waveop_name'])
                if (self.last_main_using_psum_bank != psum_bank):
                    if (self.last_psum_waveop[psum_bank] != None):
                        input_list.append(self.last_psum_waveop[psum_bank]['waveop_name'])
        for i in input_list:                        
            if i not in waveop['previous_waveops']:
                waveop['previous_waveops'].append(i)
        self.append_check(waveop)
        if (psum_bank < 0):
            self.last_main_waveop = waveop
            self.last_main_using_psum_bank = psum_bank
        else:            
            self.last_psum_waveop[psum_bank] = waveop
            if (waveop['waveop_type'] == "MatMul"):
                self.last_main_waveop = waveop
                self.last_main_using_psum_bank = psum_bank

    def add_outputs(self, waveops):
        for i in waveops:
            self.append_check(i)

"""FusedOpList: a list of fused-op
"""
class FusedOpList(list):
    def __init__(self):
        pass

"""The TPB scheduler manages scheduling of execution for all the different engines
"""
class TPBSched:
    def __init__(self, batcher):
        self.pearray = PEArray()
        self.pool = Pool()
        self.activate = BiasAddAct()
        self.statebuffer = StateBuffer(batcher)
        self.waveop_stream = WaveopStream()
        self.num_mismatches = 0
        self.fused_ops_list = FusedOpList()

    """ Get a linearized list of fused ops and keep it locally
    """
    def get_list_of_fused_ops(self, kgraph):
        fused_op_count = 0
        prev_join_batch_item_partition_usage_sz = -1
        prev_join_op_list = None
        last_pairup_batch_count = 1
        while (not kgraph.walk_ended()
                and (args.stop_after_layer_num == 0 or fused_op_count <= args.stop_after_layer_num)):
            op_list = kgraph.get_fused_ops(fused_op_count)
            if (args.stop_after_layer_num > 0 and fused_op_count == args.stop_after_layer_num):
                op_list[-1].next = []

            # get the result file for the fused operation
            last_op = op_list[-1]

            # Check the first op of fused op
            first_op = op_list[0]
            first_op_type = first_op.data['layer_type'] 
            # Dissolve Input of Placeholder types
            if first_op.is_placeholder:
                assert(len(op_list) == 1)
                first_op.result_file = first_op.data['ref_file']
                # Populate OFMAP params                        
                first_op.populate_ofmaps_file_params()
                # Treat the placeholder like a fork since there maybe multiple inputs 
                prev_join_batch_item_partition_usage_sz = first_op.ofmaps_file_params.batch_item_partition_usage_sz
                first_op.ofmaps_file_params.input_layer_ifmap = True
                # Mark the first node after Input/Placeholder as input node
                for i in first_op.next: 
                    i.is_input = True
                    if (i.data['layer_type'] == 'Conv' or i.data['layer_type'] == 'MatMul' or i.data['layer_type'] == 'Softmax2'):
                        # compute early to detect replication and to obtain some params; will be recomputed in get_fused_ops
                        i.populate_conv_params()
                        i.populate_common_params(False)
                        # IFMAP replication
                        if i.repl_multiple_of_C > 1 and not i.ifmaps_padded_and_split:
                            (file_name, new_shape) = pad_and_split_file(
                                                        first_op.data['ref_file'], 
                                                        first_op.data['ofmap_format'],
                                                        i.stride.x,
                                                        i.padWN.x, i.padES.x,
                                                        i.padWN.y, i.padES.y)
                            first_op.data['ref_file'] = file_name
                            # fake the shape to make convolution work
                            [N, C, H, W] = new_shape
                            first_op.data['ofmap_shape'] = [N, C, H//i.stride.x, W*i.stride.x]
                            #first_op.data['ofmap_shape'] = new_shape
                            # clear padding info after modifying IFMAP
                            i.data['padding'][2] = [0, 0]
                            i.data['padding'][3] = [0, 0]
                            i.ifmaps_padded_and_split = True
                            print("INFO: Pad and split input FMAPs due to replication")
                            # Populate OFMAP params                        
                            first_op.populate_ofmaps_file_params()
            # Dissolve Reshape
            elif first_op.is_nop:
                for j in first_op.prev:
                    if j.result_file is not None:
                        first_op.result_file = j.result_file
                        first_op.populate_ofmaps_file_params()
                        first_op.ofmaps_file_params = j.ofmaps_file_params
                        break
            else:       
                # kaena-452: create a file of zeros for use with Activation instruction without BiasAdd
                # Format matches existing bias formats, but it should be more like CRSM to match weights
                tpb.statebuffer.create_constants_file(first_op, args)
                # maintain a linked-list of fused_ops
                # grab the current batch count from previous fused_ops
                if len(self.fused_ops_list) > 0:
                    op_list.prev = self.fused_ops_list[-1]
                    op_list.current_batch_count = op_list.prev.next_batch_count
                self.fused_ops_list.append(op_list)
                # make sure minimum batch count = Tn
                op_list.current_batch_count = max(op_list.current_batch_count, op_list.last_op.Tn)
                # set result file
                op_list.last_op.result_file = last_op.ofmaps_file_params.file_name
                print("Output file for layer %s is %s"%(op_list.last_op.data['layer_name'], op_list.last_op.result_file))
                # Check for convenient location to pair-up batch items for processing together
                op_list.partial_batch_pairup = False
                op_list.next_batch_count = op_list.current_batch_count
                if op_list.has_join or op_list.has_pool:
                    if prev_join_batch_item_partition_usage_sz > op_list.last_op.ofmaps_file_params.batch_item_partition_usage_sz:
                        op_list.partial_batch_pairup = True
                        op_list.residue_in_scratch = False
                        #if last_pairup_batch_count*2 <= op_list.last_op.ofmaps_file_params.file_dims.N:
                        # TODO: support higher batch count for smaller data sizes
                        op_list.next_batch_count = min(16, last_pairup_batch_count * 2) # can only support batch up to 16 in this scheduler
                        last_pairup_batch_count = op_list.next_batch_count
                        # Mark all fused ops between last join and this join as "pre-pair-up" region
                        if prev_join_op_list is not None:
                            backtrack_op_list = self.fused_ops_list[-1]
                            while (backtrack_op_list != prev_join_op_list):
                                backtrack_op_list.partial_batch_pre_pairup = True
                                backtrack_op_list.residue_in_scratch = False
                                # Also mark the convolution in the other branch (that is not fused with join, and that has the same FMAP*C size) as "pair-up"
                                if (backtrack_op_list.last_op.ofmaps_file_params.file_dims.tot_elems == op_list.last_op.ofmaps_file_params.file_dims.tot_elems):
                                    backtrack_op_list.partial_batch_pairup = True
                                    backtrack_op_list.residue_in_scratch = False
                                    backtrack_op_list.next_batch_count = op_list.next_batch_count
                                backtrack_op_list = backtrack_op_list.prev
                    elif prev_join_batch_item_partition_usage_sz < op_list.last_op.ofmaps_file_params.batch_item_partition_usage_sz \
                            or (prev_join_op_list is not None and prev_join_op_list.residue_in_scratch):
                        # special case for stage after MaxPool where OFMAP residue size increases: use scratch space for OFMAP instead of residue space
                        op_list.residue_in_scratch = True
                        # Also mark all the convolutions in both branches between current join and last join/fork
                        if prev_join_op_list is not None:
                            backtrack_op_list = self.fused_ops_list[-1]
                            while (backtrack_op_list != prev_join_op_list):
                                # here we just check that the image (FMAP) sizes are the same without considering number of channels (C) 
                                if (backtrack_op_list.last_op.ofmaps_file_params.fmap_data_len == op_list.last_op.ofmaps_file_params.fmap_data_len):
                                    backtrack_op_list.residue_in_scratch = True
                                backtrack_op_list = backtrack_op_list.prev
                    prev_join_op_list = self.fused_ops_list[-1]
                    prev_join_batch_item_partition_usage_sz = op_list.last_op.ofmaps_file_params.batch_item_partition_usage_sz

            #print("Fused op #%d, fmap data len %d"%(fused_op_count, op_list.last_op.ofmaps_file_params.fmap_data_len))                

            # Mark the last node of the fused-op as output node
            if last_op.next == []:
                last_op.ofmaps_file_params.final_layer_ofmap = True
                print("Fused op %s is output, mark final_layer_ofmap=True"%(last_op.data["layer_name"]))
                # If the first operation of current fused-op is a NOP
                # propagate these flags back to the previous op so that data can be saved to file
                if first_op.is_nop:
                    for j in first_op.prev:
                        j.ofmaps_file_params.final_layer_ofmap = True
                        print("NOP is output, mark previous %s final_layer_ofmap=True"%(j.data["layer_name"]))

            # kaena-643: pad sizes to 8B to satisfy HW 8B alignment requirement
            # Only for weights/bias, input IFMAP and final OFMAP.
            # Internal layers will gang-up pairs of chunks (FP16) to satisfy 4B alignment requirement.
            last_op.ofmaps_file_params.compute_padded_sizes()

            # increment count of fused ops (for ID purpose)
            fused_op_count += 1

    """ Fix order of first 2 convolutions right after a fork
    After a fork, if one branch directly goes to ResAdd, that branch was already deleted.
    If one branch has one convolution before ResAdd ("short" branch), that conv would need to 
    write to new residue space, which overlaps the old residue space in upper half. 
    To keep old residue space intact, we need to execute the first conv in the "long" branch 
    before executing the conv in the "short" branch.
    """
    def fix_order_1st_convs_after_fork(self):
        if len(self.fused_ops_list) >= 3:
            for i in range(1, len(self.fused_ops_list)-1):
                op_list = self.fused_ops_list[i]
                op_list_next = self.fused_ops_list[i+1]
                # if two consecutive fused-ops share the same source, and source is fork, then swap fused-ops orders
                # (except for the stage after MaxPool, where residue and scratch regions are reversed, and the following stage)
                if op_list.first_op.prev[0] == op_list_next.first_op.prev[0] \
                        and op_list.first_op.prev[0].is_fork \
                        and not op_list.residue_in_scratch \
                        and not op_list.prev.residue_in_scratch:
                    print("Swapping order between fused-op ID %d (%s) and fused-op ID %d (%s)"%(op_list.fused_op_id, op_list.last_op.data['layer_name'], op_list_next.fused_op_id, op_list_next.last_op.data['layer_name']))
                    self.fused_ops_list[i], self.fused_ops_list[i+1] = self.fused_ops_list[i+1], self.fused_ops_list[i]
                    self.fused_ops_list[i].prev = self.fused_ops_list[i-1]
                    self.fused_ops_list[i+1].prev = self.fused_ops_list[i]

    """Execute fused operations with batching
    """
    def execute_fused_ops_w_batching(self):
        if len(self.fused_ops_list) == 0:
            raise RuntimeError("No fused ops found; please check input JSON")
        # Reevaluate batch set selection based on first FMAP size (for non-ResNet50 exec)
        tpb.statebuffer.batcher.reevaluate_set_select(
                self.fused_ops_list[0].first_op.ifmaps_file_params.batch_item_partition_usage_sz)
        batch_count = self.fused_ops_list[0].first_op.ofmaps_file_params.file_dims.N
        current_Tn = self.fused_ops_list[0].first_op.Tn
        first_Tn = current_Tn
        b = current_Tn-1
        while b < batch_count:
            i = 0
            while i < len(self.fused_ops_list):
                op_list = self.fused_ops_list[i]
                current_Tn = op_list.first_op.Tn
                capped_current_batch_count = min(batch_count, op_list.current_batch_count)
                capped_next_batch_count = min(batch_count, op_list.next_batch_count)
                assert(capped_next_batch_count >= capped_current_batch_count)
                if (capped_current_batch_count < current_Tn):
                    raise RuntimeError("Please use --force_batch_count to at %d (or higher powers of 2) to simulate batching in middle of network"%(current_Tn))
                for j in range(capped_current_batch_count-1, -1, -current_Tn):
                    if (args.debug > 2): print("TRACE: executing fused op %s, batch elem %d to %d, partial_batch_pre_pairup %d, partial_batch_pairup %d, has_join %d, has_pool %d"%(op_list.last_op.data['layer_name'], b - j, b - j + current_Tn - 1, op_list.partial_batch_pre_pairup, op_list.partial_batch_pairup, op_list.has_join, op_list.has_pool))
                    op_list.map_files(tpb, b - j)
                    if not args.run_malloc_only:
                        op_list.execute(tpb, b - j)
                # kaena-409: the marker must be qualified with the condition that the fused-op contains a join or fork, 
                # because the marker is set for both branches before the join 
                # (the fork condition also must be considered for the first MaxPool, since we double-up there too).
                if op_list.partial_batch_pairup and (op_list.has_join or op_list.has_pool) and i != len(self.fused_ops_list)-1:
                    if (b % capped_next_batch_count) == (capped_next_batch_count - 1):
                        if (args.debug > 2): print("TRACE: batch element %d is last of the next partial-batch group (count %d), continuing to next pairup location"%(b, capped_next_batch_count))
                        i += 1                    
                    else:
                        i = 0
                        b += first_Tn
                        if (args.debug > 2): print("TRACE: go back to beginning for batch element %d"%(b))
                else:                    
                    i += 1                    
            b += current_Tn

    """Write out wavegraph
    """
    def write_wavegraph(self):
        wavegraph_json = kgraph_json
        if (args.wavegraph != None and args.inference == False): 
            wavegraph_json['waveops'] = tpb.waveop_stream
            try:
                print("Saving Wave-Graph %s"%args.wavegraph)
                with (open(args.wavegraph, 'w')) as f:
                    s = json.dumps(wavegraph_json, indent=2, sort_keys=True)
                    s = re.sub(r'\s+(\d+,)\n\s+(\d+)', r'\1\2', s, flags=re.S)
                    s = re.sub(r',\s*(\d+)\n\s+\]', r',\1]', s, flags=re.S)
                    f.write(s)
            except Exception as e:
                print(e)
                exit(-1)

            # test by reading it back
            try:
                print("Test by loading Wave-graph %s"%args.wavegraph)
                wavegraph_json = json.load(open(args.wavegraph))
            except Exception as e:
                print(e)
                exit(-1)

            # create graph from JSON file        
            wavegraph = KGraph(args)
            wavegraph.populate_from_kgraph_json(wavegraph_json)

            # check for SBAtomLoad nodes with no input
            if (args.debug > 2):
                node_has_output_edge = {}
                print("DBG: check for all SBAtomLoad nodes with no input")
                for i in wavegraph.node_dict:
                    entry = wavegraph.node_dict[i]
                    if 'waveop_type' in entry.data:
                        if entry.data['waveop_type'] == "SBAtomLoad":
                            if entry.data['previous_waveops'] == []:
                                print(entry.data['waveop_name'])
                        if entry.data['waveop_type'] != "SBAtomSave":
                            node_has_output_edge[entry.data['waveop_name']] = False
                print("DBG: check for all non-SBAtomSave nodes with no output")
                for i in wavegraph.node_dict:
                    entry = wavegraph.node_dict[i]
                    if 'previous_waveops' in entry.data:
                        for j in entry.data['previous_waveops']:
                            node_has_output_edge[j] = True
                for i in node_has_output_edge:
                    if not node_has_output_edge[i]:
                        raise RuntimeError("There's no output edge for node %s"%i)

        # write out dot graph in SVG format
        if (args.dot != None and args.inference == False):            
            (dotfile_root, dotfile_ext) = os.path.splitext(args.dot)                
            if (dotfile_ext == '.plain'):
                f = open(args.dot, 'w')
                f.write("digraph {\n")
                for i in tpb.waveop_stream:
                    f.write("\"%s\" [label=\"%s\"]\n"%(i['waveop_name'], i['waveop_name']))
                for i in tpb.waveop_stream:
                    for j in i['previous_waveops']:
                        f.write("\"%s\" -> \"%s\"\n"%(j, i['waveop_name']))
                f.write("}")
                f.close()
            else:
                dot = Digraph()
                for i in tpb.waveop_stream:
                    dot.node(i['waveop_name'], i['waveop_name'])
                    for j in i['previous_waveops']:
                        dot.edge(j, i['waveop_name'])
                dot.format = dotfile_ext[1:]
                dot.render(dotfile_root)
            print("INFO: Wrote " + args.dot)

"""Main program
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kgraph", default="compiler.json", help="K-graph Json file to read")
    parser.add_argument("--wavegraph", default="wavegraph.json", help="Wave-graph Json file to write")
    parser.add_argument("--dot", help="Dot file to write")
    parser.add_argument("--nname", default="resnet50", help="Network name, resnet50 or lm")
    parser.add_argument("--debug", type=int, default=DEBUG_LEVEL_DEFAULT, help="Debug level")
    parser.add_argument("--eigenlib_stride", action='store_true', help="Use Eigenlib style of striding starting in the center (-1) of striding window")
    parser.add_argument("--golden_inputs", action='store_true', help="Use golden files as inputs for each layer")
    parser.add_argument("--dump_pearray_inputs", type=int, default=0, help="Dump PEArray inputs for N number of waves")
    parser.add_argument("--save_layer_output", action='store_true', help="Save intermediate layer output into files")
    parser.add_argument("--abstract_mem", action='store_true', help="Keep data chunks as abstract objects")
    parser.add_argument("--no_inter_layer_load", action='store_true', help="Don't allow inter-layer loads")
    parser.add_argument("--stop_after_layer_num", type=int, default=0, help="Stop execution after fused op number. 0 means execute all fused ops. 1 means execute 1 fused op after Input. If there's a fork, there will be two outputs.")
    parser.add_argument("--inference", action='store_true', help="Inference mode: don't write intermediate -midout.npy and -ones.npy, except for the last -midout.npy")
    parser.add_argument("--enable_replication", action='store_true', help="Enable replication for cases where number of FMAP channels is lower than PEArray rows")
    parser.add_argument("--force_batch_count", type=int, default=1, help="Force batch count number to a certain value, to simulate batched execution in middle of network")
    parser.add_argument("--verify_output_only", action='store_true', help="Verify only the output; disable intermediate FMAP verifications in order to speed up compiler time")
    parser.add_argument("--relax_dependencies", action='store_true', help="To prevent running out of events (kaena-403,449), this option when true would relax the dependency requirement (kaena-411)")
    parser.add_argument("--run_malloc_only", action='store_true', help="Run through memory allocation only; skipping verification")
    args = parser.parse_args()

    print("Middle Sched v2: Running in %s mode"%(args.nname))

    if (args.debug > 5): np.set_printoptions(threshold=np.nan)

    # loading Kgraph
    try:
        print("\nLoading K-graph %s"%args.kgraph)
        kgraph_json = json.load(open(args.kgraph))
    except Exception as e:
        print(e)
        exit(-1)

    # create graph from JSON file        
    kgraph = KGraph(args)
    kgraph.populate_from_kgraph_json(kgraph_json)
    # add forward references
    kgraph.add_forward_refs(kgraph.final_nodes)

    # instantiate TPB scheduler
    tpb = TPBSched(BatchSBDataMap(16, kgraph.data_type))

    # obtain full list of fused ops in first pass
    tpb.get_list_of_fused_ops(kgraph)   

    # fix order of first two conv after fork
    tpb.fix_order_1st_convs_after_fork()

    # Execute fused ops with advanced batching
    tpb.execute_fused_ops_w_batching()

    # write out wavegraph           
    tpb.write_wavegraph()
    
    # check for comparison errors
    if (tpb.num_mismatches > 0):
        print("\nFAILED (num mismatches %d)"%tpb.num_mismatches)
    else:        
        print("\nPASSED")
