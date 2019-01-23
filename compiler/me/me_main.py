"""
Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
"""

"""Middle Scheduler (aka Middle-End) v2
"""

import json
import os
import re
import sys
import numpy as np
import argparse
import datetime
import me_wavegraph_cleanup
from enum import IntEnum
from me_utils import *
from me_graph import *
from me_models import *
from me_batch import *
from me_foldfile import *
from graphviz import Digraph

DEBUG_LEVEL_DEFAULT=2

#np.set_printoptions(precision=14)

"""Enum for engines used by dependency trackers
"""
class EngineEnum(IntEnum):
    PEARRAY = 0
    ACT = 1
    POOL = 2
    DMA = 3

"""State buffer memory manager
"""
class StateBuffer:

    SB_NUM_PARTITIONS = 128

    def __init__(self, batcher):
        self.batcher = batcher
        self.file_mapper = FileMapper(self.batcher.data_type, args)
        self.combined_bias_file_params = None
        self.next_bias_file_start = 0
        self.next_nonbias_file_start = 0
        self.next_weights_file_start = 0
        self.printed_map_trace_header = False

    """Advance next bias allocation pointer.
        Args: 
            - advance_bytes: number of bytes to advance next bias allocation pointer
    """
    def adv_next_bias_file_start(self, advance_bytes):
        self.next_bias_file_start = align_addr_8B(self.next_bias_file_start + advance_bytes)
        bias_region_sz = self.batcher.sb_bias_sz[0]
        if self.next_bias_file_start > bias_region_sz:
            self.next_bias_file_start = 0

"""Stream of waveops: consist of list of waveops generated during scheduling loop unrolling
"""
class WaveopStream(list):

    def __init__(self):
        self.last_main_waveop = None    # main stream waveop (PEArray resource)
        self.last_main_using_psum_bank = 0    # last main waveop using PSUM bank
        self.last_engine_waveop = [None for i in list(EngineEnum)] # engine streams 
        self.last_psum_waveop = [None for i in range(PEArray.PSUM_NUM_BANKS)]   # PSUM streams (PSUM resouce)
        self.waveop_name_set = set()
        self.waveop_count = 0

    """Append waveop to stream and change name if there's duplicate
        Args:
            item: waveop (dictionary of waveop fields and values) to add
            loc: specifies fixed location to insert; if None, append to end of stream 
        Returns:
            nothing
    """
    def append_check(self, item, loc = None):
        item_name = item['waveop_name']
        i = 0
        if args.abstract_mem and item_name in self.waveop_name_set:
            return
        else:
            new_name = item_name
            while (new_name in self.waveop_name_set):
                new_name = item_name + "__" + str(i)
                i += 1
            if new_name != item_name:
                print("WARNING: waveop_name %s exists; so modifying name to %s before adding waveop to stream"%(item_name, new_name))
            item_name = new_name
        item['waveop_name'] = item_name
        self.waveop_name_set.add(item['waveop_name'])                
        if (args.debug > 2): print("INFO SB TRACE: Adding waveop %s ID %d"%(item['waveop_name'], self.waveop_count))
        if (loc == None):
            self.append(item)
        else:
            self.insert(loc, item)
        self.waveop_count += 1

    """Insert dependencies for waveop and add it to stream 
        Args:
            waveop: dictionary of waveop fields and values
            side_waveops: additional waveops to insert ahead of waveop (ie DRAM loads); their names are added to waveop's list of predecessors (previous_waveops)
            psum_bank: PSUM bank ID the waveop is accessing
            new_reader_morsels: readers of elements in SB (morsels) are tracked in FileMapper using reader objects; they have tentatively assigned waveop ID but due to on-demand loads, final waveop ID is determined after all the load waveops are added, so the waveop ID in reader objects are adjusted here.
    """
    def add_linked(self, waveop, side_waveops, psum_bank, new_reader_morsels=[]):
        # Decode the engine from waveop_type            
        engine = EngineEnum.POOL
        if waveop['waveop_type'] == 'MatMul':
            engine = EngineEnum.PEARRAY
        elif waveop['waveop_type'] == 'Activation':
            engine = EngineEnum.ACT
        elif waveop['waveop_type'] == 'SBAtomSave':
            engine = EngineEnum.DMA
        elif waveop['waveop_type'] == 'SBAtomLoad':
            engine = EngineEnum.DMA

        # Add the side waveops (DRAM loads) and start a dependency list
        input_list = []
        for i in side_waveops:
            self.add_linked(i, [], -1)
            input_list.append(i['waveop_name'])
        
        # Handle engine dependency 
        # Limit to Activation for now, to avoid running out of events
        if self.last_engine_waveop[engine] is not None \
                and engine != EngineEnum.DMA:
                #and (engine == EngineEnum.ACT or engine == EngineEnum.POOL): 
            input_list.append(self.last_engine_waveop[engine]['waveop_name'])

        # Handle PSUM bank dependency
        if psum_bank >= 0 and self.last_psum_waveop[psum_bank] is not None:
            input_list.append(self.last_psum_waveop[psum_bank]['waveop_name'])

        # Filter out duplicates            
        for i in input_list:                        
            if i not in waveop['previous_waveops']:
                waveop['previous_waveops'].append(i)

        # Update SB reader morsels
        for i in new_reader_morsels:
            if (args.debug > 3): print("INFO SB TRACE: updating morsel reader ID for file ID %d chunk_id %d to ID %d (old %d)"%(i.file_id, i.chunk_id, self.waveop_count, i.accessor_id))
            i.accessor_id = self.waveop_count

        # Add waveop to stream
        self.append_check(waveop)

        # "Main" execution path: tracking mainly PEArray and other waveops not using PSUM
        # PSUM path: tracking last waveop using a particular PSUM bank
        # TODO: split into reader/writer
        if (psum_bank < 0):
            if engine != EngineEnum.DMA:  # Add to main path if no bank is used, except for loads/stores
                self.last_main_waveop = waveop
        else:            
            self.last_psum_waveop[psum_bank] = waveop
            if engine == EngineEnum.PEARRAY or engine == EngineEnum.POOL:
                self.last_main_waveop = waveop

        # Track last user of engine 
        # (only using Activation for now, until we switch to semaphore or have wavegraph-cleaner fully qualified)
        # TODO: split into reader/writer
        self.last_engine_waveop[engine] = waveop  

    """Add waveops that are outgoing from a previous waveop
    """
    def add_outputs(self, waveops, loc = None):
        for i in waveops:
            self.append_check(i, loc)

"""FusedOpList: a list of fused-ops
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
        print("INFO: Schedule fused operations")            
        fused_op_count = 0
        prev_join_batch_item_partition_usage_sz = -1
        prev_join_op_list = None
        last_pairup_batch_count = 1
        fused_ops_map = set({})
        while (not kgraph.walk_ended()
                and (args.stop_after_layer_num == 0 or fused_op_count <= args.stop_after_layer_num)):
           
            op_list = kgraph.get_fused_ops(fused_op_count, fused_ops_map)
            if (args.stop_after_layer_num > 0 and fused_op_count == args.stop_after_layer_num):
                op_list[-1].next = []

            # get the result file for the fused operation
            last_op = op_list[-1]

            # Check the first op of fused op
            first_op = op_list[0]
            first_op_type = first_op.data['layer_type'] 

            # kaena-452: create a file of zeros for use with Activation instruction without BiasAdd
            # Format matches existing bias formats, but it should be more like CRSM to match weights
            if kgraph.combined_bias_file_params is None:
                kgraph.create_constants_file(first_op)

            if first_op.is_placeholder:
                assert(len(op_list) == 1)
                first_op.result_avail = True
                # Populate OFMAP params                        
                first_op.populate_ofmaps_file_params()
                # Treat the placeholder like a fork since there maybe multiple inputs 
                prev_join_batch_item_partition_usage_sz = first_op.ofmaps_file_params.batch_item_partition_usage_sz
                first_op.ofmaps_file_params.input_layer_ifmap = True
                # Mark the first node after Input/Placeholder as input node
                for i in first_op.next: 
                    i.is_input = True
                    #if i.is_nop:
                    #    raise RuntimeError("Sorry, cannot have a data movement op (Reshape, Squeeze, ExpandDims) as the first node after placeholder")
                    if i.data['layer_type'] in {'Conv', 'QuantizedConv', 'MatMul', 'Softmax2'}:
                        # We support matmul with the non-constant second operand.
                        # Fixme: resiude_index is not set before entering populate_common_params..
                        skip_conv_populate = False
                        if (i.data['layer_type'] == 'MatMul') and i.is_join:
                            i.residue_index = 1
                            # All inputs must be visit and ofmaps have to be created!
                            skip_conv_populate = i.prev[0].ofmaps_file_params == None or i.prev[1].ofmaps_file_params == None
                            i.data['previous_layers'][1] == first_op.data['layer_name']

                        if skip_conv_populate == False:
                            # compute early to detect replication and to obtain some params; will be recomputed in get_fused_ops
                            i.populate_conv_params()
                            i.populate_common_params(False)
                            # IFMAP replication
                            if i.repl_multiple_of_C > 1 and not i.ifmaps_padded_and_split:
                                pad_const = i.data['zero_point_input'] if 'zero_point_input' in i.data else 0
                                (file_name, new_shape) = pad_and_split_file(
                                                            first_op.data['ref_file'], 
                                                            first_op.data['ofmap_format'],
                                                            i.stride.x,
                                                            i.padWN.x, i.padES.x,
                                                            i.padWN.y, i.padES.y,
                                                            pad_const=pad_const)
                                first_op.data['ref_file'] = file_name
                                # fake the shape to make convolution work
                                [N, C, H, W] = new_shape
                                first_op.data['ofmap_shape'] = [N, C, H//i.stride.x, W*i.stride.x]
                                #first_op.data['ofmap_shape'] = new_shape
                                # clear padding info after modifying IFMAP
                                i.data['padding'][2] = [0, 0]
                                i.data['padding'][3] = [0, 0]
                                i.ifmaps_padded_and_split = True
                                # Populate OFMAP params                        
                                first_op.populate_ofmaps_file_params()
                                first_op.ofmaps_file_params.input_layer_ifmap = True
                                first_op.ofmaps_file_params.compute_params(i.stride, args, repl_multiple_of_C = i.repl_multiple_of_C)
                                print("INFO: Pad and split input FMAPs due to replication, replication multiple %d, input_layer_ifmap %d"%(i.repl_multiple_of_C, first_op.ofmaps_file_params.input_layer_ifmap))
                    # Pool cannot decompose across chunks currently, so have to keep chunks contiguous (no gap/padding)
                    elif (i.data['layer_type'] == 'MaxPool' or i.data['layer_type'] == 'AvgPool'):
                        i.populate_common_params(False)
                        first_op.populate_ofmaps_file_params()
                        first_op.ofmaps_file_params.input_layer_ifmap = True
                        first_op.ofmaps_file_params.compute_params(i.stride, args, repl_multiple_of_C = 1, no_gap=True)
            elif first_op.data['layer_type'] == 'ConstLoad': 
                first_op.result_avail = True
                # Populate OFMAP params                        
                first_op.populate_ofmaps_file_params(const_tensor_load = True)
                # ReviewMe: Treat node with ConstLoad as graph input. Otherwise, map_files doesn't create dram load waveops.
                # It is related to the modify_in_place flag.
                for i in first_op.next: 
                    i.is_input = True
            else:       
                # maintain a linked-list of fused_ops
                # grab the current batch count from previous fused_ops
                if len(self.fused_ops_list) > 0:
                    op_list.prev = self.fused_ops_list[-1]
                    op_list.current_batch_count = op_list.prev.next_batch_count
                self.fused_ops_list.append(op_list)
                # make sure minimum batch count = Tn
                op_list.current_batch_count = max(op_list.current_batch_count, op_list.last_op.Tn)
                op_list.last_op.result_avail = True
                print("Output file for layer %s is %s"%(op_list.last_op.data['layer_name'], op_list.last_op.ofmaps_file_params.file_name))
                # Check for convenient location to pair-up batch items for processing together
                op_list.partial_batch_pairup = False
                op_list.next_batch_count = op_list.current_batch_count
                if op_list.has_join or op_list.has_pool:
                    if prev_join_batch_item_partition_usage_sz > op_list.last_op.ofmaps_file_params.batch_item_partition_usage_sz:
                        op_list.partial_batch_pairup = True
                        op_list.residue_in_scratch = False
                        #if last_pairup_batch_count*2 <= op_list.last_op.ofmaps_file_params.file_dims.N:
                        # TODO: support higher batch count for smaller data sizes
                        if args.nname == "resnet50":
                            # Generic scheduling should use the same number of batch count.
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

            # For pass-through ops, pad OFMAP the same way as IFMAP
            for i in op_list:
                if i.is_nop:
                    for j in first_op.prev:
                        if j.ofmaps_file_params.input_layer_ifmap:
                            i.ofmaps_file_params.input_layer_ifmap = True

            # Mark the last node of the fused-op as output node
            if last_op.next == []:
                last_op.ofmaps_file_params.final_layer_ofmap = True
                print("Fused op %s is output, mark final_layer_ofmap=True"%(last_op.data["layer_name"]))
                # If the fused operation has a join (ResAdd, Mult, etc), the joined FMAP may share
                # the same SB space with the fused operations that feed the join's inputs.
                # These fused operations need to be aware of the fact that they are writing to 
                # the final OFMAP space, which requires alignment padding.
                if op_list.has_join:
                    for i in op_list.last_op.ofmaps_file_params.writers_of_shared_fmap:
                        i.ofmaps_file_params.share_w_final_layer_ofmap = True
                        i.ofmaps_file_params.compute_padded_sizes()
                        print("Also for fused op %s sharing same output SB space, mark share_w_final_layer_ofmap=True"%(i.data["layer_name"]))
                # If the first operation of current fused-op is a NOP
                # propagate these flags back to the previous op so that data can be saved to file
                #if first_op.is_nop:
                #    for j in first_op.prev:
                #        j.ofmaps_file_params.final_layer_ofmap = True
                #        print("NOP is output, mark previous %s final_layer_ofmap=True"%(j.data["layer_name"]))

            # kaena-643: pad sizes to 8B to satisfy HW 8B alignment requirement
            # Only for weights/bias, input IFMAP and final OFMAP.
            # Internal layers will gang-up pairs of chunks (FP16) to satisfy 4B alignment requirement.
            last_op.ofmaps_file_params.compute_padded_sizes()

            # increment count of fused ops (for ID purpose)
            fused_op_count += 1

        kgraph.map_constants_file(file_mapper = tpb.statebuffer.file_mapper, region_sz = tpb.statebuffer.batcher.sb_bias_sz[0])
        print("INFO: Number of fused operations: ", fused_op_count)            

    """ Fix order of first 2 convolutions right after a fork
    After a fork, if one branch directly goes to ResAdd, that branch was already deleted.
    If one branch has one convolution before ResAdd ("short" branch), that conv would need to 
    write to new residue space, which overlaps the old residue space in upper half. 
    To keep old residue space intact, we need to execute the first conv in the "long" branch 
    before executing the conv in the "short" branch.
    """
    def fix_order_1st_convs_after_fork(self):
        if len(self.fused_ops_list) >= 3 and args.nname == 'resnet50':
            print("INFO: Fixing order of fused operations (ResNet50)")            
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

    """Execute fused operations with ResNet50 batching: (TODO: add simple batching) 
        1- If there's not enough batch items for a partial batch, go back to beginning of list and start processing next batch item
        2- Once there's a complete partial batch, process partial batch as far as possible down the list of fused-ops (until a double-up point)
            - Now partial batch size is doubled, there's not enough batch items to continue, so #1 condition now applies
    """
    def execute_fused_ops_w_batching(self):
        print("INFO: Starting execution of fused operations")            
        if len(self.fused_ops_list) == 0:
            raise RuntimeError("No fused ops found; please check input JSON")
        # Reevaluate batch set selection based on first FMAP size (for non-ResNet50 exec)
        tpb.statebuffer.batcher.reevaluate_set_select(
                self.fused_ops_list[0].first_op.ifmaps_file_params.batch_item_partition_usage_sz)
        batch_count = self.fused_ops_list[0].first_op.ofmaps_file_params.file_dims.N
        current_Tn = self.fused_ops_list[0].first_op.Tn
        first_Tn = current_Tn
        b = min(batch_count-1, current_Tn-1)
        live_mapped_file_params = []
        while b < batch_count:
            i = 0
            while i < len(self.fused_ops_list):
                op_list = self.fused_ops_list[i]
                current_Tn = op_list.first_op.Tn
                capped_current_batch_count = min(batch_count, op_list.current_batch_count)
                capped_next_batch_count = min(batch_count, op_list.next_batch_count)
                print("N: ", batch_count, " Tn: ", current_Tn, " current_batch_count: ", op_list.current_batch_count, " capped_current_batch_count: ", capped_current_batch_count, " next_batch_count: ",  op_list.next_batch_count, " capped_next_batch_count: ", capped_next_batch_count)
                assert(capped_next_batch_count >= capped_current_batch_count)
                assert(capped_next_batch_count >= current_Tn)
                assert(capped_current_batch_count >= current_Tn)
                if (capped_current_batch_count < current_Tn):
                    raise RuntimeError("Please use --force_batch_count to at %d (or higher powers of 2) to simulate batching in middle of network"%(current_Tn))
                for j in range(capped_current_batch_count-1, -1, -current_Tn):
                    if (args.debug > 2): print("TRACE: executing fused op %s (ID %d), batch elem %d to %d, partial_batch_pre_pairup %d, partial_batch_pairup %d, has_join %d, has_pool %d"%(op_list.last_op.data['layer_name'], op_list.fused_op_id, b - j, b - j + current_Tn - 1, op_list.partial_batch_pre_pairup, op_list.partial_batch_pairup, op_list.has_join, op_list.has_pool))
                    op_list.set_live_mapped_file_params(live_mapped_file_params)
                    op_list.map_files(
                        tpb = tpb, 
                        batch_item = b - j, 
                        live_mapped_file_params = live_mapped_file_params)
                    op_list.execute(tpb, b - j)

                if args.nname != "resnet50":
                    # Only free live mapped tensor after processing the last Tn group (sub-batch that fits in PSUM).
                    if b + current_Tn >= batch_count:
                        op_list.mark_ifmaps_are_consumed(live_mapped_file_params)
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
        wavegraph_json = {}
        wavegraph_json["data_type"] = kgraph_json["data_type"]
        wavegraph_json["net_name"] = kgraph_json["net_name"]
        if (args.wavegraph != None and args.inference == False): 
            wavegraph_json['waveops'] = tpb.waveop_stream
            if (args.enable_cleanup == True):
              b4_cleanup_name = args.wavegraph.replace(".json", "-b4clean.json")  
              if b4_cleanup_name == args.wavegraph:
                  b4_cleanup_name += "-b4clean"
              print("Saving Wave-Graph %s before cleanup"%b4_cleanup_name)
              with (open(b4_cleanup_name, 'w')) as f:
                  s = json.dumps(wavegraph_json, indent=2, sort_keys=True)
                  s = re.sub(r'\s+(\d+,)\n\s+(\d+)', r'\1\2', s, flags=re.S)
                  s = re.sub(r',\s*(\d+)\n\s+\]', r',\1]', s, flags=re.S)
                  f.write(s)
              wavegraph_json =\
                    me_wavegraph_cleanup.remove_redundant_edges(wavegraph_json)
            try:
                print("Saving Wave-Graph %s"%args.wavegraph)
                with (open(args.wavegraph, 'w')) as f:
                    s = json.dumps(wavegraph_json, indent=2, sort_keys=True)
                    s = re.sub(r'\s+(\d+,)\n\s+(\d+)', r'\1\2', s, flags=re.S)
                    s = re.sub(r',\s*(\d+)\n\s+\]', r',\1]', s, flags=re.S)
                    f.write(s)
            except Exception as e:
                print(e)
                sys.exit(-1)

            # test by reading it back
            try:
                print("Test by loading Wave-graph %s"%args.wavegraph)
                wavegraph_json = json.load(open(args.wavegraph))
            except Exception as e:
                print(e)
                sys.exit(-1)

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
    parser.add_argument("--kgraph", default="compiler.json", help="K-graph Json file to read; defaults to compiler.json")
    parser.add_argument("--wavegraph", default="wavegraph.json", help="Wave-graph Json file to write; defaults to wavegraph.json")
    parser.add_argument("--dot", help="Dot file to write")
    parser.add_argument("--nname", default="generic", help="Network name, resnet50 or generic")
    parser.add_argument("--debug", type=int, default=DEBUG_LEVEL_DEFAULT, help="Debug level")
    parser.add_argument("--eigenlib_stride", action='store_true', help="Use Eigenlib style of striding starting in the center (-1) of striding window")
    parser.add_argument("--golden_inputs", action='store_true', help="Use golden files as inputs for each layer")
    parser.add_argument("--dump_pearray_inputs", type=int, default=0, help="Dump PEArray inputs for N number of waves")
    parser.add_argument("--save_layer_output", nargs='?', type=int, const=100000, default=0, help="Save intermediate layers output into files. Default 0 means no intermediate dump. N means first N fused-op layers. -N means last N fused-op layers.")
    parser.add_argument("--save_layer_regex", default="", help="Save intermediate fused-op layers with names matching regex output files.")
    parser.add_argument("--abstract_mem", action='store_true', help="Keep data chunks as abstract objects")
    parser.add_argument("--no_inter_layer_load", action='store_true', help="Don't allow inter-layer loads")
    parser.add_argument("--stop_after_layer_num", type=int, default=0, help="Stop execution after fused op number. 0 means execute all fused ops. 1 means execute 1 fused op after Input. If there's a fork, there will be two outputs.")
    parser.add_argument("--inference", action='store_true', help="Inference mode: don't write intermediate -midout.npy and -ones.npy, except for the last -midout.npy")
    parser.add_argument("--enable_replication", action='store_true', help="Enable replication for cases where number of FMAP channels is lower than PEArray rows")
    parser.add_argument("--force_batch_count", type=int, default=1, help="Force batch count number to a certain value, to simulate batched execution in middle of network")
    parser.add_argument("--verify_output_only", action='store_true', help="Verify only the output; disable intermediate FMAP verifications in order to speed up compiler time")
    parser.add_argument("--no_verify", action='store_true', help="Disables FMAPs comparison")
    parser.add_argument("--enable_eviction", action='store_true', help="Enable eviction")
    parser.add_argument("--enable_cleanup", action='store_true', help="Enable wavegraph cleanup for event pressure reduction")
    parser.add_argument("--fuse_lrelu", action='store_true', help="Fuse the function max(y, a*y) into Lrelu activation function")
    parser.add_argument("--sb_partition_sz", type=int, default=96*1024-256, help="Size of one SB partition (to reserve space at end of SB for stress test)")
    parser.add_argument("--psum_512_chunk_4k", action='store_true', help="Set PSUM to 256 and cap chunk size at 2KB (default is 512 PSUM entries and max 4KB chunk size")
    parser.add_argument("--transpose_ofmap", action='store_true', help="Transpose output vector so data can be streamed out in fewer DMAs")
    parser.add_argument("--uint8_performance_mode", action='store_true',
        help="uint8 matmul performance mode. Instruction level modes "
        "'double_row', 'double_column', or 'double_pixel' are automatically "
        "chosen per situation of each matmul instruction ")
    args = parser.parse_args()

    print("\nINFO: Middle Sched v2: Running in %s mode"%(args.nname))
    print("\nINFO: Started at time %s" % str(datetime.datetime.now()))

    if args.psum_512_chunk_4k:
        PEArray.MAX_WAVE_SIZE = 512
        FileParams.chunk_sz_limit = 4096
    print("\nINFO: %d PSUM entries and %d max chunk size" % (PEArray.MAX_WAVE_SIZE, FileParams.chunk_sz_limit))

    if (args.debug > 5): np.set_printoptions(threshold=np.nan)

    # loading Kgraph
    try:
        print("\nLoading K-graph %s"%args.kgraph)
        kgraph_json = json.load(open(args.kgraph))
    except Exception as e:
        print(e)
        sys.exit(-1)

    # create graph from JSON file        
    kgraph = KGraph(args)
    kgraph.populate_from_kgraph_json(kgraph_json)

    # add forward references
    kgraph.add_forward_refs(kgraph.final_nodes)
    #kgraph.check_nodes()

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

    # dump process info
    pid = os.getpid()
    proc_status_file = "/proc/%d/status" % os.getpid()
    try:
        with open(proc_status_file) as procFh:
            print("\n", procFh.read(), flush=True)
    except Exception as e:
        print("WARNING: Can't open process ID %d status file %s"%(pid, proc_status_file))
    
    # print free SB gaps
    tpb.statebuffer.file_mapper.find_unused_gaps()

    print("\nINFO: Finished at time %s" % str(datetime.datetime.now()))

    # check for comparison errors
    if (tpb.num_mismatches > 0):
        print("\nFAILED (num mismatches %d)"%tpb.num_mismatches)
        sys.exit(1)
    else:        
        print("\nPASSED")
        sys.exit(0)
