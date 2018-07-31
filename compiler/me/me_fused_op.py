"""
Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
"""

"""Fused operations execution, including verification and waveop generation
"""

import re
import os
import sys
import numpy as np
from me_models import PEArray
from me_utils import ceildiv

sys.path.insert(0, os.environ["KAENA_PATH"] + "/compiler/tffe")
from NpUtils import NpUtils as npu

"""ID of a PE array wave
"""
class WaveID:

    def __init__(self, n_id, m_id, h_id, w_id, c_id, r_id, s_id):
        self.format = "nmhwcrs"
        self.n_id, self.m_id, self.h_id, self.w_id = n_id, m_id, h_id, w_id
        self.c_id, self.r_id, self.s_id = c_id, r_id, s_id
        self.id_array = [self.n_id, self.m_id, self.h_id, self.w_id, self.c_id, self.r_id, self.s_id]
        self.id_string = "n%d_m%d_h%d_w%d_c%d_r%d_s%d"%(self.n_id, self.m_id, self.h_id, self.w_id, self.c_id, self.r_id, self.s_id)

""" ID of completed OFMAP tile
"""
class TileID:

    def __init__(self, n_id, m_id, h_id, w_id, n, m, h, w):
        self.format = "nmhw"
        self.n_id, self.m_id, self.h_id, self.w_id = n_id, m_id, h_id, w_id
        self.n, self.m, self.h, self.w = n, m, h, w
        self.id_array = [self.n_id, self.m_id, self.h_id, self.w_id]
        self.id_string = "n%d_m%d_h%d_w%d"%(self.n_id, self.m_id, self.h_id, self.w_id)

"""List of K-Nodes that are fused (pass data through PSUM buffers)
"""
class FusedOp(list):

    def __init__(self, out_data_type, fused_op_id, args):
        self.fused_op_id = fused_op_id 
        self.prev = None
        # only accept max one of each type in fused op
        self.has_pool = False
        self.has_join= False
        self.has_conv = False
        self.has_biasadd = False
        self.pool_op = None
        self.join_op = None
        self.conv_op = None
        self.biasadd_op = None
        self.out_data_type = out_data_type 
        self.prev_weight_wave_lower_addr = -1
        self.num_pearray_inputs_dumps = args.dump_pearray_inputs
        self.args = args
        self.ofmap_is_for_join = False
        self.residue_in_scratch = False
        # "pairup" is the region or boundary where OFMAP shrinks by 1/4 and partial-batch count doubles.
        self.partial_batch_pairup = False
        # "pre-pairup" is the region just before paired-up where OFMAP shrinks by 1/4 but partial-batch count has not doubled.
        self.partial_batch_pre_pairup = False
        self.next_batch_count = 1
        self.current_batch_count = 1
        if self.args.force_batch_count > 1:
            self.next_batch_count = self.args.force_batch_count
            self.current_batch_count = self.args.force_batch_count

    # Add operation to list of fused operations.
    # Returns True if successful; False if cannot add (i.e. Pool cannot be fused)
    def add(self, op):
        if (self.args.debug > 2):
            print("DBG: adding layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
        if (op.data['layer_type'] == 'AvgPool' or op.data['layer_type'] == 'MaxPool'):
            op.populate_pooling_params()
            # If not first op, pool cannot be fused with previous op if stride != pooling window
            if (len(self) != 0
                    and (op.stride_x != op.pool_window_x 
                        or op.stride_y != op.pool_window_y
                        or op.stride_x > 1 
                        or op.stride_y > 1)):
                if (self.args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            elif (self.has_pool):
                if (self.args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            else:
                self.pool_op = op
                self.has_pool = not self.pool_op.is_id_pool
        elif (op.data['layer_type'] == 'Conv' or op.data['layer_type'] == 'MatMul' or op.data['layer_type'] == 'Softmax2'):
            if (len(self) != 0):
                if (self.args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            elif (self.has_conv):
                if (self.args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            else:
                op.populate_conv_params()
                self.conv_op = op
                self.has_conv = True
        elif (op.is_join):
            self.ofmap_is_for_join = True
            if (self.has_join):
                if (self.args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            # cannot fuse join if there's more than one missing result
            elif op.count_missing_input_results() > 1:
                return False
            else:
                self.has_join = True
                self.join_op = op
                # set the residue selection index to the other input
                if len(self) > 0:
                    self.join_op.residue_index = 1 if op.prev[0] == self[-1] else 0
                else:
                    raise RuntimeError("Please implement unfused join, where both inputs need to be sourced from SB")
        elif (op.data['layer_type'] == 'BiasAdd'):
            if (self.has_biasadd):
                if (self.args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            else:
                self.biasadd_op = op
                self.has_biasadd = True
        # Unfused Join cannot be fused with any subsequent op at the moment                
        elif (self.has_join and self.join_op == self[0]):
            return False
        if (len(op.prev) > 0):
            op.populate_common_params(adjust_for_pool=self.has_pool)
        # recompute Conv params due to constrained Pooling tile dimensions
        # (only if it is not identity pool, where window/stride are both 1)
        if (self.has_pool and op.pool_window_y > 1 and self.has_conv):
            self.conv_op.recompute_conv_params(op.pool_window_x,op.pool_window_y)
        self.append(op)
        op.fused_op = self
        return True            

    def show(self):
        print("DBG: fused_ops collected: (ofmap_is_for_join %d, partial_batch_pre_pairup %d, partial_batch_pairup %d, residue_in_scratch %d)"\
                %(self.ofmap_is_for_join, self.partial_batch_pre_pairup, self.partial_batch_pairup, self.residue_in_scratch))
        for i in self:
            print("    ", i.data["layer_type"],":",i.data["layer_name"], )

    def map_files(self, tpb, batch_item):
        map_file = tpb.statebuffer.file_mapper.map_file

        # select SB region sizing index (only relevant for ResNet50 but we also use for BiasAdd)
        # This maybe overwritten later if next batch count doubles (pairup)
        sb_size_set_index = tpb.statebuffer.batcher.sb_size_set_index[(self.current_batch_count, self.partial_batch_pre_pairup)]

        # bias file/region defaults
        bias_file_start_addr = tpb.statebuffer.next_bias_file_start
        bias_file_sz = tpb.statebuffer.batcher.item_sz
        bias_file_params = None
        if self.has_biasadd:
            assert(self.biasadd_op is not None)
            bias_file_params = self.biasadd_op.bias_file_params
            bias_file_sz = bias_file_params.tot_partition_usage_sz
        bias_region_start_addr = 0
        bias_region_sz = tpb.statebuffer.batcher.sb_bias_sz[sb_size_set_index]
        if (bias_file_start_addr + bias_file_sz) > bias_region_sz:
            bias_file_start_addr = 0
        # weights file/region defaults            
        weights_file_start_addr = 0
        weights_file_sz = tpb.statebuffer.batcher.item_sz 
        weights_file_params = None
        if self.has_conv:
            assert(self.conv_op is not None)
            weights_file_params = self.conv_op.weights_file_params
            weights_file_sz = weights_file_params.tot_partition_usage_sz 
        weights_region_start_addr  = 0
        weights_region_sz = weights_file_sz
        # ifmap file/region defaults            
        assert(self.first_op is not None)
        ifmaps_file_params = self.first_op.ifmaps_file_params
        single_ifmap_start = 0
        single_ifmap_sz = ifmaps_file_params.batch_item_partition_usage_sz
        ifmaps_region_start_addr  = 0
        ifmaps_region_sz = single_ifmap_sz
        # ofmap file/region defaults            
        assert(self.last_op is not None)
        ofmaps_file_params = self.last_op.ofmaps_file_params
        single_ofmap_start = 0
        single_ofmap_sz = ofmaps_file_params.batch_item_partition_usage_sz
        ofmaps_region_start_addr  = 0
        ofmaps_region_sz = single_ofmap_sz

        # Bias region:
        #   - keep in contiguous region to help DMA perf: an optimizer (TBD) will need to coalesce the bias files
        #   - share bias mapping for all type of nets
        #   - make sure to keep bias region the same in the batch map (BatchSBDataMap)
        if self.has_biasadd:
            if bias_file_params.mapped_params is None:
                map_file(bias_file_params, bias_file_start_addr, wrap_around=False, region_sz=bias_file_sz)
                tpb.statebuffer.next_bias_file_start = bias_file_start_addr + bias_file_sz
            else:                
                # in case that file is already mapped, keep the mapped values
                bias_file_start_addr = bias_file_params.mapped_params.start_addr

        if self.args.nname == "resnet50":
            # Input/residue uses upper portion of the shared space
            if self.first_op.is_input:
                ifmaps_region_start_addr =   tpb.statebuffer.batcher.sb_bias_sz[sb_size_set_index] \
                                           + tpb.statebuffer.batcher.sb_partialbatch_start[self.current_batch_count]
                if self.first_op.C <= 128:
                    ifmaps_region_sz = tpb.statebuffer.batcher.first_ifmaps_region_sz
                else:                
                    ifmaps_region_sz = self.current_batch_count * ifmaps_file_params.batch_item_partition_usage_sz
                # for first IFMAP, use the residue size, which is roughly equal to 3 chunks of 224x4 input tiles
                map_file(ifmaps_file_params, ifmaps_region_start_addr, wrap_around=True, region_sz=ifmaps_region_sz)
                # obtain the adjusted region size
                ifmaps_region_sz = ifmaps_file_params.mapped_params.region_sz
                # should be the same even if file was already mapped
                assert(ifmaps_region_start_addr == ifmaps_file_params.mapped_params.start_addr)
            else:            
                ifmaps_region_start_addr = ifmaps_file_params.mapped_params.start_addr
                ifmaps_region_sz  = ifmaps_file_params.mapped_params.region_sz
            # Individual IFMAP info
            single_ifmap_start = ifmaps_region_start_addr + (batch_item % self.current_batch_count) * ifmaps_file_params.batch_item_partition_usage_sz
            single_ifmap_sz = ifmaps_file_params.batch_item_partition_usage_sz

            # Join for partial-batch region
            ofmap_batch_count = self.current_batch_count
            # "pairup" is the region or boundary where OFMAP shrinks by 1/4 and partial-batch count doubles.
            if self.partial_batch_pairup:
                ofmap_batch_count = self.next_batch_count
                sb_size_set_index = tpb.statebuffer.batcher.sb_size_set_index[(ofmap_batch_count, False)]
            if ((self.last_op.is_fork or self.ofmap_is_for_join) != self.residue_in_scratch):
                # special case for stage after MaxPool: use scratch space for OFMAP instead of residue space
                #ofmaps_region_sz = ofmap_batch_count * ofmaps_file_params.batch_item_partition_usage_sz_rounded
                ofmaps_region_sz = ofmap_batch_count * ofmaps_file_params.batch_item_partition_usage_sz
                ofmaps_region_start_addr =   tpb.statebuffer.batcher.sb_bias_sz[sb_size_set_index] \
                                           + tpb.statebuffer.batcher.sb_partialbatch_start[ofmap_batch_count]
            # Scratch (OFMAP)
            else:
                ofmaps_region_sz = tpb.statebuffer.batcher.sb_scratch_sz[sb_size_set_index]
                ofmaps_region_start_addr = tpb.statebuffer.SB_PARTITION_SZ - ofmaps_region_sz

            # If OFMAP region overlaps IFMAP region, and numober of channels > 64 or stride/filter-size > 1, offset it (to lower address) by OFMAP * Tn 
            # (stride is only relevant to Conv/Pool, and filter-size is only relevant to Conv)
            if ofmaps_region_start_addr >= ifmaps_region_start_addr \
                    and ofmaps_region_start_addr < ifmaps_region_start_addr + ifmaps_region_sz:
                if (ofmaps_file_params.file_dims.C > 64) \
                    or (self.conv_op is not None and (self.conv_op.stride_x > 1 or self.conv_op.S > 1)) \
                    or (self.has_pool and self.pool_op.stride_x > 1):
                    ofmaps_region_start_addr = ifmaps_region_start_addr - ofmaps_file_params.batch_item_partition_usage_sz * self.last_op.Tn                               
                # Allow modifying in place for IFMAPs which overlap the same region as OFMAPs
                if not self.first_op.is_input:
                    ifmaps_file_params.mapped_params.modify_in_place = True

            # Map the file to region and obtain adjusted region size
            map_file(ofmaps_file_params, ofmaps_region_start_addr, wrap_around=True, region_sz=ofmaps_region_sz)
            ofmaps_region_sz = ofmaps_file_params.mapped_params.region_sz
            # should be the same even if file was already mapped
            assert(ofmaps_region_start_addr == ofmaps_file_params.mapped_params.start_addr)

            # Individual OFMAP info
            single_ofmap_start = ofmaps_region_start_addr + (batch_item % ofmap_batch_count) * ofmaps_file_params.batch_item_partition_usage_sz 
            single_ofmap_sz = ofmaps_file_params.batch_item_partition_usage_sz

            # Weights region: remaining space after allocating for bias, residue/IFMAP, and OFMAP/scratch
            if self.has_conv:
                # reselect SB region sizing index based on input current_batch_count
                sb_size_set_index = tpb.statebuffer.batcher.sb_size_set_index[(self.current_batch_count, self.partial_batch_pre_pairup)]
                if self.first_op.is_input:
                    weights_region_start_addr = ifmaps_region_start_addr + ifmaps_region_sz
                else:                
                    # right before pairup to batch count of 16, there's a jump in weights elem count, so take from partial batch space (shared space)
                    weights_region_start_addr =  tpb.statebuffer.batcher.sb_bias_sz[sb_size_set_index] \
                                               + tpb.statebuffer.batcher.sb_partialbatch_sz[sb_size_set_index]
                # align to 8B
                weights_region_start_addr = ceildiv(weights_region_start_addr, 8) * 8
                # compute region size
                weights_region_sz = ofmaps_region_start_addr - weights_region_start_addr
                # try a different start adddress based on the last allocation                
                weights_file_start_addr = tpb.statebuffer.next_weights_file_start
                weights_file_sz = weights_file_params.tot_partition_usage_sz
                if (weights_file_start_addr < weights_region_start_addr):
                    weights_file_start_addr = weights_region_start_addr
                elif (weights_file_start_addr + weights_file_sz > weights_region_start_addr + weights_region_sz):
                    weights_file_start_addr = weights_region_start_addr
                tpb.statebuffer.next_weights_file_start = weights_file_start_addr + weights_file_sz
                # map file to region                
                map_file(weights_file_params, weights_file_start_addr, wrap_around=False, region_sz=weights_file_sz)
                # obtain the adjusted region size
                weights_region_sz = weights_file_params.mapped_params.region_sz
                # also in case that file is already mapped, keep the mapped values
                weights_file_start_addr = weights_file_params.mapped_params.start_addr
        # Simple networks: use simple mapping                
        else:
            # Get start for IFMAPs
            start_addr = tpb.statebuffer.next_nonbias_file_start
            if start_addr < bias_region_sz:
                start_addr = bias_region_sz
            # IFMAPs regions                    
            # Input/residue uses upper portion of the shared space
            if self.first_op.is_input and ifmaps_file_params.mapped_params is None:
                if start_addr + ifmaps_region_sz >= tpb.statebuffer.SB_PARTITION_SZ:                    
                    start_addr = bias_region_sz
                ifmaps_region_start_addr = start_addr
                # cap region size to be 4 chunks
                ifmaps_region_sz = 4 * ifmaps_file_params.chunk_sz * self.first_op.ifmaps_file_params.fmap_channels_folds
                #assert(ifmaps_file_params.mapped_params == None)
                map_file(ifmaps_file_params, ifmaps_region_start_addr, wrap_around=True, region_sz=ifmaps_region_sz)
                # obtain the adjusted region size
                ifmaps_region_sz = ifmaps_file_params.mapped_params.region_sz
                start_addr       += ifmaps_region_sz
            else:            
                assert(ifmaps_file_params.mapped_params != None)
                ifmaps_region_start_addr = ifmaps_file_params.mapped_params.start_addr
                ifmaps_region_sz  = ifmaps_file_params.mapped_params.region_sz
            single_ifmap_start = ifmaps_region_start_addr
            # Weights region, align to 8B
            if self.has_conv:
                if weights_file_params.mapped_params is None:
                    weights_file_start_addr = start_addr
                    weights_file_start_addr = tpb.statebuffer.file_mapper.adjust0_if_overlap(
                            region0_start    = weights_file_start_addr, 
                            region0_sz       = weights_file_sz, 
                            region1_start    = single_ifmap_start, 
                            region1_sz       = min(single_ifmap_sz, ifmaps_region_sz),
                            min_region_start = bias_region_sz
                            )
                    map_file(weights_file_params, weights_file_start_addr, wrap_around=False, region_sz=weights_file_sz)
                    # obtain the adjusted region size
                    weights_region_sz = weights_file_params.mapped_params.region_sz
                    start_addr = weights_file_start_addr + weights_region_sz
                else:                    
                    # also in case that file is already mapped, keep the mapped values
                    weights_file_start_addr = weights_file_params.mapped_params.start_addr
                weights_region_start_addr = weights_file_start_addr
            # OFMAPs region
            if ofmaps_file_params.mapped_params is None:
                ofmaps_region_start_addr = start_addr
                ofmaps_region_start_addr = tpb.statebuffer.file_mapper.adjust0_if_overlap(
                        region0_start    = ofmaps_region_start_addr, 
                        region0_sz       = ofmaps_region_sz, 
                        region1_start    = weights_file_start_addr, 
                        region1_sz       = weights_file_sz,
                        min_region_start = bias_region_sz
                        )
                ofmaps_region_start_addr = tpb.statebuffer.file_mapper.adjust0_if_overlap(
                        region0_start    = ofmaps_region_start_addr, 
                        region0_sz       = ofmaps_region_sz, 
                        region1_start    = single_ifmap_start, 
                        region1_sz       = min(single_ifmap_sz, ifmaps_region_sz),
                        min_region_start = bias_region_sz
                        )
                map_file(ofmaps_file_params, ofmaps_region_start_addr, wrap_around=True, region_sz=ofmaps_region_sz)
                # obtain the adjusted region size
                ofmaps_region_sz = ofmaps_file_params.mapped_params.region_sz
                single_ofmap_start = ofmaps_region_start_addr 
                start_addr = single_ofmap_start + ofmaps_region_sz
                if single_ofmap_start == single_ifmap_start:
                    assert(not self.first_op.is_input)
                    # Allow modifying in place for IFMAPs which overlap the same region as OFMAPs
                    if not self.first_op.is_input:
                        ifmaps_file_params.mapped_params.modify_in_place = True
            else:
                ofmaps_region_start_addr = ofmaps_file_params.mapped_params.start_addr
                ofmaps_region_sz = ofmaps_file_params.mapped_params.region_sz
                single_ofmap_start = ofmaps_region_start_addr
            # Save current start address pointer for next layer
            tpb.statebuffer.next_nonbias_file_start = start_addr

        # Trace printout
        if (self.args.debug > 2 and not tpb.statebuffer.printed_map_trace_header): 
            print("SB MAP TRACE, fused op, fused op ID, batch elem, Tn, current_batch_count, next_batch_count, partial_batch_pre_pairup, partial_batch_pairup, residue_in_scratch,, \
 bias file end_addr, bias region end_addr, bias region start_addr, bias file start_addr,,\
 weights file end_addr, weights region end_addr, weights region start_addr, weights file start_addr,,\
 ifmap single end_addr, ifmaps region end_addr, ifmaps region start_addr, ifmap single start_addr,,\
 ofmap single end_addr, ofmaps region end_addr, ofmaps region start_addr, ofmap single start_addr,, ifmap file, ofmap file")                    
            tpb.statebuffer.printed_map_trace_header = True
        if (self.args.debug > 2): print("SB MAP TRACE, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d,, %d, %d, %d, %d,, %d, %d, %d, %d,, %d, %d, %d, %d,, %s, %s"%(
                                self.last_op.data['layer_name'], 
                                self.fused_op_id, 
                                batch_item,
                                self.first_op.Tn,
                                self.current_batch_count,
                                self.next_batch_count,
                                self.partial_batch_pre_pairup,
                                self.partial_batch_pairup,
                                self.residue_in_scratch,
                                bias_file_start_addr + bias_file_sz - tpb.statebuffer.batcher.item_sz, 
                                bias_region_start_addr + bias_region_sz - tpb.statebuffer.batcher.item_sz,
                                bias_region_start_addr,
                                bias_file_start_addr,
                                weights_file_start_addr + weights_file_sz - tpb.statebuffer.batcher.item_sz, 
                                weights_region_start_addr + weights_region_sz - tpb.statebuffer.batcher.item_sz,
                                weights_region_start_addr,
                                weights_file_start_addr,
                                single_ifmap_start + single_ifmap_sz - tpb.statebuffer.batcher.item_sz, 
                                ifmaps_region_start_addr + ifmaps_region_sz - tpb.statebuffer.batcher.item_sz,
                                ifmaps_region_start_addr,
                                single_ifmap_start,
                                single_ofmap_start + single_ofmap_sz - tpb.statebuffer.batcher.item_sz, 
                                ofmaps_region_start_addr + ofmaps_region_sz - tpb.statebuffer.batcher.item_sz,
                                ofmaps_region_start_addr,
                                single_ofmap_start,
                                self.first_op.ifmaps_file_params.file_name,
                                self.last_op.ofmaps_file_params.file_name
                                ))

        # weights cannot overlap OFMAP/IFMAP
        assert(tpb.statebuffer.file_mapper.check_overlap(weights_file_start_addr, weights_region_sz, single_ofmap_start, min(single_ofmap_sz, ofmaps_region_sz))==False)
        assert(tpb.statebuffer.file_mapper.check_overlap(weights_file_start_addr, weights_region_sz, single_ifmap_start, min(single_ifmap_sz, ifmaps_region_sz))==False)

        # check that regions are either exactly overlaping or not overlapping at all
        overlap_some_percent = tpb.statebuffer.file_mapper.check_overlap(single_ifmap_start, min(single_ifmap_sz, ifmaps_region_sz), single_ofmap_start, single_ofmap_sz)
        overlap_100_percent = tpb.statebuffer.file_mapper.check_overlap100(single_ifmap_start, min(single_ifmap_sz, ifmaps_region_sz), single_ofmap_start, single_ofmap_sz)
        assert(overlap_some_percent == overlap_100_percent)

    def execute(self, tpb, batch_item):
        assert (batch_item >= 0)
        assert (batch_item < self.first_op.N)
        # Check conv fused op
        first_op_type = self.first_op.data['layer_type']
        if (first_op_type == "Conv" or first_op_type == "MatMul"):
            results = self.execute_conv_ops(tpb, batch_item)
        elif (first_op_type == "AvgPool" or first_op_type == "MaxPool"):
            results = self.execute_unfused_pool_op(tpb, batch_item)
        #elif (first_op_type == "Softmax2"):
        #    results = tpb.execute_softmax2(result_file)
        #elif (first_op_type == "Multiply" or first_op_type == "ResAdd"): # TODO: handle the scalar 
        #    first_op.src_circbuf = first_op.prev[0].dst_circbuf
        #    if (len(first_op.data['previous_layers']) == 1):
        #        inputs = first_op.src_circbuf.load_data(first_op)
        #        results = tpb.execute_unfused_pool_op(inputs, result_file)
        #    elif (len(first_op.data['previous_layers']) == 2):
        #        inputs = first_op.src_circbuf.load_data(first_op)
        #        results = tpb.execute_unfused_pool_op(inputs, result_file)
        #    else:                
        #        print("ERROR: cannot handle more than two inputs for first operation %s, layer %s"%(first_op_type, first_op.data["layer_name"]))
        #        exit(-1)
            #inputs2 = tpb.statebuffer.circbuf_residue.load_data(first_op)
            #results = tpb.execute_multiply(inputs, inputs2, result_file)
        elif re.search(r"Relu|Tanh|Sigmoid|Exp|Identity|Lrelu|Prelu|BiasAdd", first_op_type):
            results = self.execute_unfused_pool_op(tpb, batch_item)
        #else:        
        #    print("ERROR: Unrecognized first operation %s"%first_op_type)
        #    exit(-1)

        # Check results against pre-computed results and save data
        # only if there's at least one node, and first node is not Placeholder or NOP (simple Reshape, etc.)
        if len(self) > 0 and not self.first_op.is_placeholder and not self.first_op.is_nop:
            # Check last output result, unless verify_output_only=False
            if self.last_op.next == [] or not self.args.verify_output_only:
                if 'ref_file' in self.last_op.data and os.path.isfile(self.last_op.data['ref_file']):
                    try:
                        expected_ofmaps = np.load(self.last_op.data['ref_file'])
                    except:
                        raise RuntimeError("Cannot load numpy file %s"%(self.last_op.data['ref_file']))
                    last_batch_item = batch_item + self.first_op.Tn
                    for i in range(batch_item, last_batch_item):
                        ifmaps = self.first_op.ifmaps_file_params.dram_data[i, :]
                        ofmaps = self.last_op.ofmaps_file_params.dram_data[i, :]
                        expected_ofmaps_extracted = expected_ofmaps[i, :]
                        assert(expected_ofmaps_extracted.flags.c_contiguous == True)
                        diff = ofmaps - expected_ofmaps_extracted
                        if (self.args.debug > 2): print("\nInput IFMAPS:\n", ifmaps)
                        if (self.args.debug > 1): print("\nComputed OFMAPS:\n", ofmaps)
                        if (self.args.debug > 1): print("\nExpected OFMAPS:\n", expected_ofmaps_extracted)
                        if (self.args.debug > 1): print("\nDiffed   OFMAPS:\n", diff)
                        if (not npu.allclose(ofmaps, expected_ofmaps_extracted, 1/100, 1e-5, verbose=True)):
                            print("\nERROR: layer %s batch item %d computed OFMAPS is not equal to expected OFMAPS!\n"%(self.last_op.data['layer_name'], i))
                            tpb.num_mismatches += 1

            # Save results for network output or we want to save debug intermediate results
            if self.last_op.next == [] or self.args.save_layer_output:
                last_batch_item = batch_item + self.first_op.Tn
                for i in range(batch_item, last_batch_item):
                    waveops = tpb.statebuffer.file_mapper.flush_file(
                                    nonload_waveop_id   = tpb.waveop_stream.nonload_waveop_count, 
                                    nonload_waveop_list = tpb.waveop_stream.nonload_waveop_list, 
                                    file_params         = self.last_op.ofmaps_file_params, 
                                    batch_item          = i
                                    )
                    tpb.waveop_stream.add_outputs(waveops)
                    self.last_op.ofmaps_file_params.save_file()

    # generate MatMul waveop and add it to waveop stream
    def gen_matmul_waveop(self, tpb, wave_id, psum_add, dram_weights_waveops):
        batch_item = wave_id.n_id * self.conv_op.Tn
        if (self.conv_op.item_sz == 2):
            in_dtype = "float16"
            out_dtype = "float32"
        elif (self.conv_op.item_sz == 4):
            in_dtype = "float32"
            out_dtype = "float32"
        else:            
            print("ERROR: item_sz %d not yet supported"%self.conv_op.item_sz)
        # find the weights offset within atom; -1 means don't load new weights
        weights_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.conv_op.weights_file_params, 0, self.conv_op.weight_wave_lower_addr)
        # kaena-421: during execution for batch item other than the first one, need to check if there's any SBAtomLoad due to eviction
        # when the fused-op is reexecuted (since self.prev_weight_wave_lower_addr is preserved across batch item calls) 
        if (weights_sb_address == self.prev_weight_wave_lower_addr and dram_weights_waveops == []):
            weights_sb_address = -1
            if (self.args.debug > 1): print("DBG: weights has been previously loaded into PEArray; reusing them instead of reloading")
        else:            
            self.prev_weight_wave_lower_addr = weights_sb_address

        # If wave crosses atom boundaries, break it into multiple waves
        # The following assumes noodle tile (width is equal to FMAP width)
        current_chunk_id = -10000   # force the first break at start address
        current_atom_id = -10000
        break_at_y = []
        break_addr = []
        addr_step_y = self.conv_op.W * self.conv_op.stride_y * self.conv_op.item_sz
        for i in range(self.conv_op.ofmap_wave_height):
            # TODO: how to deal with partial batching here?
            address = self.conv_op.ifmap_wave_lower_addr[0] + i * addr_step_y
            if (address > self.conv_op.ifmap_wave_upper_addr[0]):
                break
            chunk_id = tpb.statebuffer.file_mapper.get_chunk_id_from_file_addr(self.conv_op.ifmaps_file_params, batch_item, address)
            atom_id = tpb.statebuffer.file_mapper.get_atom_id_from_file_addr(self.conv_op.ifmaps_file_params, batch_item, address)
            if self.args.abstract_mem:
                break_cond = chunk_id != current_chunk_id
            else:
                break_cond = not (atom_id == current_atom_id or atom_id == current_atom_id+1)
            if break_cond:
                break_at_y.append(i)
                break_addr.append(tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.conv_op.ifmaps_file_params, batch_item, address))
                current_chunk_id = chunk_id
                current_atom_id = atom_id
                if (self.args.debug > 3): print("DBG: breaking wave at row %d addr %d"%(i, break_addr[-1]))
        matmul_waveop = []
        start_tensor_calc = not(psum_add)

        # replication parameters
        fmap_x_num = self.conv_op.ofmap_wave_width
        fmap_x_step = self.conv_op.stride_x
        fmap_y_step = self.conv_op.W * self.conv_op.stride_y
        if self.conv_op.Tn > 1:
            fmap_z_step = self.conv_op.ifmaps_file_params.batch_item_partition_usage_sz//self.conv_op.item_sz
        else:
            fmap_z_step = 1
        dst_x_num = self.conv_op.ofmap_wave_width
        dst_y_step = self.conv_op.ofmap_cropped_tile_width
        ifmap_replication_resolution = 0
        ifmap_replication_num_rows = 0
        ifmap_replication_shift_amnt = 0
        if self.conv_op.repl_multiple_of_C > 1:
            # Kaena-593: ensure no bubble during IFMAP streaming (packed pattern)
            fmap_x_num = self.conv_op.W // self.conv_op.stride_x
            fmap_x_step = 1     # image gets split into even/odd
            fmap_y_step = self.conv_op.W // self.conv_op.stride_x
            dst_x_num = fmap_x_num
            dst_y_step = fmap_y_step
            ifmap_replication_resolution = self.conv_op.C * self.conv_op.stride_x
            ifmap_replication_num_rows = self.conv_op.C * self.conv_op.S
            ifmap_replication_shift_amnt = 1

        for i in range(len(break_at_y)):                
            if (i == len(break_at_y)-1):
                next_break = self.conv_op.ofmap_wave_height
            else:
                next_break = break_at_y[i+1]
            fmap_y_num = next_break - break_at_y[i]
            psum_bank_additional_offset = break_at_y[i] * self.conv_op.ofmap_cropped_tile_width
            assert((self.conv_op.psum_bank_offset + psum_bank_additional_offset) < PEArray.MAX_WAVE_SIZE)
            fmap_sb_address = break_addr[i]
            if ifmap_replication_resolution > 1:
                assert((fmap_sb_address%8) == 0), "Kaena-593: IFMAP start address must by 8B aligned for replication to work"

            # get dram waveops (weights) for the first piece of broken matmul
            dram_waveop_names = []
            if i==0:
                for j in dram_weights_waveops:
                    dram_waveop_names.append(j["waveop_name"])
            else:
                # reuse weights loaded with first piece of broken matmul
                weights_sb_address = -1

            for z in range(self.conv_op.Tn):                    
                lower_file_address = self.conv_op.ifmap_wave_lower_addr[z] + break_at_y[i] * addr_step_y
                upper_file_address = min(self.conv_op.ifmap_wave_lower_addr[z] + next_break * addr_step_y - self.conv_op.item_sz, self.conv_op.ifmap_wave_upper_addr[z])
                list_of_names = tpb.statebuffer.file_mapper.get_dram_waveop_names(self.conv_op.ifmaps_file_params, batch_item + z, lower_file_address, upper_file_address)
                for name in list_of_names:
                    if name not in dram_waveop_names:
                        dram_waveop_names.append(name)

            waveop_name = self.conv_op.data['layer_name']+"/MatMul_"+wave_id.id_string+"__"+str(i)
            if (self.args.debug > 2): print("DBG %s: MatMul wave %s subwave %d weights_sb_address %d, fmap_sb_address %d, fmap_y_num %d"%(self.conv_op.data['layer_name'], waveop_name, i, weights_sb_address, fmap_sb_address, fmap_y_num))                
            matmul_waveop.append({ 
                  'previous_waveops'        : dram_waveop_names,
                  'waveop_type'             : 'MatMul',
                  'waveop_name'             : waveop_name,
                  'layer_name'              : self.conv_op.data['layer_name'],
                  'weights_sb_address'      : weights_sb_address,
                  'in_dtype'                : in_dtype,
                  'out_dtype'               : out_dtype,
                  'start_tensor_calc'       : start_tensor_calc,
                  'stop_tensor_calc'        : False,
                  'src_is_psum'             : False,
                  'src_sb_address'          : fmap_sb_address,
                  'src_start_at_mid_part'   : False,
                  'src_x_step'              : fmap_x_step,
                  'src_x_num'               : fmap_x_num,
                  'src_y_step'              : fmap_y_step,
                  'src_y_num'               : fmap_y_num,
                  'src_z_step'              : fmap_z_step,
                  'src_z_num'               : self.conv_op.Tn,
                  'num_row_partitions'      : self.conv_op.ifmap_count,
                  'dst_is_psum'             : True,
                  'dst_psum_bank_id'        : self.conv_op.psum_bank_dst,
                  'dst_psum_bank_offset'    : self.conv_op.psum_bank_offset + psum_bank_additional_offset,
                  'dst_x_step'              : 1,
                  'dst_x_num'               : dst_x_num,
                  'dst_y_step'              : dst_y_step,
                  'dst_y_num'               : fmap_y_num,
                  'dst_z_step'              : self.conv_op.ofmap_full_tile_sz,
                  'dst_z_num'               : self.conv_op.Tn,
                  'num_column_partitions'   : self.conv_op.ofmap_count,
                  'ifmap_replication_resolution' : ifmap_replication_resolution, 
                  'ifmap_replication_num_rows' : ifmap_replication_num_rows,
                  'ifmap_replication_shift_amnt' : ifmap_replication_shift_amnt,
                })
            start_tensor_calc = False   # this is only true for the first MatMul, even when there's a break
        return matmul_waveop

    # generate Pool waveop and add it to waveop stream
    # TODO: currently, always go to SB after Pooling
    # TODO: currently, cannot process multiple batch items in one instruction
    def gen_pool_waveop(self, tpb, tile_id, src_is_psum, src_psum_bank_id, start_at_mid_part, partial_batch_item):
        batch_item = tile_id.n_id * self.pool_op.Tn
        if (src_is_psum):
            src_ifmap_width = self.pool_op.ifmap_cropped_tile_width
            if self.conv_op != None and self.conv_op.repl_multiple_of_C > 1:
                src_ifmap_width = self.conv_op.W // self.conv_op.stride_x
            src_ifmap_height = self.pool_op.ifmap_cropped_tile_height
            src_sb_address = 0
            if (self.pool_op.item_sz == 2):
                in_dtype = "float32"
            else:    
                in_dtype = "float32"
        else:
            src_ifmap_width = self.pool_op.W
            src_ifmap_height = self.pool_op.H
            src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.pool_op.ifmaps_file_params, batch_item + partial_batch_item, self.pool_op.ifmap_wave_lower_addr[partial_batch_item])
            in_dtype = self.out_data_type
        src_psum_bank_offset = src_ifmap_width * src_ifmap_height * partial_batch_item
        psum_step_multiplier = 1   # kaena-174, tonga-310: after Inkling fix, no need for multiplier         
        waveop_name = self.pool_op.data['layer_name']+"/Pool_"+tile_id.id_string
        pool_frequency = self.pool_op.pool_window_x * self.pool_op.pool_window_y
        pool_scale = float(1/pool_frequency)
        dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.pool_op.ofmaps_file_params, batch_item + partial_batch_item, self.pool_op.ofmap_tile_lower_addr[partial_batch_item])
        dst_is_psum = False
        instr = {
              'previous_waveops'        : [],   # to be added later
              'waveop_type'             : 'Pool',
              'waveop_name'             : waveop_name,
              'layer_name'              : self.pool_op.data['layer_name'],
              'tile_id_format'          : tile_id.format,
              'tile_id'                 : tile_id.id_array,
              'pool_func'               : self.pool_op.data['layer_type'],
              'in_dtype'                : in_dtype,
              'out_dtype'               : self.out_data_type,
              'src_is_psum'             : src_is_psum,
              'src_x_step'              : 1 * psum_step_multiplier,
              'src_x_num'               : self.pool_op.pool_window_x,
              'src_y_step'              : src_ifmap_width * psum_step_multiplier,
              'src_y_num'               : self.pool_op.pool_window_y,
              'src_z_step'              : self.pool_op.stride_x * psum_step_multiplier,
              'src_z_num'               : self.pool_op.ofmap_cropped_tile_width,
              'src_w_step'              : src_ifmap_width * self.pool_op.stride_y * psum_step_multiplier,
              'src_w_num'               : self.pool_op.ofmap_cropped_tile_height,
              'pool_frequency'          : pool_frequency,
              'pool_scale'              : pool_scale,
              'num_partitions'          : self.pool_op.ofmap_count,
              'dst_is_psum'             : dst_is_psum,
              'dst_x_step'              : 1,
              'dst_x_num'               : self.pool_op.ofmap_cropped_tile_width,
              'dst_y_step'              : self.pool_op.E,
              'dst_y_num'               : self.pool_op.ofmap_cropped_tile_height,
              'dst_z_step'              : 1, 
              'dst_z_num'               : 1,
            }
        if src_is_psum:
            instr['src_psum_bank_id'] = src_psum_bank_id
            instr['src_psum_bank_offset'] = src_psum_bank_offset
        else:                
            instr['src_sb_address'] = src_sb_address
            instr['src_start_at_mid_part'] = start_at_mid_part 
        if dst_is_psum:
            instr['dst_psum_bank_id'] = dst_psum_bank_id
            instr['dst_psum_bank_offset'] = dst_psum_bank_offset
        else:                
            instr['dst_sb_address'] = dst_sb_address
            instr['dst_start_at_mid_part'] = start_at_mid_part 
        return instr

    # execute PEArray matrix multiply; returns True if successful (IFMAP wave is non-zero)
    def execute_matmul_waveop(self, tpb, wave_id, inputs, weights, psum_add, repl_multiple_of_C):
        batch_item = wave_id.n_id * self.conv_op.Tn
        pearray_packed_weights = self.conv_op.pack_wave_conv_weights(
                                        weights, 
                                        wave_id, 
                                        repl_multiple_of_C
                                        )
        pearray_packed_ifmaps = self.conv_op.pack_wave_ifmaps(
                                        inputs, 
                                        wave_id,
                                        repl_multiple_of_C,
                                        for_softmax=False
                                        )
        #print("\npearray_packed_ifmaps", wave_id.id_array, "\n", pearray_packed_ifmaps)
        #print("\npearray_packed_weights", wave_id.id_array, "\n", pearray_packed_weights)
        if (self.conv_op.ifmap_wave_lower_addr[0] < 0 or self.conv_op.ifmap_wave_upper_addr[0] < 0):
            print("WARNING layer %s: IFMAP wave (%s) has no data, so don't create waveops for this wave"%(self[0].data['layer_name'], wave_id.id_string))
            return False
        else:
            tpb.pearray.wave_fp16_mm(pearray_packed_ifmaps, pearray_packed_weights, self.conv_op.psum_bank_dst, psum_add)
            # Generate weights waveops
            (writers, readers, dram_weights_waveops) = tpb.statebuffer.file_mapper.read_file_data_region(
                                        tpb.waveop_stream.nonload_waveop_count,
                                        tpb.waveop_stream.nonload_waveop_list,
                                        self.conv_op.weights_file_params,
                                        0,  # batch_item doesn't apply for weights
                                        self.conv_op.weight_wave_lower_addr, 
                                        self.conv_op.weight_wave_upper_addr - self.conv_op.weight_wave_lower_addr + self.conv_op.item_sz,
                                        repl_multiple_of_C
                                        )
            if (self.args.debug > 2): print("DBG %s: MatMul weight_wave_lower_addr %d weight_wave_upper_addr %d"%(self.conv_op.data['layer_name'], self.conv_op.weight_wave_lower_addr, self.conv_op.weight_wave_upper_addr))                
                                        
            for i in dram_weights_waveops: tpb.waveop_stream.append_check(i)

            dram_ifmaps_waveops = []
            latest_accessor = -1
            for z in range(self.conv_op.Tn):
                # TODO: move the following into gen_matmul_waveop to handle breaking wave into two
                (last_writer, last_reader, waveops) = tpb.statebuffer.file_mapper.read_file_data_region(
                                            tpb.waveop_stream.nonload_waveop_count,
                                            tpb.waveop_stream.nonload_waveop_list,
                                            self.conv_op.ifmaps_file_params,
                                            batch_item + z,
                                            self.conv_op.ifmap_wave_lower_addr[z], 
                                            self.conv_op.ifmap_wave_upper_addr[z] - self.conv_op.ifmap_wave_lower_addr[z] + self.conv_op.item_sz,
                                            self.conv_op.repl_multiple_of_C
                                            )
                if self.args.no_inter_layer_load:
                    if (not self.conv_op.is_input and len(waveops) > 0):
                        raise RuntimeError("There are DRAM loads when option no_inter_layer_load is set")
                if (self.args.debug > 2): print("DBG %s: MatMul ifmaps_wave_lower_addr %d ifmap_wave_upper_addr %d"%(self.conv_op.data['layer_name'], self.conv_op.ifmap_wave_lower_addr[z], self.conv_op.ifmap_wave_upper_addr[z]))                
                dram_ifmaps_waveops += waveops
                latest_accessor = max(last_writer, latest_accessor) # don't include readers since this matmul is a reader, and we don't need to add RAR dependency
            
            # consider all Tn batch items together to avoid redundant edges
            # TODO: roll this code into read_file_data_region
            prev_waveops = []
            if dram_ifmaps_waveops == []:
                if latest_accessor >= 0:
                    latest_accessor_waveop = tpb.waveop_stream.nonload_waveop_list[latest_accessor]
                    if not self.args.relax_dependencies or (latest_accessor_waveop['waveop_type'] != 'Activation' and latest_accessor_waveop['waveop_type'] != 'Pool'):
                        prev_waveops.append(latest_accessor_waveop['waveop_name'])

            for i in dram_ifmaps_waveops: tpb.waveop_stream.append_check(i)
            matmul_waveop = self.gen_matmul_waveop(tpb, wave_id, psum_add, dram_weights_waveops)
            for i in range(len(matmul_waveop)):
                tpb.waveop_stream.add_linked(matmul_waveop[i], [], self.conv_op.psum_bank_dst)
                # TODO: move the following into gen_matmul_waveop to handle breaking wave into two
                if i==0:    # only need to satisfy the first in group of matmul waveops
                    for j in prev_waveops:
                        if j not in matmul_waveop[i]['previous_waveops']:
                            matmul_waveop[i]['previous_waveops'].append(j)
            # mark this matmul as consumer of the 64B weights morsel
            #matmul_waveop_name = matmul_waveop[-1]["waveop_name"]
            #matmul_waveop_name = ""
            # collect for statistics
            #tpb.pearray.batching_in_wave = self.conv_op.Tn
            # dump PEArray inputs
            #if (self.num_pearray_inputs_dumps > 0):
            #    self.num_pearray_inputs_dumps -= 1
            #    actual_wave_ifmaps = pearray_packed_ifmaps[self.conv_op.psum_bank_offset:self.conv_op.psum_bank_offset+self.conv_op.ofmap_full_tile_sz, 0:self.conv_op.ifmap_count]
            #    actual_wave_weights = pearray_packed_weights[0:self.conv_op.ifmap_count, 0:self.conv_op.ofmap_count]
            #    matmul_waveop_name = re.sub("/", "_", matmul_waveop_name)
            #    np.savetxt("pearray_inputs_ifmaps_"+matmul_waveop_name, actual_wave_ifmaps.astype(self.conv_op.data_type))
            #    np.savetxt("pearray_inputs_weights_"+matmul_waveop_name, actual_wave_weights.astype(self.conv_op.data_type))
            # collect statistics
            #if (args.debug > 1):
            #    tpb.pearray.total_pearray_wave_elems += self.conv_op.ofmap_wave_elems
            #    if (matmul_waveop[0]["weights_sb_address"] < 0):
            #        tpb.pearray.total_pearray_latency_cycles += self.conv_op.ofmap_wave_elems
            #    else:    
            #        tpb.pearray.total_pearray_latency_cycles += max(self.conv_op.ofmap_count, self.conv_op.ofmap_wave_elems)
            #    tpb.pearray.num_of_ops_executed += self.conv_op.ofmap_count * self.conv_op.ofmap_wave_elems * self.conv_op.Tn * self.conv_op.ifmap_count
            return True
        
    # execute remaining fused ops
    def execute_tile_ops (self, tpb, wave_id, tile_id, psum_bank_src, bias, psum_temp):
        op_list_iter = iter(range(1, len(self)))
        op_list = self
        batch_item = wave_id.n_id * self.conv_op.Tn
        for i in op_list_iter:
            layer_type = self[i].data['layer_type'] 
            if (re.search(r"Relu|Tanh|Sigmoid|Exp|Identity|Lrelu|Prelu", layer_type)):
                dram_bias_waveops = []
                latest_accessor = -1
                if (tile_id.m_id%2 == 0):
                    (last_writer, last_reader, dram_bias_waveops) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                    tpb.waveop_stream.nonload_waveop_count,
                                                    tpb.waveop_stream.nonload_waveop_list,
                                                    tpb.statebuffer.zero_bias_file_params,
                                                    0,  # batch_item is not needed for bias
                                                    0,
                                                    tpb.statebuffer.zero_bias_file_params.item_sz)
                    latest_accessor = max(last_writer, last_reader)
                # TODO: roll this code into read_file_data_region
                prev_waveops = []
                if latest_accessor >= 0:
                    latest_accessor_waveop = tpb.waveop_stream.nonload_waveop_list[latest_accessor]
                    if not self.args.relax_dependencies or (latest_accessor_waveop['waveop_type'] != 'Activation' and latest_accessor_waveop['waveop_type'] != 'Pool'):
                        prev_waveops.append(latest_accessor_waveop['waveop_name'])
                psum_temp = tpb.activate.act(op_list[i].data['layer_type'], psum_temp)
                psum_bank_dst = psum_bank_src
                dst_is_psum = False
                if (i != len(op_list)-1):
                    dst_is_psum = True
                    tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, psum_temp)
                self.gen_act_waveop_inline(tpb, None, op_list[i], self.conv_op, tile_id, 
                                          True, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, 0)
                psum_bank_src = psum_bank_dst
                tpb.waveop_stream.last_psum_waveop[psum_bank_dst]['previous_waveops'] += prev_waveops
            elif (layer_type == 'BiasAdd'):
                bias_chan_start = (tile_id.m_id//2) * PEArray.NUM_ROWS
                bias_chan_mid_part = (tile_id.m_id%2) == 1
                bias_chan_end = min(bias_chan_start + PEArray.NUM_ROWS, self.conv_op.M)
                bias_extracted = np.zeros(PEArray.NUM_ROWS)
                bias_extracted[0 : bias_chan_end - bias_chan_start] = bias[bias_chan_start : bias_chan_end]
                bias_addr = bias_chan_start * op_list[i].item_sz
                dram_bias_waveops = []
                latest_accessor = -1
                if (tile_id.m_id%2 == 0):
                    (last_writer, last_reader, dram_bias_waveops) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                    tpb.waveop_stream.nonload_waveop_count,
                                                    tpb.waveop_stream.nonload_waveop_list,
                                                    self.biasadd_op.bias_file_params,
                                                    0,  # batch_item is not needed for bias
                                                    bias_addr,
                                                    self.biasadd_op.item_sz)
                    latest_accessor = max(last_writer, last_reader)

                # TODO: roll this code into read_file_data_region
                prev_waveops = []
                if latest_accessor >= 0:
                    latest_accessor_waveop = tpb.waveop_stream.nonload_waveop_list[latest_accessor]
                    if not self.args.relax_dependencies or (latest_accessor_waveop['waveop_type'] != 'Activation' and latest_accessor_waveop['waveop_type'] != 'Pool'):
                        prev_waveops.append(latest_accessor_waveop['waveop_name'])
                #x = DBG_DUMP_PSUM_COL("PSUM col0 before BiasAdd (FP32): ", psum_temp, 0)
                psum_temp = tpb.activate.biasadd(psum_temp, bias_extracted[bias_chan_mid_part*PEArray.NUM_COLS : (bias_chan_mid_part+1)*PEArray.NUM_COLS])
                #y = DBG_DUMP_PSUM_COL("PSUM col0 after BiasAdd: ", psum_temp, 0)
                #print(y-x)
                psum_bank_dst = psum_bank_src 
                dst_is_psum = False
                if (i+1 < len(op_list) and re.search(r"Relu|Tanh|Sigmoid|Exp|Identity|Lrelu|Prelu", op_list[i+1].data['layer_type'])):
                    psum_temp = tpb.activate.act(op_list[i+1].data['layer_type'], psum_temp)
                    if (i+1 != len(op_list)-1):
                        dst_is_psum = True
                        tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, psum_temp)
                    self.gen_act_waveop_inline(tpb, op_list[i], op_list[i+1], self.conv_op, tile_id, 
                                              True, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_addr)
                    psum_bank_src = psum_bank_dst
                    next(op_list_iter)
                else:                                    
                    if (i != len(op_list)-1):
                        dst_is_psum = True
                        tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, psum_temp)
                    self.gen_act_waveop_inline(tpb, op_list[i], None, self.conv_op, tile_id, 
                                              True, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_addr)
                    psum_bank_src = psum_bank_dst
                tpb.waveop_stream.last_psum_waveop[psum_bank_dst]['previous_waveops'] += prev_waveops
            elif (self[i].is_join):
                dram_resadd_waveops = []
                latest_accessor = -1
                for z in range(op_list.conv_op.Tn):
                    (last_writer, last_reader, waveops) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                tpb.waveop_stream.nonload_waveop_count,
                                                tpb.waveop_stream.nonload_waveop_list,
                                                self.last_op.ofmaps_file_params,
                                                batch_item + z,
                                                self.conv_op.ofmap_tile_lower_addr[z], 
                                                self.conv_op.ofmap_tile_upper_addr[z] - self.conv_op.ofmap_tile_lower_addr[z] + self.conv_op.item_sz)
                    if self.args.no_inter_layer_load:
                        if (not self.conv_op.is_input and len(waveops) > 0):
                            raise RuntimeError("There are DRAM loads when option no_inter_layer_load is set")
                    if (self.args.debug > 2): print("DBG %s: ResAdd/Mult ofmaps_tile_lower_addr %d ofmap_tile_upper_addr %d"%(self.conv_op.data['layer_name'], self.conv_op.ofmap_tile_lower_addr[z], self.conv_op.ofmap_tile_upper_addr[z]))                
                    dram_resadd_waveops += waveops
                    latest_accessor = max(last_writer, latest_accessor) # don't include readers since this join is a reader, and we don't need to add RAR dependency

                # consider all Tn batch items together to avoid redundant edges
                # TODO: roll this code into read_file_data_region
                prev_waveops = []
                if dram_resadd_waveops == []:
                    if latest_accessor >= 0:
                        latest_accessor_waveop = tpb.waveop_stream.nonload_waveop_list[latest_accessor]
                        if not self.args.relax_dependencies or (latest_accessor_waveop['waveop_type'] != 'Activation' and latest_accessor_waveop['waveop_type'] != 'Pool'):
                            prev_waveops.append(latest_accessor_waveop['waveop_name'])

                # Do the actual math
                residue_ifmaps = np.zeros((self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, PEArray.NUM_COLS), dtype=np.float32)
                for z in range(op_list.conv_op.Tn):
                    for j in range(PEArray.NUM_COLS):
                        M_idx = tile_id.m_id * PEArray.NUM_COLS + j
                        if (M_idx >= self.conv_op.M):
                            break
                        else:
                            # NCHW
                            residue_tile_ifmap = np.zeros((self.conv_op.ofmap_full_tiley_sz, self.conv_op.ofmap_full_tilex_sz), dtype=np.float32)
                            residue_tile_ifmap[0:self.conv_op.ofmap_cropped_tile_height, 0:self.conv_op.ofmap_cropped_tile_width] = \
                                self.last_op.ofmaps_file_params.dram_data[
                                    tile_id.n_id * op_list.conv_op.Tn + z, 
                                    M_idx, 
                                    self.conv_op.ofmap_tile_y_start : self.conv_op.ofmap_tile_y_start + self.conv_op.ofmap_cropped_tile_height, 
                                    self.conv_op.ofmap_tile_x_start : self.conv_op.ofmap_tile_x_start + self.conv_op.ofmap_cropped_tile_width]
                            residue_ifmaps[z * self.conv_op.ofmap_full_tile_sz : (z+1) * self.conv_op.ofmap_full_tile_sz,j] = residue_tile_ifmap.flatten()
                #x1 = DBG_DUMP_PSUM_COL("PSUM col0 before ResAdd (FP32): ", psum_temp, 0)
                #x2 = DBG_DUMP_PSUM_COL("Residue col0 before ResAdd (FP32): ", residue_ifmaps, 0)
                if (layer_type == 'ResAdd'):
                    psum_temp = tpb.pool.resadd(psum_temp, residue_ifmaps)
                elif (layer_type == 'Multiply'):    
                    psum_temp = tpb.pool.multiply(psum_temp, residue_ifmaps)
                else:
                    print("ERROR: don't know how to handle vector op %s for layer %s"%(layer_type, self[i].data["layer_name"]))
                #y1 = DBG_DUMP_PSUM_COL("PSUM col0 after RessAdd (FP32): ", psum_temp, 0)
                psum_bank_dst = psum_bank_src
                dst_is_psum = False
                if (i != len(op_list)-1):
                    dst_is_psum = True
                    tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, psum_temp)
                self.gen_join_waveop_inline(tpb, op_list[i], 
                        self.conv_op, 
                        tile_id, 
                        True,
                        psum_bank_src, 
                        dst_is_psum, 
                        psum_bank_dst, 
                        dram_resadd_waveops, 
                        self.conv_op.ofmap_tile_lower_addr[0], 
                        (tile_id.m_id%2)==1)
                for j in prev_waveops:
                    if j not in tpb.waveop_stream.last_main_waveop['previous_waveops']:
                        tpb.waveop_stream.last_main_waveop['previous_waveops'].append(j)
                psum_bank_src = psum_bank_dst
            elif ((layer_type == 'AvgPool') or (layer_type == 'MaxPool')):
                self[i].compute_ofmap_tile_info(tile_id)
                #tilex = self.conv_op.ofmap_cropped_tile_width
                #tiley = self.conv_op.ofmap_cropped_tile_height
                tilex = self[i].ofmap_full_tilex_sz * self[i].stride_x
                tiley = self[i].ofmap_full_tiley_sz * self[i].stride_y
                #x = DBG_DUMP_PSUM_COL("PSUM before pool: ", psum_temp, 0)
                psum_temp = tpb.pool.pool(layer_type, psum_temp, self[i].stride_x, self[i].pool_window_y, self[i].Tn, tilex, tiley, self[i].ofmap_full_tilex_sz, self[i].ofmap_full_tiley_sz)
                #x = DBG_DUMP_PSUM_COL("PSUM after pool: ", psum_temp, 0)
                self.gen_fused_pool_waveop_inline(tpb, tile_id, psum_bank_src, (tile_id.m_id%2) == 1)
                # Don't go to back to psum for pooling
                #psum_bank_dst = 3
                #if (i != len(op_list)-1):
                #    tpb.pearray.write_psum(psum_bank_dst, 0, self[i].ofmap_full_tile_sz, psum_temp)
                #psum_bank_src = psum_bank_dst
            else:
                print ("ERROR: %s is currently not yet implemented"%layer_type)
                exit(-1)
        return psum_temp
    # generate activation instruction and add it to instruction stream
    def gen_recip_waveop_inline(self, op, psum_bank_src, dst_is_psum, psum_bank_dst):
        layer_name = op.data["layer_name"]
        in_dtype = "float32"
        out_dtype = "float32"
        waveop_name = layer_name+"/Reciprocal"
        assert(dst_is_psum == True)
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : 'Reciprocal',
              'waveop_name'             : waveop_name,
              'layer_name'              : layer_name,
              'in_dtype'                : in_dtype,
              'out_dtype'               : out_dtype,
              'src_is_psum'             : True,
              'src_psum_bank_id'        : psum_bank_src,
              'src_psum_bank_offset'    : 0,
              'src_x_step'              : 1,
              'src_x_num'               : 1,
              'src_y_step'              : 1,
              'src_y_num'               : 1,
              'src_z_step'              : 1,
              'src_z_num'               : 1,
              'dst_is_psum'             : dst_is_psum, 
              'dst_psum_bank_id'        : psum_bank_dst,
              'dst_psum_bank_offset'    : 0,
              'dst_x_step'              : 1,
              'dst_x_num'               : 1,
              'dst_y_step'              : 1,
              'dst_y_num'               : 1,
              'dst_z_step'              : 1,
              'dst_z_num'               : 1,
              'num_partitions'          : 1
            }
        self.waveop_stream.add_linked(instr, [], -1)

    # generate scaleadd instruction and add it to instruction stream
    def gen_scaleadd_waveop_inline(self, op, tile_id, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_waveops, scale_val, add_val):
        batch_item = tile_id.n_id * op.Tn
        layer_name = op.data["layer_name"]
        # TODO: update in_dtype when src_is_psum is added
        in_dtype = "float32"
        out_dtype = "float32"
        # TODO: refactor to some class to determine in_dtype and out_dtype
        if (op.item_sz == 2 and not src_is_psum):
            in_dtype = "float16"
        elif (op.item_sz == 1 and not src_is_psum):
            print("ERROR: item_sz %d not yet supported"%op.item_sz)
            exit(-1)
        if (op.item_sz == 2 and not dst_is_psum):
            out_dtype = "float16"
        elif (op.item_sz == 1 and not dst_is_psum):
            print("ERROR: item_sz %d not yet supported"%op.item_sz)
            exit(-1)
        if (src_is_psum):
            print("ERROR: for scale/add waveop, cannot handle source coming from PSUM")
            exit(-1)
            src_sb_address = 0
        else:
            src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_address(op.ifmaps_file_params, batch_item, op.ifmap_wave_lower_addr[0])
        if (dst_is_psum):
            print("ERROR: for scale/add waveop, cannot handle destination PSUM")
            exit(-1)
        dst_x_num = op.ofmap_full_tilex_sz
        dst_y_step = op.E
        dst_y_num = op.ofmap_full_tiley_sz
        dst_z_step = (op.ofmaps_file_params.batch_item_partition_usage_sz//op.item_sz) if op.Tn > 1 else 1
        dst_z_num = op.Tn  # Need CNHW data format
        num_partitions = op.ofmap_count
        dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, batch_item, op.ofmap_tile_lower_addr[0])
        waveop_name = layer_name+"/ScaleAdd_"+tile_id.id_string            
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : 'ScaleAdd',
              'waveop_name'             : waveop_name,
              'layer_name'              : layer_name,
              'tile_id_format'          : tile_id.format,
              'tile_id'                 : tile_id.id_array,
              'in_dtype'                : in_dtype,
              'out_dtype'               : out_dtype,
              'src_is_psum'             : src_is_psum,
              'src_x_step'              : 1,
              'src_x_num'               : dst_x_num,
              'src_y_step'              : dst_y_step,
              'src_y_num'               : dst_y_num,
              'src_z_step'              : dst_z_step,
              'src_z_num'               : dst_z_num,
              'dst_is_psum'             : dst_is_psum,
              'dst_x_step'              : 1,
              'dst_x_num'               : dst_x_num,
              'dst_y_step'              : dst_y_step,
              'dst_y_num'               : dst_y_num,
              'dst_z_step'              : dst_z_step,
              'dst_z_num'               : dst_z_num,
              'num_partitions'          : num_partitions,
              'scale'                   : scale_val,
              'add'                     : add_val,
            }
        if src_is_psum:
            instr['src_psum_bank_id'] = psum_bank_src
            instr['src_psum_bank_offset'] = 0 
        else:                
            instr['src_sb_address'] = src_sb_address
            instr['src_start_at_mid_part'] = tile_id.m_id%2 == 1
        if dst_is_psum:
            instr['dst_psum_bank_id'] = psum_bank_dst
            instr['dst_psum_bank_offset'] = 0
        else:                
            instr['dst_sb_address'] = dst_sb_address
            instr['dst_start_at_mid_part'] = tile_id.m_id%2 == 1
        self.waveop_stream.add_linked(instr, dram_waveops, psum_bank_src if src_is_psum else -1)

    # generate activation instruction and add it to instruction stream
    def gen_act_waveop_inline(self, tpb, biasadd_op, act_op, conv_op, tile_id, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_start):
        layer_name = ""
        # kaena-452: load zeros into start of SB and use it for Activation instruction when there's no BiasAdd
        bias_add_en = True
        bias_sb_address = 0
        # TODO: update in_dtype when src_is_psum is added
        in_dtype = "float32"
        out_dtype = "float32"
        act_or_biasadd_op = None
        if (biasadd_op != None):
            act_or_biasadd_op = biasadd_op
            bias_add_en = True
            bias_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(biasadd_op.bias_file_params, 0, bias_start)
            layer_name = biasadd_op.data['layer_name']
            if (biasadd_op.item_sz == 2 and not src_is_psum):
                in_dtype = "float16"
            elif (biasadd_op.item_sz == 1 and not src_is_psum):
                print("ERROR: item_sz %d not yet supported"%biasadd_op.item_sz)
            if (biasadd_op.item_sz == 2 and not dst_is_psum):
                out_dtype = "float16"
            elif (biasadd_op.item_sz == 1 and not dst_is_psum):
                print("ERROR: item_sz %d not yet supported"%biasadd_op.item_sz)
        act_type = "Identity"    
        if (act_op != None):
            act_or_biasadd_op = act_op
            act_type = act_op.data['layer_type']
            layer_name = act_op.data['layer_name']
            # TODO: refactor to some class to determine in_dtype and out_dtype
            if (act_op.item_sz == 2 and not src_is_psum):
                in_dtype = "float16"
            elif (act_op.item_sz == 1 and not src_is_psum):
                print("ERROR: item_sz %d not yet supported"%act_op.item_sz)
            if (act_op.item_sz == 2 and not dst_is_psum):
                out_dtype = "float16"
            elif (act_op.item_sz == 1 and not dst_is_psum):
                print("ERROR: item_sz %d not yet supported"%act_op.item_sz)
        assert(act_or_biasadd_op != None)
        batch_item = tile_id.n_id * act_or_biasadd_op.Tn
        dst_x_num, dst_y_num, dst_z_num = 1, 1, 1
        dst_y_step, dst_z_step = 1, 1
        src_y_step, src_z_step = 1, 1
        num_partitions = PEArray.NUM_COLS
        if (conv_op != None):
            dst_x_num = conv_op.ofmap_cropped_tile_width
            dst_y_num = conv_op.ofmap_cropped_tile_height
            dst_z_num = conv_op.Tn
            if (dst_is_psum):
                dst_y_step = conv_op.ofmap_cropped_tile_width
                dst_z_step = dst_y_step * dst_y_num 
            else:                
                dst_y_step = conv_op.E
                dst_z_step = conv_op.ofmaps_file_params.batch_item_partition_usage_sz//conv_op.item_sz
            if src_is_psum:
                src_y_step = conv_op.ofmap_cropped_tile_width
                # Kaena-593: ensure no bubble during IFMAP streaming (packed pattern)
                if conv_op.repl_multiple_of_C > 1:
                    src_y_step = conv_op.W // conv_op.stride_x
                src_z_step = conv_op.ofmap_cropped_tile_width * conv_op.ofmap_cropped_tile_height
            else:
                src_y_step = conv_op.E
                src_z_step = conv_op.ofmaps_file_params.batch_item_partition_usage_sz//conv_op.item_sz
            num_partitions = conv_op.ofmap_count
        elif (act_or_biasadd_op !=  None):
            # unfused
            dst_x_num = act_or_biasadd_op.E
            dst_y_num = act_or_biasadd_op.F
            dst_z_num = act_or_biasadd_op.Tn
            dst_y_step = act_or_biasadd_op.E
            dst_z_step = act_or_biasadd_op.ofmaps_file_params.batch_item_partition_usage_sz//act_or_biasadd_op.item_sz
            src_y_step = act_or_biasadd_op.E
            src_z_step = act_or_biasadd_op.ofmaps_file_params.batch_item_partition_usage_sz//act_or_biasadd_op.item_sz
            num_partitions = act_or_biasadd_op.ofmap_count
        # SB start addresses
        if src_is_psum:
            src_sb_address = 0
        else:            
            src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(act_or_biasadd_op.ifmaps_file_params, batch_item, act_or_biasadd_op.ifmap_wave_lower_addr[0])
        if dst_is_psum:
            dst_sb_address = 0
        else:            
            if (conv_op != None):
                dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(act_or_biasadd_op.ofmaps_file_params, batch_item, conv_op.ofmap_tile_lower_addr[0])
            else:                
                dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(act_or_biasadd_op.ofmaps_file_params, batch_item, act_or_biasadd_op.ofmap_tile_lower_addr[0])
        waveop_name = layer_name+"/Activation_"+tile_id.id_string            
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : 'Activation',
              'waveop_name'             : waveop_name,
              'layer_name'              : layer_name,
              'tile_id_format'          : tile_id.format,
              'tile_id'                 : tile_id.id_array,
              'activation_func'         : act_type,
              'in_dtype'                : in_dtype,
              'bias_dtype'              : act_or_biasadd_op.ifmaps_file_params.data_type, 
              'out_dtype'               : out_dtype,
              'src_is_psum'             : src_is_psum,
              'src_x_step'              : 1,
              'src_x_num'               : dst_x_num,
              'src_y_step'              : src_y_step,
              'src_y_num'               : dst_y_num,
              'src_z_step'              : src_z_step,
              'src_z_num'               : dst_z_num,
              'dst_is_psum'             : dst_is_psum,
              'dst_x_step'              : 1,
              'dst_x_num'               : dst_x_num,
              'dst_y_step'              : dst_y_step,
              'dst_y_num'               : dst_y_num,
              'dst_z_step'              : dst_z_step,
              'dst_z_num'               : dst_z_num,
              'num_partitions'          : num_partitions,
              'bias_add_en'             : bias_add_en,
              'bias_sb_address'         : bias_sb_address,
              'bias_start_at_mid_part'  : tile_id.m_id%2 == 1,
            }
        if src_is_psum:
            instr['src_psum_bank_id'] = psum_bank_src
            instr['src_psum_bank_offset'] = 0 
        else:                
            instr['src_sb_address'] = src_sb_address
            instr['src_start_at_mid_part'] = tile_id.m_id%2 == 1
        if dst_is_psum:
            instr['dst_psum_bank_id'] = psum_bank_dst
            instr['dst_psum_bank_offset'] = 0
        else:                
            instr['dst_sb_address'] = dst_sb_address
            instr['dst_start_at_mid_part'] = tile_id.m_id%2 == 1
        tpb.waveop_stream.add_linked(instr, dram_bias_waveops, psum_bank_src if src_is_psum else -1)

    # generate ResAdd instruction and add it to instruction stream
    def gen_join_waveop_inline(self, tpb, op, conv_op, tile_id, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_resadd_waveops, data_start, start_at_mid_part):
        batch_item = tile_id.n_id * op.Tn
        in_a_dtype = "float32"
        in_b_dtype = "float32"
        out_dtype = "float32"
        if (op.item_sz == 2):
            in_a_dtype = "float16"
            in_b_dtype = "float32" # Source B is PSUM for now
            if (dst_is_psum):
                out_dtype = "float32"
            else:                
                out_dtype = "float16"
        elif (op.item_sz == 4):
            in_a_dtype = "float32"
            in_b_dtype = "float32"
            out_dtype = "float32"
        else:            
            print("ERROR: item_sz %d not yet supported"%self.conv_op.item_sz)

        # setup source/destination memory patterns (x step is always 1 here)
        dst_x_num, dst_y_num, dst_z_num = 1, 1, 1
        dst_y_step, dst_z_step = 1, 1
        src_a_y_step, src_a_z_step = 1, 1
        src_b_y_step, src_b_z_step = 1, 1
        num_partitions = PEArray.NUM_COLS
        if conv_op is not None:
            # fused
            dst_x_num = conv_op.ofmap_cropped_tile_width
            dst_y_num = conv_op.ofmap_cropped_tile_height
            dst_z_num = conv_op.Tn
            if dst_is_psum:
                dst_y_step = conv_op.ofmap_cropped_tile_width
                dst_z_step = conv_op.ofmap_cropped_tile_width * conv_op.ofmap_cropped_tile_height
            else:                
                dst_y_step = conv_op.E
                dst_z_step = conv_op.ofmaps_file_params.batch_item_partition_usage_sz//conv_op.item_sz
            # Source-B is PSUM if fused, or SB if unfused               
            if src_is_psum:
                src_b_y_step = conv_op.ofmap_cropped_tile_width
                src_b_z_step = conv_op.ofmap_cropped_tile_width * conv_op.ofmap_cropped_tile_height
            else:
                src_b_y_step = conv_op.E
                src_b_z_step = conv_op.ofmaps_file_params.batch_item_partition_usage_sz//conv_op.item_sz
            # Source-A (Residue) is always SB for now (TODO: make swappable for flexibility)              
            src_a_y_step = conv_op.E
            src_a_z_step = conv_op.ofmaps_file_params.batch_item_partition_usage_sz//conv_op.item_sz
            num_partitions = conv_op.ofmap_count
        else:
            # unfused
            dst_x_num = op.E
            dst_y_num = op.F
            dst_z_num = op.Tn
            dst_y_step = op.E
            dst_z_step = dst_y_step * dst_y_num
            src_b_y_step = op.E
            src_b_z_step = op.ofmaps_file_params.batch_item_partition_usage_sz//op.item_sz
            # Source-A (Residue) is always SB for now (TODO: make swappable for flexibility)              
            src_a_y_step = op.E
            src_a_z_step = op.ofmaps_file_params.batch_item_partition_usage_sz//op.item_sz
            num_partitions = op.ofmap_count
        # SB start addresses
        # Source-B is PSUM if fused, or SB if unfused
        if src_is_psum:
            src_b_sb_address = 0
        else:            
            src_b_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ifmaps_file_params, batch_item, data_start)
        # Source-A (Residue) is always SB for now (TODO: make swappable for flexibility)             
        src_a_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, batch_item, data_start)
        # Destination SB address
        if dst_is_psum:
            dst_sb_address = 0
        else:            
            if (conv_op != None):
                dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, batch_item, conv_op.ofmap_tile_lower_addr[0])
            else:                
                dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, batch_item, op.ofmap_tile_lower_addr[0])
        waveop_name = op.data['layer_name']+"/"+op.data['layer_type']+"_"+tile_id.id_string
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : "ResAdd", #op.data['layer_type'],
              'waveop_name'             : waveop_name,
              'multiply'                : op.data['layer_type'] == "Multiply",    # Hack to use ResAdd in old ISA to run Multiply 
              'layer_name'              : op.data['layer_name'],
              'tile_id_format'          : tile_id.format,
              'tile_id'                 : tile_id.id_array,
              'in_a_dtype'              : in_a_dtype,
              'in_b_dtype'              : in_b_dtype,
              'out_dtype'               : out_dtype,
              'src_a_is_psum'           : False,    # Source-A is always SB for now (TODO: make swappable for flexibility)
              'src_a_sb_address'        : src_a_sb_address,
              'src_a_start_at_mid_part' : start_at_mid_part,
              'src_a_x_num'             : dst_x_num,
              'src_a_y_num'             : dst_y_num,
              'src_a_z_num'             : dst_z_num,
              'src_a_x_step'            : 1,
              'src_a_y_step'            : src_a_y_step,
              'src_a_z_step'            : src_a_z_step,
              'src_b_is_psum'           : src_is_psum,
              'src_b_psum_bank_id'      : psum_bank_src,
              'src_b_psum_bank_offset'  : 0,
              'src_b_sb_address'        : src_b_sb_address,
              'src_b_start_at_mid_part' : start_at_mid_part,
              'src_b_x_num'             : dst_x_num,
              'src_b_y_num'             : dst_y_num,
              'src_b_z_num'             : dst_z_num,
              'src_b_x_step'            : 1,
              'src_b_y_step'            : src_b_y_step,
              'src_b_z_step'            : src_b_z_step,
              'dst_is_psum'             : dst_is_psum,
              'dst_psum_bank_id'        : psum_bank_dst,
              'dst_psum_bank_offset'    : 0,
              'dst_sb_address'          : dst_sb_address,
              'dst_start_at_mid_part'   : start_at_mid_part,
              'dst_x_num'               : dst_x_num,
              'dst_y_num'               : dst_y_num,
              'dst_z_num'               : dst_z_num,
              'dst_x_step'              : 1,
              'dst_y_step'              : dst_y_step,
              'dst_z_step'              : dst_z_step,
              'num_partitions'          : num_partitions,
            }
        if src_is_psum:
            instr['src_b_psum_bank_id'] = psum_bank_src 
            instr['src_b_psum_bank_offset'] = 0 
        else:                
            instr['src_b_sb_address'] = src_b_sb_address
            instr['src_b_start_at_mid_part'] = start_at_mid_part 
        if dst_is_psum:
            instr['dst_psum_bank_id'] = psum_bank_dst
            instr['dst_psum_bank_offset'] = 0 
        else:                
            instr['dst_sb_address'] = dst_sb_address
            instr['dst_start_at_mid_part'] = start_at_mid_part 
        tpb.waveop_stream.add_linked(instr, dram_resadd_waveops, psum_bank_src if src_is_psum else -1)

    def gen_fused_pool_waveop_inline (self, tpb, tile_id, psum_bank_src, start_at_mid_part):
        for z in range(self.pool_op.Tn):
            pool_waveop = self.gen_pool_waveop(tpb, tile_id, True, psum_bank_src, start_at_mid_part, z)
            tpb.waveop_stream.add_linked(pool_waveop, [], psum_bank_src)

    def gen_unfused_pool_waveop_inline (self, tpb, tile_id, dram_waveops, start_at_mid_part):
        for z in range(self.pool_op.Tn):
            pool_waveop = self.gen_pool_waveop(tpb, tile_id, False, 0, start_at_mid_part, z)
            tpb.waveop_stream.add_linked(pool_waveop, dram_waveops if z==0 else [], -1)

    # Execute softmax (second part, which includes Sum, Reciprocate, Scale)
    def execute_softmax2(self, tpb, inputs, result_file):
        # create and save ones as weights file, then load them back
        ones_shape = [self[0].C, 1, 1, 1]
        ones_tensor = np.ones(ones_shape, dtype=self[0].data_type)
        ones_file = self[0].data['ref_file'].replace(".npy", "-ones.npy")
        if (not args.inference):
            np.save(ones_file, ones_tensor)
        weights = []
        # TODO: needs better way to load ones into weights region
        #if (self.has_conv):
        #    self[0].data['kernel_file'] = ones_file
        #    self[0].data['kernel_format'] = "CRSM"
        #    self[0].data['kernel_shape'] = ones_shape
        #    weights = self.statebuffer.file_mapper.load_data(self.conv_op)

        # reallocate statebuffer resources
        #self.statebuffer.reallocate_capacities()

        # initial psum bank is 0
        self.conv_op.set_psum_bank(tpb.pearray.last_psum_bank_used)
        # start tensor computation by clearing psum bank
        psum_add = False                               

        # use conv to sum the exponential results
        # wave loop ordering scheme: nmhwcRS
        for n_id in range(self.conv_op.n):
            for m_id in range(self.conv_op.m):
                for h_id in range(self.conv_op.h):
                    for w_id in range(self.conv_op.w):
                        tile_id = TileID(n_id, m_id, h_id, w_id, self.conv_op.n, self.conv_op.m, self.conv_op.h, self.conv_op.w)
                        # compute ofmap tile information (tile startx, starty, height, width)
                        self.conv_op.compute_ofmap_tile_info(tile_id)
                        self.conv_op.compute_tile_weight_bounds(weights, tile_id)
                        # loops for constructing a tile
                        for c_id in range(self.conv_op.c):
                            for r_id in range(self.conv_op.R):
                                for s_id in range(self.conv_op.S):
                                    wave_id = WaveID(n_id, m_id, h_id, w_id, c_id, r_id, s_id)
                                    if (self.parent.args.debug > 2): print (wave_id.id_array)
                                    # execute PEArray matrix multiply, and add to PSUM after first wave
                                    if (self.execute_matmul_waveop(self, wave_id, inputs, weights, psum_add, 1)):
                                        psum_add = True
                        # tile is done                                   
                        self.waveop_stream.last_main_waveop['stop_tensor_calc'] = True
                        # extract PSUM data
                        psum_bank_src = self.conv_op.get_psum_bank()
                        psum_temp = tpb.pearray.extract_psum(psum_bank_src, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn)
                        # go through the remaining operations
                        psum_temp = self.pool.reciprocate(psum_temp, self.conv_op.M)
                        psum_bank_dst = psum_bank_src
                        tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, psum_temp)
                        tpb.gen_recip_waveop_inline(self.conv_op, psum_bank_src, True, psum_bank_dst)
                        psum_bank_src = psum_bank_dst
                        # loops for final scaling
                        for c_id in range(ceildiv(self.conv_op.C, PEArray.NUM_COLS)):
                            wave_id = WaveID(n_id, m_id, h_id, w_id, c_id, 0, 0)
                            pearray_packed_ifmaps = self.conv_op.pack_wave_ifmaps(inputs, wave_id, 1, for_softmax=True)
                            scale_val = tpb.pearray.extract_psum(psum_bank_src, 0, 1)[0,0]
                            psum_temp = self.pool.scale(pearray_packed_ifmaps, scale_val)
                            # if operation is the last one, dump current result into a portion of final result
                            # use c_id instead of m_id because we collapsed M to 1 to do the summation
                            output_params_op = self.conv_op
                            dram_output_waveops = []                            
                            for z in range(self.conv_op.Tn):
                                for j in range(PEArray.NUM_COLS):
                                    M_idx = wave_id.c_id * PEArray.NUM_COLS + j
                                    if (M_idx >= output_params_op.C):
                                        break
                                    else:
                                        # For now, multiply zeros, and at the ofmap, extract tile with zeros, then clip
                                        result_tile_tmp = (psum_temp[z*output_params_op.ofmap_full_tile_sz : (z+1)*output_params_op.ofmap_full_tile_sz, j])
                                        result_tile = result_tile_tmp.reshape((output_params_op.ofmap_full_tiley_sz, output_params_op.ofmap_full_tilex_sz))
                                        #DBG_DUMP_ARRAY("M_idx %d Intermediate result (FP32): "%M_idx, result_tile)
                                        # NCHW
                                        result[n_id * output_params_op.Tn + z, 
                                                M_idx, 
                                                output_params_op.ofmap_tile_y_start : output_params_op.ofmap_tile_y_start + output_params_op.ofmap_cropped_tile_height, 
                                                output_params_op.ofmap_tile_x_start : output_params_op.ofmap_tile_x_start + output_params_op.ofmap_cropped_tile_width]\
                                            = result_tile[0:output_params_op.ofmap_cropped_tile_height, 0:output_params_op.ofmap_cropped_tile_width]
                                # for scheduling, map resulting tile into portion of atom that is itself mapped to a portion in DRAM (file)
                                # TODO: fix waveop generation
                                #dram_output_waveops += tpb.statebuffer.circbuf_scratch.write_data_region(
                                #                            tile_id, 
                                #                            output_params_op.ofmap_tile_lower_addr[z], 
                                #                            output_params_op.ofmap_tile_upper_addr[z], 
                                #                            output_params_op.ifmap_count,   # Do we have to use IFMAP count here?
                                #                            self.waveop_stream.last_main_waveop)
                       # The scale_add destination need to be adjusted after the above writes to data region
                        if (self.waveop_stream.last_main_waveop['waveop_type'] == "ScaleAdd"):
                            sb_addr = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.conv_op.ofmaps_file_params, 0, output_params_op.ofmap_tile_lower_addr[0])
                            self.waveop_stream.last_main_waveop['dst_sb_address'] = sb_addr
                        self.waveop_stream.add_outputs(dram_output_waveops)
                        if self.args.abstract_mem:
                            if len(dram_output_waveops) > 0:
                                self.waveop_stream.last_main_waveop = None

                        # Advance to new bank (ping-pong between 0 and 1) for PEArray, while the old bank is being processed by other engines
                        self.conv_op.set_psum_bank((self.conv_op.get_psum_bank()+1)%4)
                        tpb.pearray.last_psum_bank_used = self.conv_op.get_psum_bank()
                        psum_add = False

        return result

    # Execute an unfused pooling operator
    def execute_unfused_pool_op(self, tpb, batch_item):
        inputs = self.first_op.ifmaps_file_params.dram_data

        # load bias values
        bias = []
        if (self.has_biasadd):
            bias_temp = self.biasadd_op.bias_file_params.dram_data
            bias = bias_temp.flatten()

        # save result to create a scratch space (in DRAM), then use circular buffer load to populate params
        result = self.last_op.ofmaps_file_params.dram_data

        # wave loop ordering scheme: nmhw
        pool_op = self[0]
        n_id = batch_item // pool_op.Tn
        if True:
            for m_id in range(pool_op.m):
                for h_id in range(pool_op.h):
                    for w_id in range(pool_op.w):
                        tile_id = TileID(n_id, m_id, h_id, w_id, pool_op.n, pool_op.m, pool_op.h, pool_op.w)
                        pool_op.compute_ofmap_tile_info(tile_id)
                        # set r_id and s_id in wave_id to zero since we are not doing convolution
                        wave_id = WaveID(n_id, m_id, h_id, w_id, 0, 0, 0)
                        psum_fake = pool_op.pack_wave_ifmaps_unfused_pooling(inputs, wave_id)
                        input_tiley = pool_op.ifmap_wave_upper_coordy[0] - pool_op.ifmap_wave_lower_coordy[0] + 1
                        input_tilex = pool_op.ifmap_wave_upper_coordx[0] - pool_op.ifmap_wave_lower_coordx[0] + 1
                        output_tiley = pool_op.ofmap_full_tiley_sz
                        output_tilex = pool_op.ofmap_full_tilex_sz
                        psum_fake_extract = psum_fake [0:input_tiley*input_tilex*pool_op.Tn, :]
                        if (pool_op.data['layer_type'] == "AvgPool" or pool_op.data['layer_type'] == "MaxPool"):
                            psum_temp = tpb.pool.pool(pool_op.data['layer_type'], psum_fake_extract, pool_op.stride_x, pool_op.pool_window_y, pool_op.Tn, input_tilex, input_tiley, output_tilex, output_tiley)
                        elif (pool_op.data['layer_type'] == "Multiply" or pool_op.data['layer_type'] == "ResAdd"):
                            if ("mul_scalar" in pool_op.data):
                                assert (pool_op.data['layer_type'] == "Multiply")
                                psum_temp = tpb.pool.scale(psum_fake_extract, pool_op.data['mul_scalar'])
                            else:
                                dram_resadd_waveops = []
                                for z in range(pool_op.Tn):
                                    fmap_count = pool_op.ofmap_count
                                    if (tile_id.m_id+1 != tile_id.m):
                                        fmap_count = 2*pool_op.ofmap_count
                                    # TODO: fix waveop generation    
                                    #dram_resadd_waveops += tpb.statebuffer.circbuf_residue.read_data_region(
                                    #                                wave_id,
                                    #                                pool_op.ofmap_tile_lower_addr[z], 
                                    #                                pool_op.ofmap_tile_upper_addr[z], 
                                    #                                fmap_count,
                                    #                                ifmaps_replicate = False,
                                    #                                start_at_mid_part = False)
                                residue_ifmaps = np.zeros((input_tiley * input_tilex * pool_op.Tn, PEArray.NUM_COLS), dtype=np.float32)
                                for z in range(pool_op.Tn):
                                    for j in range(PEArray.NUM_COLS):
                                        M_idx = tile_id.m_id * PEArray.NUM_COLS + j
                                        if (M_idx >= pool_op.M):
                                            break
                                        else:
                                            # NCHW
                                            residue_tile_ifmap = np.zeros((pool_op.ofmap_cropped_tile_height, pool_op.ofmap_cropped_tile_width), dtype=np.float32)
                                            residue_tile_ifmap[0:pool_op.ofmap_cropped_tile_height, 0:pool_op.ofmap_cropped_tile_width] = tpb.statebuffer.circbuf_residue.dram_data[
                                                    tile_id.n_id * pool_op.Tn + z, 
                                                    M_idx, 
                                                    pool_op.ofmap_tile_y_start : pool_op.ofmap_tile_y_start + pool_op.ofmap_cropped_tile_height, 
                                                    pool_op.ofmap_tile_x_start : pool_op.ofmap_tile_x_start + pool_op.ofmap_cropped_tile_width]
                                            residue_ifmaps[z * input_tiley * input_tilex : (z+1) * input_tiley * input_tilex, j] = residue_tile_ifmap.flatten()
                                if (pool_op.data['layer_type'] == "ResAdd"):
                                    psum_temp = tpb.pool.resadd(psum_fake_extract, residue_ifmaps)
                                elif (pool_op.data['layer_type'] == "Multiply"):                                    
                                    psum_temp = self.pool.multiply(psum_fake_extract, residue_ifmaps)
                                else:
                                    print("ERROR: don't know how to handle vector op %s for layer %s"%(pool_op.data['layer_type'], pool_op.data['layer_name']))
                        elif (pool_op.data['layer_type'] == "BiasAdd"):
                            bias_chan_start = tile_id.m_id * PEArray.NUM_COLS
                            bias_chan_end = min(bias_chan_start + PEArray.NUM_COLS, pool_op.M)
                            bias_extracted = np.zeros(PEArray.NUM_COLS)
                            bias_extracted[0 : bias_chan_end - bias_chan_start] = bias[bias_chan_start : bias_chan_end]
                            bias_addr = bias_chan_start * pool_op.item_sz
                            dram_bias_waveops = tpb.statebuffer.file_mapper.read_file_data_region(
                                                            tpb.waveop_stream.nonload_waveop_count,
                                                            tpb.waveop_stream.nonload_waveop_list,
                                                            pool_op.bias_file_params,
                                                            0,  # batch_item is not needed for bias
                                                            bias_addr,
                                                            pool_op.item_sz)
                            psum_temp = tpb.activate.biasadd(psum_fake_extract, bias_extracted)
                        elif re.search(r"Relu|Tanh|Sigmoid|Exp|Identity|Lrelu|Prelu", pool_op.data['layer_type']):
                            psum_temp = tpb.activate.act(pool_op.data['layer_type'], psum_fake_extract)
                        else:
                            print("ERROR: cannot execute %s in execute_unfused_pool_op"%pool_op.data['layer_type'])
                            exit(-1)

                        # TODO: fix waveop generation
                        dram_ifmaps_waveops = []
                        latest_accessor = -1
                        for z in range(pool_op.Tn):
                           if (tile_id.m_id%2 == 0):
                                fmap_count = pool_op.ifmap_count
                                if (tile_id.m_id+1 != tile_id.m):
                                    fmap_count = 2*pool_op.ifmap_count
                                (last_writer, last_reader, waveops) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                            tpb.waveop_stream.nonload_waveop_count,
                                                            tpb.waveop_stream.nonload_waveop_list,
                                                            pool_op.ifmaps_file_params,
                                                            batch_item + z,
                                                            pool_op.ifmap_wave_lower_addr[z], 
                                                            pool_op.ifmap_wave_upper_addr[z] - pool_op.ifmap_wave_lower_addr[z] + pool_op.item_sz)
                                dram_ifmaps_waveops += waveops
                                latest_accessor = max(last_writer, latest_accessor) # don't include readers since this pool is a reader, and we don't need to add RAR dependency

                        # TODO: roll this code into read_file_data_region
                        prev_waveops = []
                        if dram_ifmaps_waveops == []:
                            if self.args.relax_dependencies:
                                # kaena-403/449 hack: reduce dependencies to prevent event overflow
                                latest_accessor = -1
                            if latest_accessor >= 0:
                                accessor_name = tpb.waveop_stream.nonload_waveop_list[latest_accessor]['waveop_name']
                                if accessor_name not in prev_waveops:
                                    prev_waveops.append(accessor_name)

                        start_at_mid_part = tile_id.m_id%2 == 1
                        if (pool_op.data['layer_type'] == "AvgPool" or pool_op.data['layer_type'] == "MaxPool"):
                            self.gen_unfused_pool_waveop_inline(tpb, tile_id, dram_ifmaps_waveops, start_at_mid_part)
                       # elif (pool_op.data['layer_type'] == "Multiply" or pool_op.data['layer_type'] == "ResAdd"):
                       #     if ("mul_scalar" in pool_op.data):
                       #         self.gen_scaleadd_waveop_inline(pool_op, tile_id, False, 0, False, 0, dram_ifmaps_waveops, pool_op.data['mul_scalar'], 0.0)
                       #     else:
                       #         self.gen_join_waveop_inline(tpb, pool_op, None, tile_id, False, 0, False, 0, dram_ifmaps_waveops+dram_resadd_waveops, pool_op.ofmap_tile_lower_addr[0], start_at_mid_part)
                       # elif (pool_op.data['layer_type'] == "BiasAdd"): 
                       #     self.gen_act_waveop_inline(tpb, pool_op, None, None, tile_id, False, 0, False, 0, dram_ifmaps_waveops + dram_bias_waveops, bias_addr)
                       #     #tpb.statebuffer.circbuf_bias.free_data_region(bias_addr, bias_addr, self.waveop_stream.last_main_waveop)
                        else:                            
                            self.gen_act_waveop_inline(tpb, None, pool_op, None, tile_id, False, 0, False, 0, dram_ifmaps_waveops, 0)

                        tpb.waveop_stream.last_main_waveop['previous_waveops'] += prev_waveops

                        dram_output_waveops = []                            
                        latest_accessor = -1
                        for z in range(pool_op.Tn):
                            #if (tile_id.m_id+1 == tile_id.m or tile_id.m_id%2 == 1):
                            #    tpb.statebuffer.circbuf_ifmaps.free_data_region(pool_op.ifmap_wave_lower_addr[z], pool_op.ifmap_wave_upper_addr[z], self.waveop_stream.last_main_waveop)
                            for j in range(PEArray.NUM_COLS):
                                M_idx = wave_id.m_id * PEArray.NUM_COLS + j
                                if (M_idx >= pool_op.M):
                                    break
                                else:
                                    # For now, multiply zeros, and at the ofmap, extract tile with zeros, then clip
                                    #result_tile_tmp = (psum_temp[z * pool_op.ofmap_full_tile_sz : (z+1) * pool_op.ofmap_full_tile_sz, j])
                                    #result_tile = result_tile_tmp.reshape((pool_op.ofmap_full_tiley_sz, pool_op.ofmap_full_tilex_sz))
                                    result_tile_tmp = (psum_temp[   z * pool_op.ofmap_cropped_tile_height * pool_op.ofmap_cropped_tile_width 
                                                                : (z+1) * pool_op.ofmap_cropped_tile_height * pool_op.ofmap_cropped_tile_width, j])
                                    result_tile = result_tile_tmp.reshape((pool_op.ofmap_cropped_tile_height, pool_op.ofmap_cropped_tile_width))
                                    # NCHW
                                    result[n_id * pool_op.Tn + z, 
                                            M_idx, 
                                            pool_op.ofmap_tile_y_start : pool_op.ofmap_tile_y_start + pool_op.ofmap_cropped_tile_height, 
                                            pool_op.ofmap_tile_x_start : pool_op.ofmap_tile_x_start + pool_op.ofmap_cropped_tile_width]\
                                        = result_tile[0:pool_op.ofmap_cropped_tile_height, 0:pool_op.ofmap_cropped_tile_width]
                            # for scheduling, map resulting tile into portion of atom that is itself mapped to a portion in DRAM (file)
                            # only record the writer to SB chunks in below code; use flush_file to write chunks to DRAM
                            (last_writer, last_reader, waveops) = tpb.statebuffer.file_mapper.write_file_data_region(
                                                        tpb.waveop_stream.nonload_waveop_count - 1,    # adjust since pool waveop already generated
                                                        tpb.waveop_stream.nonload_waveop_list,
                                                        pool_op.ofmaps_file_params,
                                                        batch_item + z,
                                                        pool_op.ofmap_tile_lower_addr[z], 
                                                        pool_op.ofmap_tile_upper_addr[z] - pool_op.ofmap_tile_lower_addr[z] + pool_op.item_sz, 
                                                        start_at_mid_part)
                            assert(len(waveops) == 0)                            
                            # TODO: roll this code into write_file_data_region
                            latest_accessor = max(last_writer, last_reader, latest_accessor)

                            if (self.args.debug > 3): print("TRACE execute_unfused_pool_op %s: tile %s done, ifmap_tile_lower_addr %d ifmap_tile_upper_addr %d psum_bank %d, ofmap_tile_lower_addr %d ofmap_tile_upper_addr %dx"\
                                        %(pool_op.data["layer_name"], tile_id.id_string, pool_op.ifmap_wave_lower_addr[z], pool_op.ifmap_wave_upper_addr[z], -1, pool_op.ofmap_tile_lower_addr[z], pool_op.ofmap_tile_upper_addr[z]))

                        prev_waveops = tpb.waveop_stream.last_main_waveop['previous_waveops']
                        if latest_accessor >= 0:
                            accessor_name = tpb.waveop_stream.nonload_waveop_list[latest_accessor]['waveop_name']
                            if accessor_name not in prev_waveops:
                                prev_waveops.append(accessor_name)
                        #if self.args.abstract_mem:
                        #    if len(dram_output_waveops) > 0:
                        #        self.waveop_stream.last_main_waveop = None
        return result

    # Execute conv and other operations in list: for each op, load parameters and perform op with input
    def execute_conv_ops(self, tpb, batch_item):
        inputs = self.first_op.ifmaps_file_params.dram_data
        weights = self.conv_op.weights_file_params.dram_data

        # load bias values
        bias = []
        if (self.has_biasadd):
            bias_temp = self.biasadd_op.bias_file_params.dram_data
            bias = bias_temp.flatten()

        # save result to create a scratch space (in DRAM), then use circular buffer load to populate params
        result = self.last_op.ofmaps_file_params.dram_data

        # initial psum bank is 0
        self.conv_op.set_psum_bank(tpb.pearray.last_psum_bank_used)
        # start tensor computation by clearing psum bank
        psum_add = False                               

        # wave loop ordering scheme: nmhwcRS
        n_id = batch_item // self.conv_op.Tn
        if True:
            for m_id in range(self.conv_op.m):
                for h_id in range(self.conv_op.h):
                    for w_id in range(self.conv_op.w):
                        tile_id = TileID(n_id, m_id, h_id, w_id, self.conv_op.n, self.conv_op.m, self.conv_op.h, self.conv_op.w)
                        # compute ofmap tile information (tile startx, starty, height, width)
                        self.conv_op.compute_ofmap_tile_info(tile_id)
                        self.conv_op.compute_tile_weight_bounds(weights, tile_id)
                        # loops for constructing a tile
                        for c_id in range(self.conv_op.c):
                            r_id = 0
                            s_id = 0
                            remaining_filter_elems = self.conv_op.RS
                            while r_id < self.conv_op.weights_file_params.file_dims.R:
                                while s_id < self.conv_op.weights_file_params.file_dims.S:
                                    wave_id = WaveID(n_id, m_id, h_id, w_id, c_id, r_id, s_id)
                                    # execute PEArray matrix multiply, and add to PSUM after first wave
                                    repl_multiple_per_wave = 1
                                    if self.conv_op.repl_multiple_of_C > 1:
                                        repl_multiple_per_wave = min(remaining_filter_elems, self.conv_op.repl_multiple_of_C)
                                        remaining_filter_elems -= self.conv_op.repl_multiple_of_C
                                    if (self.execute_matmul_waveop(tpb, wave_id, inputs, weights, psum_add, repl_multiple_per_wave)):
                                        psum_add = True
                                    s_id += self.conv_op.repl_multiple_of_C
                                r_id += s_id//self.conv_op.S
                                s_id = s_id%self.conv_op.S
                        # tile is done                                   
                        # extract PSUM data
                        psum_bank_src = self.conv_op.get_psum_bank()
                        psum_temp = tpb.pearray.extract_psum(psum_bank_src, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn)
                        #x = DBG_DUMP_PSUM_COL("PSUM after PEArray: ", psum_temp, 0)
                        # go through the remaining operations
                        psum_temp = self.execute_tile_ops(tpb, wave_id, tile_id, psum_bank_src, bias, psum_temp)
                        #x = DBG_DUMP_PSUM_COL("PSUM after PEArray: ", psum_temp, 0)
                        # if operation is the last one, dump current result into a portion of final result
                        output_params_op = self.conv_op
                        if (self.has_pool):
                            output_params_op = self.pool_op
                        dram_output_waveops = []                            
                        latest_accessor = -1
                        for z in range(self.conv_op.Tn):
                            for j in range(PEArray.NUM_COLS):
                                M_idx = wave_id.m_id * PEArray.NUM_COLS + j
                                if (M_idx >= output_params_op.M):
                                    break
                                else:
                                    # For now, multiply zeros, and at the ofmap, extract tile with zeros, then clip
                                    result_tile_tmp = (psum_temp[z * output_params_op.ofmap_full_tile_sz : (z+1) * output_params_op.ofmap_full_tile_sz, j])
                                    result_tile = result_tile_tmp.reshape((output_params_op.ofmap_full_tiley_sz, output_params_op.ofmap_full_tilex_sz))
                                    #DBG_DUMP_ARRAY("Intermediate result: ", result_tile)
                                    # NCHW
                                    result[n_id * output_params_op.Tn + z, 
                                            M_idx, 
                                            output_params_op.ofmap_tile_y_start : output_params_op.ofmap_tile_y_start + output_params_op.ofmap_cropped_tile_height, 
                                            output_params_op.ofmap_tile_x_start : output_params_op.ofmap_tile_x_start + output_params_op.ofmap_cropped_tile_width]\
                                        = result_tile[0:output_params_op.ofmap_cropped_tile_height, 0:output_params_op.ofmap_cropped_tile_width]
                            # for scheduling, map resulting tile into portion of atom that is itself mapped to a portion in DRAM (file)
                            # only record the writer to SB chunks in below code; use flush_file to write chunks to DRAM
                            start_at_mid_part = tile_id.m_id%2 == 1
                            (last_writer, last_reader, waveops) = tpb.statebuffer.file_mapper.write_file_data_region(
                                                        tpb.waveop_stream.nonload_waveop_count - 1,    # adjust since pool waveop already generated
                                                        tpb.waveop_stream.nonload_waveop_list,
                                                        self.last_op.ofmaps_file_params,
                                                        batch_item + z,
                                                        output_params_op.ofmap_tile_lower_addr[z], 
                                                        output_params_op.ofmap_tile_upper_addr[z] - output_params_op.ofmap_tile_lower_addr[z] + output_params_op.item_sz, 
                                                        start_at_mid_part)
                            assert(len(waveops) == 0)                            
                            latest_accessor = max(last_writer, last_reader, latest_accessor) 

                        # consider all Tn batch items together to avoid redundant edges
                        # TODO: roll this code into write_file_data_region
                        prev_waveops = tpb.waveop_stream.last_psum_waveop[psum_bank_src]['previous_waveops']
                        #if self.args.relax_dependencies:
                            # kaena-403/449 hack: reduce dependencies to prevent event overflow
                        #    latest_accessor = -1
                        if latest_accessor >= 0:
                            accessor_name = tpb.waveop_stream.nonload_waveop_list[latest_accessor]['waveop_name']
                            if accessor_name not in prev_waveops and accessor_name != tpb.waveop_stream.last_psum_waveop[psum_bank_src]['waveop_name']:
                                prev_waveops.append(accessor_name)

                        #if self.parent.args.abstract_mem:
                        #    if len(dram_output_waveops) > 0:
                        #        tpb.waveop_stream.last_psum_waveop[psum_bank_src] = None
                        # Advance to new bank (ping-pong between 0 and 1) for PEArray, while the old bank is being processed by other engines
                        self.conv_op.set_psum_bank((self.conv_op.get_psum_bank()+1)%4)
                        tpb.pearray.last_psum_bank_used = self.conv_op.get_psum_bank()
                        psum_add = False
        return result               
