"""
Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
"""

"""Fused operations execution, including verification and waveop generation
"""

import re
import os
import sys
import copy
import numpy as np
import math
from me_models import PEArray
from me_utils import ceildiv
from me_utils import align_addr_8B
from me_utils import Coord
from me_utils import Dim2D
from me_utils import Rect
from me_utils import ShapeDims
from me_utils import FileParams
from me_pool import Pool
from me_concat import Concat

sys.path.insert(0, os.environ["KAENA_PATH"] + "/compiler/tffe")
from NpUtils import NpUtils as npu

"""Macros for dumping arrays
"""
def DBG_DUMP_ARRAY(msg, a):
    print (msg, "\n" , a)
    return a

def DBG_DUMP_PSUM_COL(msg, psum, col):
    x = psum[:, col]
    print (msg, "\n" , x)
    return x

"""PE array wave of a tile
"""
class PEWave:
    def __init__(self, tile, c_id, r_id, s_id, stridedslice_chan_offset=0):
        self.format = "nmhwcrs"
        self.tile = tile
        self.c_id, self.r_id, self.s_id = c_id, r_id, s_id
        self.id_array = self.tile.id_array  + [self.c_id, self.r_id, self.s_id]
        self.id_string = self.tile.id_string + "_c%d_r%d_s%d"%(self.c_id, self.r_id, self.s_id)
        self.ifmap_channel_start = self.c_id * PEArray.NUM_ROWS + stridedslice_chan_offset
        self.ifmap_channel_stop = min(self.tile.file_params.file_dims.C,
                                      self.ifmap_channel_start + PEArray.NUM_ROWS)
        self.ifmap_channel_count = self.ifmap_channel_stop - self.ifmap_channel_start
        self.subtile_psum_offset = 0
        self.subtile_rect = tile.tile_rect

    def __str__(self):
        return self.id_string

    def get_sb_fmap_count (self):
        if self.tile.is_pe_input:
            return self.ifmap_channel_count
        else:            
            return self.tile.channel_count

    def get_subtile_file_addrs(self):
        lower_addrs = [-1 for z in range(self.tile.Tn)]
        upper_addrs = [-1 for z in range(self.tile.Tn)]
        if not self.subtile_rect.is_empty:
            for z in range(self.tile.Tn):
                if self.tile.is_pe_input:
                    channel_start = self.ifmap_channel_start
                else:
                    channel_start = self.tile.channel_start
                lower_addrs[z] = (self.tile.file_params.ravel_nchw(
                    self.tile.n_id * self.tile.Tn + z,
                    channel_start,
                    self.subtile_rect.lower.y,
                    self.subtile_rect.lower.x))
                upper_addrs[z] = (self.tile.file_params.ravel_nchw(
                    self.tile.n_id * self.tile.Tn + z,
                    channel_start,
                    self.subtile_rect.upper.y,
                    self.subtile_rect.upper.x))
        return (lower_addrs, upper_addrs)

    def get_subtile_data_from_file(self, flatten=False):
        if not self.subtile_rect.is_empty:
            for z in range(self.tile.Tn):
                if self.tile.is_pe_input:
                    channel_start = self.ifmap_channel_start
                    channel_stop = self.ifmap_channel_stop
                else:
                    channel_start = self.tile.channel_start
                    channel_stop = self.tile.channel_stop
            # NCHW
            tile_data = self.tile.file_params.dram_data[self.tile.n_start : self.tile.n_stop,
                                              channel_start : channel_stop,
                                              self.subtile_rect.lower.y : self.subtile_rect.upper.y + 1,
                                              self.subtile_rect.lower.x : self.subtile_rect.upper.x + 1]

            if flatten:
                tile_rect_sz = self.subtile_rect.dim2d.get_tot_size()
                tile_data_flat = np.zeros((self.tile.Tn * tile_rect_sz, tile_data.shape[1]))
                for z in range(self.tile.Tn):
                    for i in range(tile_data.shape[1]):
                        tile_data_flat[z * tile_rect_sz : (z+1) * tile_rect_sz, i] = tile_data[z, i, :, :].flatten()
                tile_data = tile_data_flat                    
            return tile_data
        else:
            raise RuntimeError("Cannot extract a view using empty rectangle ", self.subtile_rect)

    def set_waveop_pattern(self, file_mapper, waveop, prefix, psum_bank_id, sb_address, start_at_mid_part):      
        prefix2 = prefix.replace("src","in").replace("dst","out")
        x_step = 1
        x_num = self.subtile_rect.dim2d.x
        y_num = self.subtile_rect.dim2d.y
        z_num = 1 if waveop['waveop_type'] == 'Pool' else self.tile.Tn 
        batch_item = self.tile.n_id * self.tile.Tn
        if psum_bank_id >= 0:
            dtype  = "float32"
            y_step = self.subtile_rect.dim2d.x
            z_step = y_step * y_num if z_num > 1 else 1
            waveop[prefix + "_psum_bank_id"]      = psum_bank_id
            waveop[prefix + "_psum_bank_offset"]  = self.subtile_psum_offset
        else:   # negative psum_bank_id indicates SB
            dtype  = self.tile.file_params.data_type
            y_step = self.tile.file_params.file_dims.W
            z_step = self.tile.file_params.batch_item_partition_usage_elems_padded if z_num > 1 else 1
            waveop[prefix + "_start_at_mid_part"] = start_at_mid_part
            waveop[prefix + "_sb_address"]        = sb_address
        waveop[prefix2 + "_dtype"] = dtype
        waveop[prefix + "_is_psum"] = psum_bank_id >= 0
        waveop[prefix + "_x_step"] = x_step
        waveop[prefix + "_y_step"] = y_step
        waveop[prefix + "_z_step"] = z_step
        waveop[prefix + "_x_num"] = x_num
        waveop[prefix + "_y_num"] = y_num
        waveop[prefix + "_z_num"] = z_num
        # TODO: add pattern for pooling


""" Pool subtile
"""
class PoolSubtile(PEWave):
    def __init__(self, tile, subtile_rect, window):
        self.format = "nmhw"
        self.tile = tile
        self.subtile_rect = subtile_rect
        self.window = window
        self.id_array = self.tile.id_array 
        self.id_string = self.tile.id_string #+ "_lx%d_ly%d_ux%d_uy%d"%(self.c_id, self.r_id, self.s_id)
        (self.lower_addr, self.upper_addr) = self.get_subtile_file_addrs()
        assert(tile.is_pe_input == False)

""" FMAP tile object
"""
class Tile:

    def __init__(self, tile_id, file_params, Tn, is_pe_input=False, stridedslice_chan_offset=0):
        self.format = "nmhw"
        (self.n_id, self.m_id, self.h_id, self.w_id, self.n, self.m, self.h, self.w) = tile_id
        self.id_array = [self.n_id, self.m_id, self.h_id, self.w_id]
        self.id_string = "n%d_m%d_h%d_w%d"%(self.n_id, self.m_id, self.h_id, self.w_id)
        self.tile_rect = None
        self.padded_tile_rect = None
        self.file_params = file_params
        self.Tn = Tn
        self.batch_item_start = self.n_id * Tn
        self.is_pe_input = is_pe_input
        self.channel_start = self.m_id * PEArray.NUM_COLS + stridedslice_chan_offset
        self.channel_stop = min(self.file_params.file_dims.C, self.channel_start + PEArray.NUM_COLS)
        self.channel_count = self.channel_stop - self.channel_start
        self.start_at_mid_part = (self.m_id % 2) == 1
        self.stridedslice_chan_offset = stridedslice_chan_offset
        self.n_start = self.n_id * self.Tn
        self.n_stop  = min(self.file_params.file_dims.N, self.n_start + self.Tn)
        #FIXME
        self.mepoolspec = None

    def copy(self):
        tile_id = (self.n_id, self.m_id, self.h_id, self.w_id, self.n, self.m, self.h, self.w)
        new_tile = Tile(tile_id, 
                        self.file_params, 
                        self.Tn, 
                        is_pe_input=self.is_pe_input, 
                        stridedslice_chan_offset=self.stridedslice_chan_offset)
        new_tile.tile_rect = self.tile_rect
        new_tile.padded_tile_rect = self.padded_tile_rect
        new_tile.mepoolspec = self.mepoolspec
        return new_tile

    def make_pewave(self, stridedslice_chan_offset=0):
        return PEWave(self, 0, 0, 0, stridedslice_chan_offset)

    def get_fmap_coord(self, tile_sz):
        x = self.w_id * tile_sz.x
        y = self.h_id * tile_sz.y
        return Coord(x, y)

    def get_ofmap_count(self):
        return self.channel_count

    def get_tile_data_from_file(self, flatten=False):
        # NCHW
        assert(self.file_params.file_dims.format_str == 'NCHW')
        tile_data = self.file_params.dram_data[self.n_start : self.n_stop,
                                          self.channel_start : self.channel_stop,
                                          self.tile_rect.lower.y : self.tile_rect.upper.y + 1,
                                          self.tile_rect.lower.x : self.tile_rect.upper.x + 1]
        if flatten:
            tile_rect_sz = self.tile_rect.dim2d.get_tot_size()
            tile_data_flat = np.zeros((self.Tn * tile_rect_sz, tile_data.shape[1]))
            for z in range(self.Tn):
                for i in range(tile_data.shape[1]):
                    tile_data_flat[z * tile_rect_sz : (z+1) * tile_rect_sz, i] = tile_data[z, i, :, :].flatten()
            tile_data = tile_data_flat                    
        return tile_data.astype(np.float32)

    def get_subtile_data_from_file(self, subtile_rect):
        # NCHW
        assert(self.file_params.file_dims.format_str == 'NCHW')
        return self.file_params.dram_data[self.n_start : self.n_stop,
                                          self.channel_start : self.channel_stop,
                                          subtile_rect.lower.y : subtile_rect.upper.y + 1,
                                          subtile_rect.lower.x : subtile_rect.upper.x + 1]

    def set_tile_data_in_file2(self, tile_data):
        # NCHW
        assert(self.file_params.file_dims.format_str == 'NCHW')
        self.file_params.dram_data[self.n_start : self.n_stop,
                                   self.channel_start : self.channel_stop,
                                   self.tile_rect.lower.y : self.tile_rect.upper.y + 1,
                                   self.tile_rect.lower.x : self.tile_rect.upper.x + 1] = tile_data

    def set_subtile_data_in_file2(self, subtile_rect, subtile_data):
        # NCHW
        assert(self.file_params.file_dims.format_str == 'NCHW')
        self.file_params.dram_data[self.n_start : self.n_stop,
                                   self.channel_start : self.channel_stop,
                                   subtile_rect.lower.y : subtile_rect.upper.y + 1,
                                   subtile_rect.lower.x : subtile_rect.upper.x + 1] = subtile_data

    def set_padded_tile_data_in_file(self, tile_data_flatten):
        assert(self.file_params.file_dims.format_str == 'NCHW')
        for z in range(self.Tn):
            for j in range(self.channel_start, self.channel_stop):
                result_tile_tmp = (tile_data_flatten[      z * self.padded_tile_rect.dim2d.get_tot_size()
                                                     : (z+1) * self.padded_tile_rect.dim2d.get_tot_size(), j - self.channel_start])
                result_tile = result_tile_tmp.reshape((self.padded_tile_rect.dim2d.y, self.padded_tile_rect.dim2d.x))
                # NCHW
                result = self.file_params.dram_data
                result[self.n_id * self.Tn + z, 
                        j, 
                        self.tile_rect.lower.y : self.tile_rect.lower.y + self.tile_rect.dim2d.y, 
                        self.tile_rect.lower.x : self.tile_rect.lower.x + self.tile_rect.dim2d.x]\
                    = result_tile[0:self.tile_rect.dim2d.y, 0:self.tile_rect.dim2d.x]

    def set_tile_data_in_file(self, tile_data_flatten):
        assert(self.file_params.file_dims.format_str == 'NCHW')
        for z in range(self.Tn):
            for j in range(self.channel_start, self.channel_stop):
                result_tile_tmp = (tile_data_flatten[      z * self.tile_rect.dim2d.get_tot_size()
                                                     : (z+1) * self.tile_rect.dim2d.get_tot_size(), j - self.channel_start])
                result_tile = result_tile_tmp.reshape((self.tile_rect.dim2d.y, self.tile_rect.dim2d.x))
                # NCHW
                result = self.file_params.dram_data
                result[self.n_id * self.Tn + z,
                        j,
                        self.tile_rect.lower.y : self.tile_rect.lower.y + self.tile_rect.dim2d.y,
                        self.tile_rect.lower.x : self.tile_rect.lower.x + self.tile_rect.dim2d.x]\
                    = result_tile[0:self.tile_rect.dim2d.y, 0:self.tile_rect.dim2d.x]

    def set_subtile_data_in_tile(self, tile_data_flatten, subtile_rect, subtile_data_flatten):
        assert(self.file_params.file_dims.format_str == 'NCHW')
        for z in range(self.Tn):
            for j in range(self.channel_start, self.channel_stop):
                result_subtile_tmp = (subtile_data_flatten[      z * subtile_rect.dim2d.y * subtile_rect.dim2d.x 
                                                           : (z+1) * subtile_rect.dim2d.y * subtile_rect.dim2d.x, j - self.channel_start])
                result_subtile = result_tile_tmp.reshape((subtile_rect.dim2d.y, subtile_rect.dim2d.x))
                # NCHW
                tile_data_flatten[self.n_id * self.Tn + z, 
                        j, 
                        subtile_rect.lower.y : subtile_rect.lower.y + subtile_rect.dim2d.y, 
                        subtile_rect.lower.x : subtile_rect.lower.x + subtile_rect.dim2d.x]\
                    = result_tile[0:subtile_rect.dim2d.y, 0:subtile_rect.dim2d.x]
        return tile_data_flatten                    
    
    def set_subtile_data_strided_in_tile(self, tile_data_flatten, subtile_rect, subtile_data_flatten, stride, psum_add):
        assert(self.file_params.file_dims.format_str == 'NCHW')
        for z in range(self.Tn):
            for j in range(self.channel_start, self.channel_stop):
                result_subtile_tmp = (subtile_data_flatten[      z * subtile_rect.dim2d.y * subtile_rect.dim2d.x 
                                                           : (z+1) * subtile_rect.dim2d.y * subtile_rect.dim2d.x, j - self.channel_start])
                result_subtile = result_tile_tmp.reshape((subtile_rect.dim2d.y, subtile_rect.dim2d.x))
                # NCHW
                if not psum_add:
                    tile_data_flatten[self.n_id * self.Tn + z, 
                            j, 
                            self.tile_rect.lower.y : self.tile_rect.lower.y + self.tile_rect.dim2d.y : stride.y, 
                            self.tile_rect.lower.x : self.tile_rect.lower.x + self.tile_rect.dim2d.x : stride.x]\
                        = result_tile[0:subtile_rect.dim2d.y, 0:subtile_rect.dim2d.x]
                else:
                    tile_data_flatten[self.n_id * self.Tn + z, 
                            j, 
                            self.tile_rect.lower.y : self.tile_rect.lower.y + self.tile_rect.dim2d.y : stride.y, 
                            self.tile_rect.lower.x : self.tile_rect.lower.x + self.tile_rect.dim2d.x : stride.x]\
                        += result_tile[0:subtile_rect.dim2d.y, 0:subtile_rect.dim2d.x]
        return tile_data_flatten                    
    
    def snap_rect_to_stride_grid(self, origin, stride):
        self.lower.snap_up_to_nearest_grid(origin, stride)
        self.upper.snap_down_to_nearest_grid(origin, stride)

"""List of K-Nodes that are fused (pass data through PSUM buffers)
"""
class FusedOp(list):
    """RegExs to determine whether next node is fusable or not
    """
    act_ops_regex = "Relu|Softplus|Sigmoid|Tanh|^Exp$|Identity|Lrelu|Prelu"
    bias_ops_regex = "BiasAdd"
    pool_ops_regex = ".*Pool|Add|Multiply|ResAdd|Maximum|Minimum"
    identity_ops_regex = "Reshape|Squeeze|ExpandDims"
    next_is_fusable_regex = bias_ops_regex + "|" + act_ops_regex + "|" + pool_ops_regex # + "|" + identity_ops_regex
    next_is_fusable = {
            'Conv'     : next_is_fusable_regex,
            'ConvTranspose': next_is_fusable_regex,
            'MatMul'   : next_is_fusable_regex,
            'BiasAdd'  : next_is_fusable_regex,
            'Add'      : next_is_fusable_regex,
            'ResAdd'   : next_is_fusable_regex,
            'Multiply' : next_is_fusable_regex,
            'Relu'     : next_is_fusable_regex,
            'Softplus' : next_is_fusable_regex,
            'Sigmoid'  : next_is_fusable_regex,
            'Tanh'     : next_is_fusable_regex,
            'Lrelu'    : next_is_fusable_regex,
            'Prelu'    : next_is_fusable_regex,
            #'Reshape'  : next_is_fusable_regex,
            #'Squeeze'  : next_is_fusable_regex,
            #'ExpandDims': next_is_fusable_regex,
            }

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
        self.no_writer_for_save = False
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
        #9-10-2018
        self.concat_dram_weights_waveops = []
        self.concat_dram_weights_file_params = dict()

    # Add operation to list of fused operations.
    # Returns True if successful; False if cannot add (i.e. Pool cannot be fused)
    def set_live_mapped_file_params(self, live_mapped_file_params):
        self.live_mapped_file_params = live_mapped_file_params 

    def add(self, op):
        if (self.args.debug > 2):
            print("DBG: adding layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
        # kaena-643: Alignment happens for weights/bias/first IFMAP and final OFMAP. To prevent mismatch
        # in data chunk sizes during ResNet50 execution (around ResAdd where SB region sharing is allowed)
        # the last op is not allowed to fuse
        #if op.next == [] and self.has_join:
        #    if (self.args.debug > 2):
        #        print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"], " to prevent alignment problems around ResAdd where region sharing happens.")
        #    return False
        if (op.data['layer_type'] == 'AvgPool' or op.data['layer_type'] == 'MaxPool'):
            op.populate_pooling_params()
            # If not first op, pool cannot be fused with previous op if stride != pooling window
            if (len(self) != 0
                    and (op.stride.x != op.pool_window.x 
                        or op.stride.y != op.pool_window.y
                        or op.stride.x > 1 
                        or op.stride.y > 1)):
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
        elif (op.data['layer_type'] == 'Conv' or op.data['layer_type'] == 'ConvTranspose' or op.data['layer_type'] == 'MatMul' or op.data['layer_type'] == 'Softmax2'):
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
                    #FIXME : This is only for Concat whose residue inputs are
                    # all from SB. This may not be the case for different multi-
                    # input operations. In that case, it may need to be computed
                    # differently.
                    self.join_op.residue_index = 0
                    #raise RuntimeError("Please implement unfused join, where both inputs need to be sourced from SB")
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
        if (self.has_pool and op.pool_window.y > 1 and self.has_conv):
            self.conv_op.recompute_conv_params()
        self.append(op)
        op.fused_op = self
        return True            

    def show(self):
        print("DBG: fused_ops collected: (ofmap_is_for_join %d, partial_batch_pre_pairup %d, partial_batch_pairup %d, residue_in_scratch %d)"\
                %(self.ofmap_is_for_join, self.partial_batch_pre_pairup, self.partial_batch_pairup, self.residue_in_scratch))
        for i in self:
            print("    ", i.data["layer_type"],":",i.data["layer_name"], "(end op)" if i.next == [] else "")

    # We allocate space in SB for files generated during execute
    def map_files_gen_during_exec(self, tpb, file_params):
        file_start_addr = tpb.statebuffer.next_nonbias_file_start
        file_sz = file_params.tot_partition_usage_sz_padded
        bias_region_sz = tpb.statebuffer.batcher.sb_bias_sz[0]
        def check_overlap (file_param, start_addr):
            print ("check_overlap::file_param.file_name = %s"\
                   %file_param.file_name)
            single_ifmap_start = file_param.mapped_params.start_addr
            single_ifmap_sz = file_param.mapped_params.region_sz
            next_nonbias_file_start =\
                    tpb.statebuffer.file_mapper.adjust0_if_overlap(
                        region0_start    = start_addr, 
                        region0_sz       = file_sz, 
                        region1_start    = single_ifmap_start, 
                        region1_sz       = single_ifmap_sz,
                        min_region_start = bias_region_sz
                    )
            return next_nonbias_file_start
        if file_start_addr + file_sz >= tpb.statebuffer.SB_PARTITION_SZ:
            file_start_addr = bias_region_sz
        # Check overlap with IFMAP
        for i in self.first_op.ifmaps_file_params_concat:
            file_start_addr = check_overlap(i, file_start_addr)
        # Check overlap with OFMAP
        ofmap_file_params = self.last_op.ofmaps_file_params
        assert(ofmap_file_params.mapped_params != None)
        file_start_addr = check_overlap(ofmap_file_params, file_start_addr)
        file_start_addr =\
                self.adjust_if_overlap_with_live_fmaps(
                    tpb
                    , file_start_addr 
                    , file_sz
                    , bias_region_sz
                    , self.live_mapped_file_params)
        tpb.statebuffer.file_mapper.map_file(\
                                             file_params\
                                             , file_start_addr\
                                             , wrap_around=False\
                                             , region_sz=file_sz\
                                            )
        tpb.statebuffer.next_nonbias_file_start = file_start_addr\
                + file_params.tot_partition_usage_sz_padded
        return

    def adjust_if_overlap_with_concat_ifmaps(
        self, tpb, st_addr, region_sz, bias_region_sz, single_ifmap_sz):
        i_st_addr = 0
        i_region_sz = 0
        for f in self.first_op.ifmaps_file_params_concat:
            self.print_SB_addr(f)
            if (f.mapped_params.start_addr > st_addr):
                i_st_addr = f.mapped_params.start_addr
                i_region_sz = f.mapped_params.region_sz
        o_st_addr = tpb.statebuffer.file_mapper.adjust0_if_overlap(
            region0_start    = st_addr,
            region0_sz       = region_sz, 
            region1_start    = i_st_addr,
            region1_sz       = min(single_ifmap_sz, i_region_sz),
            min_region_start = bias_region_sz
        )
        return o_st_addr

    def adjust_if_overlap_with_live_fmaps(self
                                          , tpb
                                          , st_addr
                                          , region_sz
                                          , bias_region_sz
                                          , live_mapped_file_params
                                         ):
        st = st_addr
        for c in live_mapped_file_params:
            c_mp = c.mapped_params
            fmapper = tpb.statebuffer.file_mapper
            r1_sz = c_mp.end_addr+c.item_sz-c_mp.start_addr
            st =\
                    fmapper.adjust0_if_overlap(
                        region0_start = st,
                        region0_sz = region_sz,
                        region1_start = c_mp.start_addr,
                        region1_sz = r1_sz,
                        min_region_start = bias_region_sz
                    )
        return st


    def print_SB_addr (self, file_params):
        print ("file_name = %s"%file_params.file_name)
        print ("\tstart_addr = %d"%file_params.mapped_params.start_addr)
        print ("\tend_addr = %d"%file_params.mapped_params.end_addr)

    def map_files(self, tpb, batch_item, last_concat_ofmap_file_params
                 , live_mapped_file_params):
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
            bias_file_sz = bias_file_params.tot_partition_usage_sz_padded
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
            weights_file_sz = weights_file_params.tot_partition_usage_sz_padded
        weights_region_start_addr  = 0
        weights_region_sz = weights_file_sz
        # ifmap file/region defaults            
        assert(self.first_op is not None)
        ifmaps_file_params = self.first_op.ifmaps_file_params
        single_ifmap_start = 0
        single_ifmap_sz = ifmaps_file_params.batch_item_partition_usage_sz_padded
        ifmaps_region_start_addr  = 0
        ifmaps_region_sz = single_ifmap_sz
        # ofmap file/region defaults            
        assert(self.last_op is not None)
        ofmaps_file_params = self.last_op.ofmaps_file_params
        single_ofmap_start = 0
        single_ofmap_sz = ofmaps_file_params.batch_item_partition_usage_sz_padded
        ofmaps_region_start_addr  = 0
        ofmaps_region_sz = single_ofmap_sz

        # Bias region:
        #   - keep in contiguous region to help DMA perf: an optimizer (TBD) will need to coalesce the bias files
        #   - share bias mapping for all type of nets
        #   - make sure to keep bias region the same in the batch map (BatchSBDataMap)
        if self.has_biasadd:
            if bias_file_params.mapped_params is None:
                map_file(bias_file_params, bias_file_start_addr, wrap_around=False, region_sz=bias_file_sz)
                tpb.statebuffer.next_bias_file_start = align_addr_8B(bias_file_start_addr + bias_file_sz)
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
                    ifmaps_region_sz = self.current_batch_count * ifmaps_file_params.batch_item_partition_usage_sz_padded
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
            single_ifmap_start = ifmaps_region_start_addr + (batch_item % self.current_batch_count) * ifmaps_file_params.batch_item_partition_usage_sz_padded
            single_ifmap_sz = ifmaps_file_params.batch_item_partition_usage_sz_padded

            # Join for partial-batch region
            ofmap_batch_count = self.current_batch_count
            # "pairup" is the region or boundary where OFMAP shrinks by 1/4 and partial-batch count doubles.
            if self.partial_batch_pairup:
                ofmap_batch_count = self.next_batch_count
                sb_size_set_index = tpb.statebuffer.batcher.sb_size_set_index[(ofmap_batch_count, False)]
            if ((self.last_op.is_fork or self.ofmap_is_for_join) != self.residue_in_scratch):
                # special case for stage after MaxPool: use scratch space for OFMAP instead of residue space
                #ofmaps_region_sz = ofmap_batch_count * ofmaps_file_params.batch_item_partition_usage_sz_rounded
                ofmaps_region_sz = ofmap_batch_count * ofmaps_file_params.batch_item_partition_usage_sz_padded
                ofmaps_region_start_addr =   tpb.statebuffer.batcher.sb_bias_sz[sb_size_set_index] \
                                           + tpb.statebuffer.batcher.sb_partialbatch_start[ofmap_batch_count]
            # Scratch (OFMAP)
            else:
                if self.first_op.is_nop:
                    ofmaps_region_sz = ifmaps_region_sz
                    ofmaps_region_start_addr = ifmaps_region_start_addr \
                                                + self.first_op.slice_offset.x * self.first_op.item_sz \
                                                + self.first_op.slice_offset.y * self.first_op.item_sz * self.first_op.W 
                else:    
                    ofmaps_region_sz = tpb.statebuffer.batcher.sb_scratch_sz[sb_size_set_index]
                    ofmaps_region_start_addr = tpb.statebuffer.SB_PARTITION_SZ - ofmaps_region_sz

            # If OFMAP region overlaps IFMAP region, and numober of channels > 64 or stride/filter-size > 1, offset it (to lower address) by OFMAP * Tn 
            # (stride is only relevant to Conv/Pool, and filter-size is only relevant to Conv)
            if ofmaps_region_start_addr >= ifmaps_region_start_addr \
                    and ofmaps_region_start_addr < ifmaps_region_start_addr + ifmaps_region_sz \
                    and not self.first_op.is_nop:
                if (ofmaps_file_params.file_dims.C > 64) \
                    or (self.conv_op is not None and (self.conv_op.stride.x > 1 or self.conv_op.S > 1)) \
                    or (self.has_pool and self.pool_op.stride.x > 1):
                    if (ofmaps_file_params.batch_item_partition_usage_sz_padded <= ifmaps_file_params.batch_item_partition_usage_sz_padded):
                        ofmaps_region_start_addr = ifmaps_region_start_addr - ofmaps_file_params.batch_item_partition_usage_sz_padded * self.last_op.Tn                               
                    else:    
                        ofmaps_region_start_addr = ifmaps_region_start_addr - ofmaps_file_params.tot_partition_usage_sz_padded
                # Allow modifying in place for IFMAPs which overlap the same region as OFMAPs
                if not self.first_op.is_input:
                    ifmaps_file_params.mapped_params.modify_in_place = True

            # Map the file to region and obtain adjusted region size
            map_file(ofmaps_file_params, ofmaps_region_start_addr, wrap_around=True, region_sz=ofmaps_region_sz)
            ofmaps_region_sz = ofmaps_file_params.mapped_params.region_sz
            # should be the same even if file was already mapped
            assert(ofmaps_region_start_addr == ofmaps_file_params.mapped_params.start_addr)

            # Individual OFMAP info
            single_ofmap_start = ofmaps_region_start_addr + (batch_item % ofmap_batch_count) * ofmaps_file_params.batch_item_partition_usage_sz_padded 
            single_ofmap_sz = ofmaps_file_params.batch_item_partition_usage_sz_padded

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
                weights_file_sz = weights_file_params.tot_partition_usage_sz_padded
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
                ifmaps_region_sz = ifmaps_file_params.tot_partition_usage_sz_padded
                live_mapped_file_params.append(ifmaps_file_params)
                if ifmaps_region_sz > (tpb.statebuffer.SB_PARTITION_SZ/4):
                    if self.first_op.data["layer_type"] == "AvgPool" or self.first_op.data["layer_type"] == "Pool":
                        if self.first_op.pool_window.x > 1:
                            raise RuntimeError("Cannot support yet pooling with window size > 1 if IFMAP size (%d) takes more than quarter of SB partition. This requires wrapping around the region, causing discontinuity at the end of region. Feature required: breaking pool at the discontinuity."%(ifmaps_region_sz))
                    # cap region size to be 4 chunks
                    ifmaps_region_sz = 4 * ifmaps_file_params.chunk_sz_padded * ifmaps_file_params.fmap_channels_folds
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

            # check residue and map it if it's not mapped yet (case with two inputs to a join)
            # (if it's mapped, it's likely shared SB space with OFMAP for ResAdd/Multiply)
            if self.has_join and not self.join_op.is_concat:
                residue_file_params = self.join_op.prev[self.join_op.residue_index].ofmaps_file_params
                if residue_file_params.mapped_params is None:
                    residue_region_sz = residue_file_params.tot_partition_usage_sz_padded
                    if start_addr + residue_region_sz >= tpb.statebuffer.SB_PARTITION_SZ:                    
                        start_addr = bias_region_sz
                    live_mapped_file_params.append(residue_file_params)
                    if residue_region_sz > (tpb.statebuffer.SB_PARTITION_SZ/4):
                        # cap region size to be 4 chunks
                        residue_region_sz = 4 * residue_file_params.chunk_sz_padded * residue_file_params.fmap_channels_folds
                    map_file(residue_file_params, start_addr, wrap_around=True, region_sz=residue_region_sz)
                    residue_region_sz = residue_file_params.mapped_params.region_sz
                    start_addr += residue_region_sz

            # Weights region, align to 8B
            if self.has_conv:
                if weights_file_params.mapped_params is None:
                    wrap_around = False
                    weights_file_start_addr = start_addr
                    # Ff weights file is too large, make intake buffer a circular buffer.
                    # Due to channel folding there's high chance of thrashing (high eviction rate) with a factor of 4,
                    # so use an odd factor say 7.
                    if weights_file_sz > (tpb.statebuffer.SB_PARTITION_SZ/2):
                        weights_file_sz = 7 * weights_file_params.chunk_sz_padded * weights_file_params.fmap_channels_folds
                        wrap_around = True
                    t = weights_file_start_addr
                    weights_file_start_addr =\
                            self.adjust_if_overlap_with_live_fmaps(
                                tpb
                                , weights_file_start_addr 
                                , weights_file_sz
                                , bias_region_sz
                                , live_mapped_file_params)
                    map_file(weights_file_params, weights_file_start_addr, wrap_around=wrap_around, region_sz=weights_file_sz)
                    live_mapped_file_params.append(weights_file_params)
                    # obtain the adjusted region size
                    weights_region_sz = weights_file_params.mapped_params.region_sz
                    start_addr = weights_file_start_addr + weights_region_sz
                else:                    
                    # also in case that file is already mapped, keep the mapped values
                    weights_file_start_addr = weights_file_params.mapped_params.start_addr
                weights_region_start_addr = weights_file_start_addr
            # OFMAPs region
            if ofmaps_file_params.mapped_params is None:
                if self.first_op.is_nop:
                    if self.first_op.ifmaps_file_params.file_dims.format_str == 'HNWC':
                        assert(self.first_op.slice_offset.x == 0), "Unstacking HNWC: only slicing in H dimension allowed"
                        ofmaps_region_start_addr = ifmaps_region_start_addr \
                                                    + self.first_op.slice_offset.y \
                                                        * self.first_op.ofmaps_file_params.tot_partition_usage_sz_padded
                    else:                                                    
                        ofmaps_region_start_addr = ifmaps_region_start_addr \
                                                    + self.first_op.slice_offset.x * self.first_op.item_sz \
                                                    + self.first_op.slice_offset.y * self.first_op.item_sz * self.first_op.W 
                elif self.has_join and len(self.last_op.ofmaps_file_params.writers_of_shared_fmap) > 1 \
                        and not self.join_op.is_input:
                    residue_op = self.join_op.prev[self.join_op.residue_index]                
                    ofmaps_region_start_addr = residue_op.ofmaps_file_params.mapped_params.start_addr
                else:    
                    ofmaps_region_start_addr = start_addr
                    ofmaps_region_start_addr =\
                            self.adjust_if_overlap_with_live_fmaps(
                                tpb
                                , ofmaps_region_start_addr
                                , ofmaps_region_sz
                                , bias_region_sz
                                , live_mapped_file_params)
                map_file(ofmaps_file_params\
                         , ofmaps_region_start_addr\
                         , wrap_around=True, region_sz=ofmaps_region_sz)
                if (self.first_op.is_concat == True):
                    last_concat_ofmap_file_params.append(ofmaps_file_params)
                live_mapped_file_params.append(ofmaps_file_params)
                # obtain the adjusted region size
                ofmaps_region_sz = ofmaps_file_params.mapped_params.region_sz
                single_ofmap_start = ofmaps_region_start_addr 
                start_addr = single_ofmap_start + ofmaps_region_sz
                if single_ofmap_start == single_ifmap_start:
                    #assert(not self.first_op.is_input)
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
        assert(tpb.statebuffer.file_mapper.check_overlap(weights_file_start_addr, weights_region_sz, single_ifmap_start, min(single_ifmap_sz, ifmaps_region_sz))==False),\
               "weights_file_start_addr = %d, weights_region_sz = %d, single_ifmap_start = %d, min(single_ifmap_sz, ifmaps_region_sz) = %d"%(
                   weights_file_start_addr, weights_region_sz, single_ifmap_start, (min(single_ifmap_sz, ifmaps_region_sz)))

        if not self.first_op.is_nop:
            # check that regions are either exactly overlaping or not overlapping at all
            overlap_some_percent = tpb.statebuffer.file_mapper.check_overlap(single_ifmap_start, min(single_ifmap_sz, ifmaps_region_sz), single_ofmap_start, single_ofmap_sz)
            overlap_100_percent = tpb.statebuffer.file_mapper.check_overlap100(single_ifmap_start, min(single_ifmap_sz, ifmaps_region_sz), single_ofmap_start, single_ofmap_sz)
            assert(overlap_some_percent == overlap_100_percent)

    def mark_ifmaps_are_consumed(self, live_mapped_file_params):
        # Note that each of ifmaps of current knode is ofmap of its predecessor.
        # That's why ofmap is retrieved from the predecessor knode.
        if (self.has_conv):
            weight = self.conv_op.weights_file_params
            weight.mapped_params.mark_consumed(self.first_op)
            if (weight.mapped_params.is_consumed_by_all_readers() == True):
                live_mapped_file_params.remove(weight)
        if (self.first_op.is_concat):
            for w in self.first_op.weights_file_params_concat:
                w.mapped_params.mark_consumed(self.first_op)
                if (w.mapped_params.is_consumed_by_all_readers() == True):
                    live_mapped_file_params.remove(w)
        for pred in self.first_op.prev:
            ofmap = pred.ofmaps_file_params
            if ofmap.mapped_params is not None:
                ofmap.mapped_params.mark_consumed(self.first_op)
            if (ofmap in live_mapped_file_params and\
                ofmap.mapped_params.is_consumed_by_all_readers() == True):
                live_mapped_file_params.remove(ofmap)
        return

    def execute(self, tpb, batch_item):
        assert (batch_item >= 0)
        assert (batch_item < self.first_op.N)
        # Check conv fused op
        first_op_type = self.first_op.data['layer_type']
        if (first_op_type == "Conv" or first_op_type == "ConvTranspose" or first_op_type == "MatMul"):
            self.execute_conv_ops(tpb, batch_item)
        elif (first_op_type == "AvgPool" or first_op_type == "MaxPool" or first_op_type == "ClipByValue"):
            self.execute_unfused_pool_op(tpb, batch_item)
        elif (first_op_type == "Softmax2"):
            self.execute_softmax2(tpb, batch_item)
        elif (first_op_type == "Multiply" 
                or first_op_type == "Maximum" 
                or first_op_type == "Minimum" 
                or first_op_type == "ResAdd"): # TODO: handle the scalar 
            if (len(self.first_op.data['previous_layers']) <= 2):
                results = self.execute_unfused_pool_op(tpb, batch_item)
            else:                
                raise RuntimeError("ERROR: cannot handle more than two inputs for first operation %s, layer %s"%(first_op_type, first_op.data["layer_name"]))
        elif re.search(self.act_ops_regex, first_op_type):
            self.execute_unfused_pool_op(tpb, batch_item)
        elif (first_op_type == "BiasAdd"):
            self.execute_unfused_pool_op(tpb, batch_item)
        elif (first_op_type == "Concat"):
            self.execute_unfused_concat_op(tpb, batch_item)
        elif (first_op_type == "StridedSlice"):
            self.execute_unfused_stridedslice_op(tpb, batch_item)
        elif (first_op_type == "Transpose"):
            self.execute_unfused_transpose_op(tpb, batch_item)
        elif (re.search("Reshape|Squeeze|ExpandDims", first_op_type)):
            self.execute_unfused_reshape_op(tpb, batch_item)
        elif (first_op_type == "Slice"):
            self.execute_unfused_slice_op(tpb, batch_item)
        else:        
            raise RuntimeError("ERROR: Unrecognized first operation %s"%first_op_type)

        # Check computed results against pre-computed results and save data
        # only if there's at least one node, and first node is not Placeholder
        if len(self) > 0 and not self.first_op.is_placeholder:
            # Check last output result, unless verify_output_only=False
            if self.last_op.next == [] or not self.args.verify_output_only:
                if (not self.args.no_verify):
                    if 'ref_file' in self.last_op.data and os.path.isfile(self.last_op.data['ref_file']):
                        try:
                            expected_ofmaps = np.load(self.last_op.data['ref_file'])
                        except:
                            raise RuntimeError("Cannot load numpy file %s"%(self.last_op.data['ref_file']))
                        last_batch_item = batch_item + self.first_op.Tn
                        for i in range(batch_item, last_batch_item):
                            ifmaps = self.first_op.ifmaps_file_params.dram_data[i, :]
                            ofmaps = self.last_op.ofmaps_file_params.dram_data[i, :]
                            print("ofmap name = %s",self.last_op.ofmaps_file_params.file_name)
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
            if self.last_op.next == [] or self.args.save_layer_output and not self.no_writer_for_save:
                last_batch_item = batch_item + self.first_op.Tn
                for i in range(batch_item, last_batch_item):
                    waveops = tpb.statebuffer.file_mapper.flush_file(
                                    waveop_id   = tpb.waveop_stream.waveop_count, 
                                    waveop_list = tpb.waveop_stream, 
                                    file_params         = self.last_op.ofmaps_file_params, 
                                    batch_item          = i
                                    )
                    tpb.waveop_stream.add_outputs(waveops)
                self.last_op.ofmaps_file_params.save_file()

    # generate MatMul waveop and add it to waveop stream
    def gen_matmul_waveop(self, tpb, ifmap_pewave, ofmap_pewave, psum_add, dram_weights_waveops, repl_multiple_of_C=1, conv_transpose=False):
        batch_item = ofmap_pewave.tile.n_id * ofmap_pewave.tile.Tn
        if (self.first_op.item_sz == 2):
            in_dtype = "float16"
            out_dtype = "float32"
        elif (self.first_op.item_sz == 4):
            in_dtype = "float32"
            out_dtype = "float32"
        else:            
            raise RuntimeError("ERROR: item_sz %d not yet supported"%self.first_op.item_sz)
        # find the weights offset within atom; -1 means don't load new weights
        weights_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.first_op.weights_file_params, 0, self.first_op.weight_wave_lower_addr)
        # kaena-421: during execution for batch item other than the first one, need to check if there's any SBAtomLoad due to eviction
        # when the fused-op is reexecuted (since self.prev_weight_wave_lower_addr is preserved across batch item calls) 
        if (weights_sb_address == self.prev_weight_wave_lower_addr and dram_weights_waveops == []):
          if (self.first_op.is_concat == False):
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
        subtile_rect_dim2d = ifmap_pewave.subtile_rect.dim2d if conv_transpose else ofmap_pewave.subtile_rect.dim2d
        addr_step_y = self.first_op.W * self.first_op.stride.y * self.first_op.item_sz
        for i in range(subtile_rect_dim2d.y):
            # TODO: how to deal with partial batching here?
            address = ifmap_pewave.lower_addr[0] + i * addr_step_y
            if (address > ifmap_pewave.upper_addr[0]):
                break
            chunk_id = tpb.statebuffer.file_mapper.get_chunk_id_from_file_addr(self.first_op.ifmaps_file_params, batch_item, address)
            atom_id = tpb.statebuffer.file_mapper.get_atom_id_from_file_addr(self.first_op.ifmaps_file_params, batch_item, address)
            if self.args.abstract_mem:
                break_cond = chunk_id != current_chunk_id
            else:
                break_cond = not (atom_id == current_atom_id or atom_id == current_atom_id+1)
            if break_cond:
                break_at_y.append(i)
                break_addr.append(tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.first_op.ifmaps_file_params, batch_item, address))
                current_chunk_id = chunk_id
                current_atom_id = atom_id
                if (self.args.debug > 3): print("DBG: breaking wave at row %d addr %d"%(i, break_addr[-1]))
        matmul_waveop = []
        start_tensor_calc = not(psum_add)

        # source and destination patterns: x num elems
        fmap_x_num = subtile_rect_dim2d.x
        dst_x_num = subtile_rect_dim2d.x
        # source and destination patterns: x step
        fmap_x_step = 1 if conv_transpose else self.first_op.stride.x
        dst_x_step = self.first_op.stride.x if conv_transpose else 1
        # source and destination patterns: y step
        fmap_y_step = self.first_op.W * (1 if conv_transpose else self.first_op.stride.y)
        dst_y_step = ofmap_pewave.tile.tile_rect.dim2d.x * (self.first_op.stride.y if conv_transpose else 1)
        # source and destination patterns: z step
        fmap_z_step = self.first_op.ifmaps_file_params.batch_item_partition_usage_elems_padded if self.first_op.Tn > 1 else 1

        # replication parameters
        ifmap_replication_resolution = 0
        ifmap_replication_num_rows = 0
        ifmap_replication_shift_amnt = 0
        if self.first_op.repl_multiple_of_C > 1:
            assert(conv_transpose == False)
            # Kaena-593: ensure no bubble during IFMAP streaming (packed pattern)
            fmap_x_num = self.first_op.W // self.first_op.stride.x
            fmap_x_step = 1     # image gets split into even/odd
            fmap_y_step = self.first_op.W // self.first_op.stride.x
            dst_x_num = fmap_x_num
            dst_y_step = fmap_y_step
            ifmap_replication_resolution = self.first_op.C * self.first_op.stride.x
            ifmap_replication_num_rows = self.first_op.C * self.first_op.S
            ifmap_replication_shift_amnt = 1

        for i in range(len(break_at_y)):                
            if (i == len(break_at_y)-1):
                next_break = subtile_rect_dim2d.y
            else:
                next_break = break_at_y[i+1]
            fmap_y_num = next_break - break_at_y[i]
            dst_psum_bank_offset = break_at_y[i] * ofmap_pewave.tile.tile_rect.dim2d.x
            dst_psum_bank_offset += ofmap_pewave.subtile_psum_offset
            assert(dst_psum_bank_offset < PEArray.MAX_WAVE_SIZE)
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

            for z in range(self.first_op.Tn):                    
                lower_file_address = ifmap_pewave.lower_addr[z] + break_at_y[i] * addr_step_y
                upper_file_address = min(ifmap_pewave.lower_addr[z] + next_break * addr_step_y - self.first_op.item_sz, ifmap_pewave.upper_addr[z])
                list_of_names = tpb.statebuffer.file_mapper.get_dram_waveop_names(self.first_op.ifmaps_file_params, batch_item + z, lower_file_address, upper_file_address)
                for name in list_of_names:
                    if name not in dram_waveop_names:
                        dram_waveop_names.append(name)

            waveop_name = self.first_op.data['layer_name']+"/MatMul_"+ofmap_pewave.id_string+"__"+str(i)
            if (self.args.debug > 2): print("DBG %s: MatMul wave %s subwave %d weights_sb_address %d, fmap_sb_address %d, fmap_y_num %d"%(self.first_op.data['layer_name'], waveop_name, i, weights_sb_address, fmap_sb_address, fmap_y_num))                
            matmul_waveop.append({ 
                  'previous_waveops'        : dram_waveop_names,
                  'waveop_type'             : 'MatMul',
                  'waveop_name'             : waveop_name,
                  'layer_name'              : self.first_op.data['layer_name'],
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
                  'src_z_num'               : self.first_op.Tn,
                  'num_row_partitions'      : ifmap_pewave.ifmap_channel_count * repl_multiple_of_C,
                  'dst_is_psum'             : True,
                  'dst_psum_bank_id'        : self.first_op.psum_bank_dst,
                  'dst_psum_bank_offset'    : dst_psum_bank_offset,
                  'dst_x_step'              : dst_x_step,
                  'dst_x_num'               : dst_x_num,
                  'dst_y_step'              : dst_y_step,
                  'dst_y_num'               : fmap_y_num,
                  'dst_z_step'              : self.first_op.ofmap_full_tile_sz,
                  'dst_z_num'               : self.first_op.Tn,
                  'num_column_partitions'   : ofmap_pewave.tile.channel_count,
                  'ifmap_replication_resolution' : ifmap_replication_resolution, 
                  'ifmap_replication_num_rows' : ifmap_replication_num_rows,
                  'ifmap_replication_shift_amnt' : ifmap_replication_shift_amnt,
                })
            start_tensor_calc = False   # this is only true for the first MatMul, even when there's a break
        return matmul_waveop

    # generate Pool waveop and add it to waveop stream
    # TODO: currently, always go to SB after Pooling
    # TODO: currently, cannot process multiple batch items in one instruction
    def gen_pool_waveop(self, tpb, ifmap_tile, ofmap_tile, src_is_psum, src_psum_bank_id, start_at_mid_part, partial_batch_item):
        batch_item = ofmap_tile.tile.n_id * self.pool_op.Tn
        window_x = ifmap_tile.window.x
        window_y = ifmap_tile.window.y
        if (src_is_psum):
            src_ifmap_width = ifmap_tile.subtile_rect.dim2d.x
            if self.conv_op != None and self.conv_op.repl_multiple_of_C > 1:
                src_ifmap_width = self.conv_op.W // self.conv_op.stride.x
            src_ifmap_height = ifmap_tile.subtile_rect.dim2d.y
            src_sb_address = 0
            if (self.pool_op.item_sz == 2):
                in_dtype = "float32"
            else:    
                in_dtype = "float32"
        else:
            src_ifmap_width = self.pool_op.W
            src_ifmap_height = self.pool_op.H
            src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.pool_op.ifmaps_file_params, batch_item + partial_batch_item, ifmap_tile.lower_addr[partial_batch_item])
            in_dtype = self.out_data_type
        src_psum_bank_offset = src_ifmap_width * src_ifmap_height * partial_batch_item
        waveop_name = self.pool_op.data['layer_name'] + "/Pool_" + ofmap_tile.id_string
        #pool_frequency = self.pool_op.pool_window.x * self.pool_op.pool_window.y
        pool_frequency = window_x * window_y
        pool_scale = float(1/pool_frequency)
        dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.pool_op.ofmaps_file_params, batch_item + partial_batch_item, ofmap_tile.lower_addr[partial_batch_item])
        dst_is_psum = False
        instr = {
              'previous_waveops'        : [],   # to be added later
              'waveop_type'             : 'Pool',
              'waveop_name'             : waveop_name,
              'layer_name'              : self.pool_op.data['layer_name'],
              'tile_id_format'          : ofmap_tile.format,
              'tile_id'                 : ofmap_tile.id_array,
              'pool_func'               : self.pool_op.data['layer_type'],
              'in_dtype'                : in_dtype,
              'out_dtype'               : self.out_data_type,
              'src_is_psum'             : src_is_psum,
              'src_x_step'              : 1,
              'src_x_num'               : window_x,
              'src_y_step'              : src_ifmap_width,
              'src_y_num'               : window_y,
              'src_z_step'              : self.pool_op.stride.x,
              'src_z_num'               : ofmap_tile.subtile_rect.dim2d.x,
              'src_w_step'              : src_ifmap_width * self.pool_op.stride.y,
              'src_w_num'               : ofmap_tile.subtile_rect.dim2d.y,
              'pool_frequency'          : pool_frequency,
              'pool_scale'              : pool_scale,
              'num_partitions'          : ofmap_tile.tile.get_ofmap_count(),
              'dst_is_psum'             : dst_is_psum,
              'dst_x_step'              : 1,
              'dst_x_num'               : ofmap_tile.subtile_rect.dim2d.x,
              'dst_y_step'              : self.pool_op.E,
              'dst_y_num'               : ofmap_tile.subtile_rect.dim2d.y,
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
    def execute_matmul_waveop(self, tpb, ifmap_pewave, ofmap_pewave, inputs, weights, psum_add, repl_multiple_of_C):
        batch_item = ofmap_pewave.tile.n_id * self.conv_op.Tn
        self.conv_op.compute_ifmap_ofmap_pewave_info(ifmap_pewave, ofmap_pewave, self.conv_op.is_conv_transpose)
        pearray_packed_weights = self.conv_op.pack_wave_conv_weights(
                                        weights, 
                                        ifmap_pewave,
                                        ofmap_pewave, 
                                        repl_multiple_of_C
                                        )
        if self.conv_op.is_conv_transpose:
            pearray_packed_ifmaps = self.conv_op.pack_wave_ifmaps_deconv(
                                        ifmap_pewave,
                                        ofmap_pewave
                                        )
        else:            
            pearray_packed_ifmaps = self.conv_op.pack_wave_ifmaps(
                                        inputs, 
                                        ifmap_pewave,
                                        ofmap_pewave,
                                        repl_multiple_of_C,
                                        for_softmax=False
                                        )
            """
            length = self.conv_op.ifmap_wave_upper_addr[0] - self.conv_op.ifmap_wave_lower_addr[0] + self.conv_op.item_sz
            print("ifmap tile rect: ", ifmap_pewave.tile.tile_rect, " ifmap wave rect: ", ifmap_pewave.subtile_rect)
            print("ofmap tile rect: ", ofmap_pewave.tile.tile_rect, " ofmap wave rect: ", ofmap_pewave.subtile_rect)
            print(self.conv_op.ifmap_wave_lower_addr[0], 
                    "ifmap_wave_lower: ", self.conv_op.ifmap_wave_lower_coordx[0], self.conv_op.ifmap_wave_lower_coordy[0], 
                    "ifmap_wave_upper: ", self.conv_op.ifmap_wave_upper_coordx[0], self.conv_op.ifmap_wave_upper_coordy[0], 
                    "ofmap_wave_lower: ", self.conv_op.ofmap_wave_lower_coordx[0], self.conv_op.ofmap_wave_lower_coordy[0],
                    "ofmap_wave_upper: ", self.conv_op.ofmap_wave_upper_coordx[0], self.conv_op.ofmap_wave_upper_coordy[0],
                    "ofmap_wave_width: ", self.conv_op.ofmap_wave_width,
                    "ofmap_wave_height: ", self.conv_op.ofmap_wave_height,
                    "psum_bank_offset: ", self.conv_op.psum_bank_offset,
                    "length: ", length,
                    )
            print(ifmap_pewave.lower_addr[0], 
                    "ifmap_wave_lower: ", ifmap_pewave.subtile_rect.lower.x, ifmap_pewave.subtile_rect.lower.y, 
                    "ifmap_wave_upper: ", ifmap_pewave.subtile_rect.upper.x, ifmap_pewave.subtile_rect.upper.y, 
                    "ofmap_wave_lower: ", ofmap_pewave.subtile_rect.lower.x, ofmap_pewave.subtile_rect.lower.y,
                    "ofmap_wave_upper: ", ofmap_pewave.subtile_rect.upper.x, ofmap_pewave.subtile_rect.upper.y,
                    "ofmap_wave_width: ", ofmap_pewave.subtile_rect.dim2d.x,
                    "ofmap_wave_height: ", ofmap_pewave.subtile_rect.dim2d.y,
                    "psum_bank_offset: ", ofmap_pewave.subtile_psum_offset,
                    "length: ", ifmap_pewave.lower_to_upper_len_bytes[0],
                    )
            """
        if (ifmap_pewave.lower_addr[0] < 0 or ifmap_pewave.upper_addr[0] < 0):
            assert(ifmap_pewave.subtile_rect.is_empty)
            assert(ifmap_pewave.lower_addr[0] < 0)
            assert(ifmap_pewave.upper_addr[0] < 0)
            print("WARNING layer %s: IFMAP wave (%s) has no data, so don't create waveops for this wave"%(self[0].data['layer_name'], ofmap_pewave.id_string))
            return False
        else:
            if self.conv_op.is_conv_transpose:
                packed_ofmaps = np.matmul(pearray_packed_ifmaps.astype(np.float32), pearray_packed_weights.astype(np.float32))
                tpb.pearray.unpack_wave_ofmaps_deconv(
                        packed_ofmaps
                        , ifmap_pewave
                        , ofmap_pewave
                        , tpb.pearray.psum_buf[self.conv_op.psum_bank_dst]
                        , not psum_add
                        , self.conv_op.stride)         
            else:
                #assert(self.conv_op.psum_bank_offset == ofmap_pewave.subtile_psum_offset)
                tpb.pearray.wave_fp16_mm(pearray_packed_ifmaps, pearray_packed_weights, self.conv_op.psum_bank_dst, psum_add)
            # Generate weights waveops
            (writers, readers, dram_weights_waveops, new_reader_morsels) = tpb.statebuffer.file_mapper.read_file_data_region(
                                        tpb.waveop_stream.waveop_count,
                                        tpb.waveop_stream,
                                        self.conv_op.weights_file_params,
                                        0,  # batch_item doesn't apply for weights
                                        self.conv_op.weight_wave_lower_addr, 
                                        self.conv_op.weight_wave_upper_addr - self.conv_op.weight_wave_lower_addr + self.conv_op.item_sz,
                                        start_at_mid_part = False,
                                        end_after_mid_part = True,
                                        repl_multiple_of_C = repl_multiple_of_C)
            if (self.args.debug > 2): print("DBG %s: MatMul weight_wave_lower_addr %d weight_wave_upper_addr %d"%(self.conv_op.data['layer_name'], self.conv_op.weight_wave_lower_addr, self.conv_op.weight_wave_upper_addr))                
            for i in dram_weights_waveops: tpb.waveop_stream.append_check(i)

            dram_ifmaps_waveops = []
            list_of_accessors = []
            for z in range(self.conv_op.Tn):
                # TODO: move the following into gen_matmul_waveop to handle breaking wave into two
                (writers, readers, waveops, reader_morsels) = tpb.statebuffer.file_mapper.read_file_data_region(
                                            tpb.waveop_stream.waveop_count,
                                            tpb.waveop_stream,
                                            ifmap_pewave.tile.file_params,
                                            batch_item + z,
                                            ifmap_pewave.lower_addr[z],
                                            ifmap_pewave.lower_to_upper_len_bytes[z],
                                            start_at_mid_part = False,
                                            end_after_mid_part = True,
                                            repl_multiple_of_C = self.conv_op.repl_multiple_of_C
                                            )

                if self.args.no_inter_layer_load:
                    if (not self.conv_op.is_input 
                            and self.conv_op.ifmaps_file_params.unstack_from_file is None
                            and len(waveops) > 0):
                        raise RuntimeError("There are DRAM loads when option no_inter_layer_load is set")
                if (self.args.debug > 2): print("DBG %s: MatMul ifmaps_wave_lower_addr %d ifmap_pewave rect "%(self.conv_op.data['layer_name'], ifmap_pewave.lower_addr[z]), ifmap_pewave.subtile_rect)
                dram_ifmaps_waveops += waveops
                list_of_accessors += writers    # only include writers for RAW dependency
                new_reader_morsels += reader_morsels
            
            # consider all Tn batch items together to avoid redundant edges
            # TODO: roll this code into read_file_data_region
            prev_waveops = self.extract_predecessors(
                    list_of_accessors   = list_of_accessors, 
                    waveop_list         = tpb.waveop_stream,
                    dram_waveops        = dram_ifmaps_waveops,
                    relax_dependencies  = self.args.relax_dependencies,
                    enable_cleanup      = self.args.enable_cleanup)

            for i in dram_ifmaps_waveops: 
                tpb.waveop_stream.append_check(i)
            matmul_waveop = self.gen_matmul_waveop(tpb, ifmap_pewave, ofmap_pewave, psum_add, dram_weights_waveops, repl_multiple_of_C, self.conv_op.is_conv_transpose)
            for i in range(len(matmul_waveop)):
                tpb.waveop_stream.add_linked(matmul_waveop[i], [], self.conv_op.psum_bank_dst, new_reader_morsels)
                if i==0:    # only need to satisfy the first in group of matmul waveops
                    self.attach_predecessors(matmul_waveop[i], prev_waveops)
            return True
        
    # execute remaining fused ops
    def execute_postconv_tile_ops (self, tpb, ifmap_tile, ofmap_tile, psum_bank_src):
        psum_temp = tpb.pearray.extract_psum(psum_bank_src, 0, self.first_op.ofmap_full_tile_sz * ofmap_tile.Tn)
        op_list_iter = iter(range(1, len(self)))
        op_list = self
        batch_item = ofmap_tile.n_id * ofmap_tile.Tn
        prev_waveops = []
        for i in op_list_iter:
            layer_type = self[i].data['layer_type'] 
            if (re.search(self.act_ops_regex, layer_type)):
                dram_bias_waveops = []
                list_of_accessors = []
                if 1: # (ofmap_tile.m_id%2 == 0):
                    (writers, readers, dram_bias_waveops, _) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                    tpb.waveop_stream.waveop_count,
                                                    tpb.waveop_stream,
                                                    tpb.statebuffer.zero_bias_file_params,
                                                    0,  # batch_item is not needed for bias
                                                    0,
                                                    tpb.statebuffer.zero_bias_file_params.item_sz)
                    list_of_accessors += writers + readers   # include readers for implicit dep on prev bias load
                #print("DBG:", list_of_accessors)
                # TODO: roll this code into read_file_data_region
                prev_waveops = self.extract_predecessors(
                        list_of_accessors   = list_of_accessors, 
                        waveop_list = tpb.waveop_stream,
                        dram_waveops        = [],
                        relax_dependencies  = self.args.relax_dependencies,
                        enable_cleanup      = self.args.enable_cleanup)

                alpha = 1.0
                if layer_type == 'Lrelu':
                    alpha = self[i].data['mul_scalar']
                psum_temp = tpb.activate.act(op_list[i].data['layer_type'], psum_temp, alpha)
                psum_bank_dst = psum_bank_src
                dst_is_psum = False
                if (i != len(op_list)-1):
                    dst_is_psum = True
                    tpb.pearray.write_psum(psum_bank_dst, 0, psum_temp.shape[0], psum_temp)
                waveop = self.gen_act_waveop_inline(tpb, None, op_list[i], self.first_op, ifmap_tile, ofmap_tile, 
                                          True, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, 0)
                psum_bank_src = psum_bank_dst
                self.attach_predecessors(waveop, prev_waveops)
                #tpb.waveop_stream.last_psum_waveop[psum_bank_dst]['previous_waveops'] += prev_waveops
            elif (layer_type == 'BiasAdd'):
                # load bias values
                bias = []
                bias_temp = self.biasadd_op.bias_file_params.dram_data
                bias = bias_temp.flatten()
                bias_chan_start = (ofmap_tile.m_id//2) * PEArray.NUM_ROWS
                bias_chan_mid_part = (ofmap_tile.m_id%2) == 1
                bias_chan_end = min(bias_chan_start + PEArray.NUM_ROWS, ofmap_tile.file_params.file_dims.C)
                bias_extracted = np.zeros(PEArray.NUM_ROWS)
                bias_extracted[0 : bias_chan_end - bias_chan_start] = bias[bias_chan_start : bias_chan_end]
                bias_addr = bias_chan_start * op_list[i].item_sz
                dram_bias_waveops = []
                list_of_accessors = []
                if 1: #(ofmap_tile.m_id%2 == 0):
                    (writers, readers, dram_bias_waveops, _) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                    tpb.waveop_stream.waveop_count,
                                                    tpb.waveop_stream,
                                                    self.biasadd_op.bias_file_params,
                                                    0,  # batch_item is not needed for bias
                                                    bias_addr,
                                                    self.biasadd_op.item_sz)
                    list_of_accessors += writers + readers   # include readers for implicit dep on prev bias load

                prev_waveops = self.extract_predecessors(
                        list_of_accessors   = list_of_accessors, 
                        waveop_list = tpb.waveop_stream,
                        dram_waveops        = [],
                        relax_dependencies  = self.args.relax_dependencies,
                        enable_cleanup      = self.args.enable_cleanup)
                #x = DBG_DUMP_PSUM_COL("PSUM col0 before BiasAdd (FP32): ", psum_temp, 0)
                psum_temp = tpb.activate.biasadd(psum_temp, bias_extracted[bias_chan_mid_part*PEArray.NUM_COLS : (bias_chan_mid_part+1)*PEArray.NUM_COLS])
                #y = DBG_DUMP_PSUM_COL("PSUM col0 after BiasAdd: ", psum_temp, 0)
                psum_bank_dst = psum_bank_src 
                dst_is_psum = False
                if (i+1 < len(op_list) and re.search(self.act_ops_regex, op_list[i+1].data['layer_type'])):
                    act_type = op_list[i+1].data['layer_type']
                    alpha = 1.0
                    if act_type == 'Lrelu':
                        alpha = op_list[i+1].data['mul_scalar']
                    psum_temp = tpb.activate.act(act_type, psum_temp, alpha)
                    if (i+1 != len(op_list)-1):
                        dst_is_psum = True
                        tpb.pearray.write_psum(psum_bank_dst, 0, self.first_op.ofmap_full_tile_sz * ofmap_tile.Tn, psum_temp)
                    waveop = self.gen_act_waveop_inline(tpb, op_list[i], op_list[i+1], self.first_op, ifmap_tile, ofmap_tile, 
                                              True, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_addr)
                    psum_bank_src = psum_bank_dst
                    next(op_list_iter)
                else:                                    
                    if (i != len(op_list)-1):
                        dst_is_psum = True
                        tpb.pearray.write_psum(psum_bank_dst, 0, self.first_op.ofmap_full_tile_sz * ofmap_tile.Tn, psum_temp)
                    waveop = self.gen_act_waveop_inline(tpb, op_list[i], None, self.first_op, ifmap_tile, ofmap_tile, 
                                              True, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_addr)
                    psum_bank_src = psum_bank_dst
                self.attach_predecessors(waveop, prev_waveops)
                #tpb.waveop_stream.last_psum_waveop[psum_bank_dst]['previous_waveops'] += prev_waveops
            elif (self[i].is_join):
                residue_file_params = self[i].prev[self[i].residue_index].ofmaps_file_params
                if residue_file_params.mapped_params.start_addr == ofmap_tile.file_params.mapped_params.start_addr:
                    if not self[i].is_input:
                        residue_file_params.mapped_params.modify_in_place = True
                dram_resadd_waveops = []
                list_of_accessors = []
                if 1: # (ofmap_tile.m_id%2 == 0):
                    for z in range(op_list.first_op.Tn):
                        (writers, readers, waveops, _) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                    tpb.waveop_stream.waveop_count,
                                                    tpb.waveop_stream,
                                                    residue_file_params,
                                                    batch_item + z,
                                                    ofmap_tile.lower_addr[z], 
                                                    ofmap_tile.upper_addr[z] - ofmap_tile.lower_addr[z] + self.first_op.item_sz)
                        #if self.args.no_inter_layer_load:
                        #    if (not self.first_op.is_input and len(waveops) > 0):
                        #        raise RuntimeError("There are DRAM loads when option no_inter_layer_load is set")
                        if (self.args.debug > 2): print("DBG %s: ResAdd/Mult ofmaps_tile_lower_addr %d ofmap_tile_upper_addr %d"%(self.first_op.data['layer_name'], ofmap_tile.lower_addr[z], ofmap_tile.upper_addr[z]))                
                        dram_resadd_waveops += waveops
                        list_of_accessors += writers    # only include writers for RAW dependency

                # consider all Tn batch items together to avoid redundant edges
                # TODO: roll this code into read_file_data_region
                prev_waveops = self.extract_predecessors(
                        list_of_accessors   = list_of_accessors, 
                        waveop_list = tpb.waveop_stream,
                        dram_waveops        = [],
                        relax_dependencies  = self.args.relax_dependencies,
                        enable_cleanup      = self.args.enable_cleanup)

                # Do the actual math
                residue_tile = ofmap_tile.copy()
                residue_tile.file_params = residue_file_params
                residue_ifmaps = residue_tile.get_tile_data_from_file(flatten=True)
                num_cols = residue_ifmaps.shape[1]
                num_elems = residue_ifmaps.shape[0]
                #x1 = DBG_DUMP_PSUM_COL("PSUM col0 before ResAdd (FP32): ", psum_temp, 0)
                #x2 = DBG_DUMP_PSUM_COL("Residue col0 before ResAdd (FP32): ", residue_ifmaps, 0)
                if (layer_type == 'ResAdd'):
                    psum_temp[0:num_elems, 0:num_cols] = tpb.pool.resadd(psum_temp[0:num_elems, 0:num_cols], residue_ifmaps)
                elif (layer_type == 'Multiply'):    
                    psum_temp[0:num_elems, 0:num_cols] = tpb.pool.multiply(psum_temp[0:num_elems, 0:num_cols], residue_ifmaps)
                elif (layer_type == 'Maximum'):    
                    psum_temp[0:num_elems, 0:num_cols] = tpb.pool.maximum(psum_temp[0:num_elems, 0:num_cols], residue_ifmaps)
                elif (layer_type == 'Minimum'):    
                    psum_temp[0:num_elems, 0:num_cols] = tpb.pool.minimum(psum_temp[0:num_elems, 0:num_cols], residue_ifmaps)
                else:
                    raise RuntimeError("ERROR: don't know how to handle vector op %s for layer %s"%(layer_type, self[i].data["layer_name"]))
                #y1 = DBG_DUMP_PSUM_COL("PSUM col0 after RessAdd (FP32): ", psum_temp, 0)
                dst_is_psum = False
                psum_bank_dst = -1
                if (i != len(op_list)-1):
                    dst_is_psum = True
                    psum_bank_dst = psum_bank_src
                    tpb.pearray.write_psum(psum_bank_dst, 0, self.first_op.ofmap_full_tile_sz * ofmap_tile.Tn, psum_temp)
                residue_file_addr = residue_file_params.ravel_nchw(
                        ofmap_tile.n_id * ofmap_tile.Tn,
                        ofmap_tile.m_id // 2 * PEArray.NUM_ROWS,
                        ofmap_tile.tile_rect.lower.y,
                        ofmap_tile.tile_rect.lower.x)
                residue_sb_addr =  tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(residue_file_params, batch_item, residue_file_addr)
                waveop = self.gen_join_waveop_inline(tpb, op_list[i], 
                        ifmap_tile, 
                        ofmap_tile, 
                        True,
                        psum_bank_src, 
                        dst_is_psum, 
                        psum_bank_dst, 
                        dram_resadd_waveops, 
                        self[i].residue_index,
                        residue_sb_addr)
                self.attach_predecessors(waveop, prev_waveops)
                psum_bank_src = psum_bank_dst
            elif re.search("Reshape|Squeeze|ExpandDims", layer_type):
                pass
            elif ((layer_type == 'AvgPool') or (layer_type == 'MaxPool')):
                tilex = self[i].ofmap_full_tilex_sz * self[i].stride.x
                tiley = self[i].ofmap_full_tiley_sz * self[i].stride.y
                #x = DBG_DUMP_PSUM_COL("PSUM before pool: ", psum_temp, 0)
                psum_temp  = tpb.pool.pool(
                                        type             = layer_type,
                                        in_array         = psum_temp, 
                                        stride           = self[i].stride, 
                                        pool_window      = self[i].pool_window, 
                                        Tn               = self[i].Tn, 
                                        ifmap_tilex_sz   = tilex,
                                        ifmap_tiley_sz   = tiley,
                                        ofmap_tilex_sz   = self[i].ofmap_full_tilex_sz,
                                        ofmap_tiley_sz   = self[i].ofmap_full_tiley_sz)
                #x = DBG_DUMP_PSUM_COL("PSUM after pool: ", psum_temp, 0)
                ifmap_subtile = PoolSubtile(ifmap_tile, ifmap_tile.tile_rect, self[i].pool_window)
                ofmap_subtile = PoolSubtile(ofmap_tile, ofmap_tile.tile_rect, None)
                waveop = self.gen_fused_pool_waveop_inline(tpb, ifmap_subtile, ofmap_subtile, psum_bank_src, (ofmap_tile.m_id%2) == 1)
                self.attach_predecessors(waveop, prev_waveops)
            else:
                raise RuntimeError("ERROR: %s is currently not yet implemented"%layer_type)

            ofmap_tile.set_tile_data_in_file(psum_temp)
            ofmap_pewave = PEWave(ofmap_tile, 0, 0, 0)
            if not re.search("Reshape|Squeeze|ExpandDims", layer_type):
                self.mark_producers_for_subtile_region(
                        tpb          = tpb, 
                        fmap_subtile = ofmap_pewave,
                        waveop       = waveop)
        return psum_temp

    # generate activation instruction and add it to instruction stream
    def gen_recip_waveop_inline(self, tpb, op, psum_bank_src, dst_is_psum, psum_bank_dst):
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
        tpb.waveop_stream.add_linked(instr, [], -1)
        return instr

    # generate scaleadd instruction and add it to instruction stream
    def gen_scaleadd_or_clip_waveop_inline(self, tpb, op, ifmap_tile, ofmap_tile, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_waveops, scale_or_min_val, add_or_max_val):
        batch_item = ofmap_tile.n_id * op.Tn
        layer_name = op.data["layer_name"]
        # TODO: update in_dtype when src_is_psum is added
        in_dtype = "float32"
        out_dtype = "float32"
        # TODO: refactor to some class to determine in_dtype and out_dtype
        if (op.item_sz == 2 and not src_is_psum):
            in_dtype = "float16"
        elif (op.item_sz == 1 and not src_is_psum):
            raise RuntimeError("ERROR: item_sz %d not yet supported"%op.item_sz)
        if (op.item_sz == 2 and not dst_is_psum):
            out_dtype = "float16"
        elif (op.item_sz == 1 and not dst_is_psum):
            raise RuntimeError("ERROR: item_sz %d not yet supported"%op.item_sz)
        if (src_is_psum):
            src_sb_address = 0
        else:
            src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ifmaps_file_params, batch_item, ifmap_tile.lower_addr[0])
        dst_x_num = op.ofmap_full_tilex_sz
        dst_y_step = op.E
        dst_y_num = op.ofmap_full_tiley_sz
        dst_z_step = op.ofmaps_file_params.batch_item_partition_usage_elems_padded if op.Tn > 1 else 1
        dst_z_num = op.Tn  # Need CNHW data format
        num_partitions = op.ofmap_count
        if dst_is_psum:
            dst_sb_address = 0
        else:            
            dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, batch_item, ofmap_tile.lower_addr[0])
        waveop_name = layer_name + "/" + op.data['layer_type'] + "_" + ofmap_tile.id_string            
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : op.data['layer_type'],
              'waveop_name'             : waveop_name,
              'layer_name'              : layer_name,
              'tile_id_format'          : ofmap_tile.format,
              'tile_id'                 : ofmap_tile.id_array,
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
            }
        if op.data['layer_type'] == "ClipByValue":
            instr['min_val'] = scale_or_min_val
            instr['max_val'] = add_or_max_val
        else:            
            instr['waveop_type'] = "ScaleAdd"
            instr['scale'] = scale_or_min_val
            instr['add'] = add_or_max_val
        if src_is_psum:
            instr['src_psum_bank_id'] = psum_bank_src
            instr['src_psum_bank_offset'] = 0 
        else:                
            instr['src_sb_address'] = src_sb_address
            instr['src_start_at_mid_part'] = ofmap_tile.m_id%2 == 1
        if dst_is_psum:
            instr['dst_psum_bank_id'] = psum_bank_dst
            instr['dst_psum_bank_offset'] = 0
        else:                
            instr['dst_sb_address'] = dst_sb_address
            instr['dst_start_at_mid_part'] = ofmap_tile.m_id%2 == 1
        tpb.waveop_stream.add_linked(instr, dram_waveops, psum_bank_src if src_is_psum else -1)
        return instr

    # generate activation instruction and add it to instruction stream
    def gen_act_waveop_inline(self, tpb, biasadd_op, act_op, conv_op, ifmap_tile, ofmap_tile, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_start, new_reader_morsels=[]):
        layer_name = ""
        # kaena-452: load zeros into start of SB and use it for Activation instruction when there's no BiasAdd
        bias_add_en = True
        bias_sb_address = 0
        # TODO: update in_dtype when src_is_psum is added
        in_dtype = "float32"
        out_dtype = "float32"
        act_or_biasadd_op = None
        alpha = 1.0
        if (biasadd_op != None):
            act_or_biasadd_op = biasadd_op
            bias_add_en = True
            bias_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(biasadd_op.bias_file_params, 0, bias_start)
            layer_name = biasadd_op.data['layer_name']
            if (biasadd_op.item_sz == 2 and not src_is_psum):
                in_dtype = "float16"
            elif (biasadd_op.item_sz == 1 and not src_is_psum):
                raise RuntimeError("ERROR: item_sz %d not yet supported"%biasadd_op.item_sz)
            if (biasadd_op.item_sz == 2 and not dst_is_psum):
                out_dtype = "float16"
            elif (biasadd_op.item_sz == 1 and not dst_is_psum):
                raise RuntimeError("ERROR: item_sz %d not yet supported"%biasadd_op.item_sz)
        act_type = "Identity"    
        if (act_op != None):
            act_or_biasadd_op = act_op
            act_type = act_op.data['layer_type']
            layer_name = act_op.data['layer_name']
            # TODO: refactor to some class to determine in_dtype and out_dtype
            if (act_op.item_sz == 2 and not src_is_psum):
                in_dtype = "float16"
            elif (act_op.item_sz == 1 and not src_is_psum):
                raise RuntimeError("ERROR: item_sz %d not yet supported"%act_op.item_sz)
            if (act_op.item_sz == 2 and not dst_is_psum):
                out_dtype = "float16"
            elif (act_op.item_sz == 1 and not dst_is_psum):
                raise RuntimeError("ERROR: item_sz %d not yet supported"%act_op.item_sz)
            if act_type == 'Lrelu':
                alpha = act_op.data['mul_scalar']
        # Two combinations possible: conv + biasadd/act or just biasadd/act                
        # In either cases, biasadd and/or act exists
        assert(act_or_biasadd_op != None)
        batch_item = ofmap_tile.n_id * act_or_biasadd_op.Tn
        dst_x_num = ofmap_tile.tile_rect.dim2d.x
        dst_y_num = ofmap_tile.tile_rect.dim2d.y
        dst_z_num = act_or_biasadd_op.Tn
        dst_y_step, dst_z_step = 1, 1
        src_y_step, src_z_step = 1, 1
        num_partitions = PEArray.NUM_COLS
        if (conv_op != None):
            assert(act_or_biasadd_op.Tn == conv_op.Tn)
            if (dst_is_psum):
                dst_y_step = ofmap_tile.tile_rect.dim2d.x
                dst_z_step = dst_y_step * dst_y_num 
            else:                
                dst_y_step = act_or_biasadd_op.F
                dst_z_step = act_or_biasadd_op.ofmaps_file_params.batch_item_partition_usage_elems_padded
            if src_is_psum:
                src_y_step = ofmap_tile.tile_rect.dim2d.x
                # Kaena-593: ensure no bubble during IFMAP streaming (packed pattern)
                if conv_op.repl_multiple_of_C > 1:
                    src_y_step = conv_op.W // conv_op.stride.x
                src_z_step = ofmap_tile.tile_rect.dim2d.x * ofmap_tile.tile_rect.dim2d.y
            else:
                src_y_step = conv_op.F
                src_z_step = conv_op.ofmaps_file_params.batch_item_partition_usage_elems_padded
            num_partitions = conv_op.ofmap_count
        else:
            # unfused
            dst_y_step = ofmap_tile.tile_rect.dim2d.x
            dst_z_step = act_or_biasadd_op.ofmaps_file_params.batch_item_partition_usage_elems_padded
            src_y_step = ofmap_tile.tile_rect.dim2d.x
            src_z_step = act_or_biasadd_op.ofmaps_file_params.batch_item_partition_usage_elems_padded
            num_partitions = act_or_biasadd_op.ofmap_count
        # SB start addresses
        if src_is_psum:
            src_sb_address = 0
        else:            
            src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(act_or_biasadd_op.ifmaps_file_params, batch_item, ifmap_tile.lower_addr[0])
        if dst_is_psum:
            dst_sb_address = 0
        else:            
            if (conv_op != None):
                dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(act_or_biasadd_op.ofmaps_file_params, batch_item, ofmap_tile.lower_addr[0])
            else:                
                dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(act_or_biasadd_op.ofmaps_file_params, batch_item, ofmap_tile.lower_addr[0])
        waveop_name = layer_name+"/Activation_"+ofmap_tile.id_string            
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : 'Activation',
              'waveop_name'             : waveop_name,
              'layer_name'              : layer_name,
              'tile_id_format'          : ofmap_tile.format,
              'tile_id'                 : ofmap_tile.id_array,
              'activation_func'         : act_type,
              'alpha'                   : alpha,
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
              'bias_start_at_mid_part'  : ofmap_tile.m_id%2 == 1,
            }
        if src_is_psum:
            instr['src_psum_bank_id'] = psum_bank_src
            instr['src_psum_bank_offset'] = 0 
        else:                
            instr['src_sb_address'] = src_sb_address
            instr['src_start_at_mid_part'] = ofmap_tile.m_id%2 == 1
        if dst_is_psum:
            instr['dst_psum_bank_id'] = psum_bank_dst
            instr['dst_psum_bank_offset'] = 0
        else:                
            instr['dst_sb_address'] = dst_sb_address
            instr['dst_start_at_mid_part'] = ofmap_tile.m_id%2 == 1
        psum_bank_used = -1
        if src_is_psum:
            psum_bank_used = psum_bank_src
        if dst_is_psum:
            psum_bank_used = psum_bank_dst
        tpb.waveop_stream.add_linked(instr, dram_bias_waveops, psum_bank_used, new_reader_morsels)
        return instr

    # generate ResAdd instruction and add it to instruction stream
    def gen_join_waveop_inline(self, tpb, op, ifmap_tile, ofmap_tile, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_resadd_waveops, residue_index, residue_sb_addr, new_reader_morsels=[]):
        batch_item = ofmap_tile.batch_item_start
        num_partitions = PEArray.NUM_COLS
        num_partitions = ofmap_tile.channel_count
        start_at_mid_part = ofmap_tile.start_at_mid_part
        sb_addr = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(ifmap_tile.file_params, batch_item, ifmap_tile.lower_addr[0])
        # SB start addresses
        if residue_index == 0:
            # Source-B is PSUM if fused, or SB if unfused
            src_b_sb_address = 0 if src_is_psum else sb_addr
            # Source-A (Residue) is SB
            src_a_sb_address = residue_sb_addr
        else:
            # Source-A is PSUM if fused, or SB if unfused
            src_a_sb_address = 0 if src_is_psum else sb_addr
            # Source-B (Residue) is SB
            src_b_sb_address = residue_sb_addr

        # Destination SB address
        if dst_is_psum:
            dst_sb_address = 0
        else:            
            dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(ofmap_tile.file_params, batch_item, ofmap_tile.lower_addr[0])
        waveop_name = op.data['layer_name'] + "/" + op.data['layer_type'] + "_" + ofmap_tile.id_string
        waveop_type = op.data['layer_type']

        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : waveop_type,
              'waveop_name'             : waveop_name,
              'layer_name'              : op.data['layer_name'],
              'tile_id_format'          : ofmap_tile.format,
              'tile_id'                 : ofmap_tile.id_array,
              'num_partitions'          : num_partitions,
            }

        ofmap_subtile = ofmap_tile.make_pewave()
        ifmap_subtile = ifmap_tile.make_pewave()
        ofmap_subtile.set_waveop_pattern(tpb.statebuffer.file_mapper, instr, 'dst', psum_bank_dst, dst_sb_address, start_at_mid_part)
        if residue_index == 0:
            ifmap_subtile.set_waveop_pattern(tpb.statebuffer.file_mapper, instr, 'src_b', psum_bank_src, src_b_sb_address, start_at_mid_part)
            ifmap_subtile.set_waveop_pattern(tpb.statebuffer.file_mapper, instr, 'src_a', -1, src_a_sb_address, start_at_mid_part)
        else:            
            ifmap_subtile.set_waveop_pattern(tpb.statebuffer.file_mapper, instr, 'src_a', psum_bank_src, src_a_sb_address, start_at_mid_part)
            ifmap_subtile.set_waveop_pattern(tpb.statebuffer.file_mapper, instr, 'src_b', -1, src_b_sb_address, start_at_mid_part)
        tpb.waveop_stream.add_linked(instr, dram_resadd_waveops, psum_bank_src if src_is_psum else -1, new_reader_morsels)
        return instr

    def gen_fused_pool_waveop_inline (self, tpb, ifmap_tile, ofmap_tile, psum_bank_src, start_at_mid_part):
        first_waveop = None
        for z in range(self.pool_op.Tn):
            pool_waveop = self.gen_pool_waveop(tpb, ifmap_tile, ofmap_tile, True, psum_bank_src, start_at_mid_part, z)
            tpb.waveop_stream.add_linked(pool_waveop, [], psum_bank_src)
            if z==0:
                first_waveop = pool_waveop                        
        return first_waveop

    def gen_unfused_pool_waveop_inline (self, tpb, ifmap_tile, ofmap_tile, dram_waveops, prev_waveops, start_at_mid_part):
        first_waveop = None
        for z in range(self.pool_op.Tn):
            pool_waveop = self.gen_pool_waveop(tpb, ifmap_tile, ofmap_tile, False, 0, start_at_mid_part, z)
            tpb.waveop_stream.add_linked(pool_waveop, dram_waveops if z==0 else [], -1)
            if z==0:
                # Add non-DRAM waveops to a previouse_waveops list
                existing_prev_waveops = pool_waveop['previous_waveops']
                for i in prev_waveops:
                    if i not in existing_prev_waveops and i != pool_waveop['waveop_name']:
                        existing_prev_waveops.append(i)
                        print(existing_prev_waveops)
                first_waveop = pool_waveop                        
        return first_waveop

    # Execute softmax (second part, which includes Sum, Reciprocate, Scale)
    def execute_softmax2(self, tpb, inputs):
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
                        tile_id = (n_id, m_id, h_id, w_id, self.conv_op.n, self.conv_op.m, self.conv_op.h, self.conv_op.w)
                        ifmap_tile = Tile(tile_id, self.conv_op.ifmaps_file_params, self.conv_op.Tn, is_pe_input=True,
                                            stridedslice_chan_offset = self.conv_op.stridedslice_chan_offset)
                        ofmap_tile = Tile(tile_id, self.conv_op.ofmaps_file_params, self.conv_op.Tn, is_pe_input=False)
                        self.conv_op.compute_ifmap_ofmap_tile_info(ifmap_tile, ofmap_tile)
                        # compute ofmap tile information (tile startx, starty, height, width)
                        # loops for constructing a tile
                        for c_id in range(self.conv_op.c):
                            for r_id in range(self.conv_op.R):
                                for s_id in range(self.conv_op.S):
                                    ifmap_pewave = PEWave(ifmap_tile, c_id, r_id, s_id)
                                    ofmap_pewave = PEWave(ofmap_tile, c_id, r_id, s_id)
                                    if (self.parent.args.debug > 2): print (ofmap_pewave.id_array)
                                    # execute PEArray matrix multiply, and add to PSUM after first wave
                                    if (self.execute_matmul_waveop(self, ifmap_pewave, ofmap_pewave, inputs, weights, psum_add, 1)):
                                        psum_add = True
                        # tile is done                                   
                        self.waveop_stream.last_main_waveop['stop_tensor_calc'] = True
                        # extract PSUM data
                        psum_bank_src = self.conv_op.get_psum_bank()
                        psum_temp = tpb.pearray.extract_psum(psum_bank_src, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn)
                        # go through the remaining operations
                        psum_temp = tpb.pool.reciprocate(psum_temp, self.conv_op.M)
                        psum_bank_dst = psum_bank_src
                        tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, psum_temp)
                        tpb.gen_recip_waveop_inline(tpb, self.conv_op, psum_bank_src, True, psum_bank_dst)
                        psum_bank_src = psum_bank_dst
                        # loops for final scaling
                        for c_id in range(ceildiv(self.conv_op.C, PEArray.NUM_COLS)):
                            wave_id = ofmap_tile.make_pewave()
                            pearray_packed_ifmaps = self.conv_op.pack_wave_ifmaps(inputs, wave_id, 1, for_softmax=True)
                            scale_val = tpb.pearray.extract_psum(psum_bank_src, 0, 1)[0,0]
                            psum_temp = tpb.pool.scale(pearray_packed_ifmaps, scale_val)
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
                                                ofmap_tile.tile_rect.lower.y : ofmap_tile.tile_rect.lower.y + ofmap_tile.tile_rect.dim2d.y, 
                                                ofmap_tile.tile_rect.lower.x : ofmap_tile.tile_rect.lower.x + ofmap_tile.tile_rect.dim2d.x]\
                                            = result_tile[0:ofmap_tile.tile_rect.dim2d.y, 0:ofmap_tile.tile_rect.dim2d.x]
                                # for scheduling, map resulting tile into portion of atom that is itself mapped to a portion in DRAM (file)
                                # TODO: fix waveop generation
                                #dram_output_waveops += tpb.statebuffer.circbuf_scratch.write_data_region(
                                #                            ofmap_tile, 
                                #                            output_params_op.ofmap_tile_lower_addr[z], 
                                #                            output_params_op.ofmap_tile_upper_addr[z], 
                                #                            output_params_op.ifmap_count,   # Do we have to use IFMAP count here?
                                #                            self.waveop_stream.last_main_waveop)
                       # The scale_add destination need to be adjusted after the above writes to data region
                        if (self.waveop_stream.last_main_waveop['waveop_type'] == "ScaleAdd"):
                            sb_addr = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.conv_op.ofmaps_file_params, 0, ofmap_tile.lower_addr[0])
                            self.waveop_stream.last_main_waveop['dst_sb_address'] = sb_addr
                        self.waveop_stream.add_outputs(dram_output_waveops)
                        if self.args.abstract_mem:
                            if len(dram_output_waveops) > 0:
                                self.waveop_stream.last_main_waveop = None

                        # Advance to new bank (ping-pong between 0 and 1) for PEArray, while the old bank is being processed by other engines
                        self.conv_op.set_psum_bank((self.conv_op.get_psum_bank()+1)%4)
                        tpb.pearray.last_psum_bank_used = self.conv_op.get_psum_bank()
                        psum_add = False

    def execute_pool_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_dst):
        first_op = self.first_op
        ifmap_subtile = ifmap_tile.make_pewave(self.first_op.stridedslice_chan_offset)
        ifmaps_data = first_op.pack_wave_ifmaps_unfused_pooling(ifmap_tile.file_params.dram_data, ifmap_subtile)
        input_tilex = ifmap_tile.tile_rect.dim2d.x
        input_tiley = ifmap_tile.tile_rect.dim2d.y
        output_tiley = first_op.ofmap_full_tiley_sz
        output_tilex = first_op.ofmap_full_tilex_sz
        ifmaps_data_extract = ifmaps_data [0:input_tiley*input_tilex*first_op.Tn, :]
        layer_type = first_op.data['layer_type'] 
        if (layer_type == "AvgPool" or layer_type == "MaxPool"):
            assert(first_op.dst_is_psum == False)
            pool_decomposer = Pool.init_from_rectangles(
                                ifmap_tile.padded_tile_rect,
                                ifmap_tile.tile_rect,
                                ofmap_tile.tile_rect,
                                first_op.padWN,
                                first_op.pool_window,
                                first_op.stride)
            pool_waves = pool_decomposer.Decompose()
            print("execute_pool_tile::pool_waves size = %d"%len(pool_waves))
            for pool_wave in pool_waves:
                ifmaps_data_extract = ifmap_tile.get_subtile_data_from_file(pool_wave.ifmap)
                subtile_data = tpb.pool.pool2(
                                        type             = layer_type, 
                                        in_array         = ifmaps_data_extract, 
                                        stride           = pool_wave.stride, 
                                        pool_window      = pool_wave.window, 
                                        Tn               = first_op.Tn, 
                                        ifmap_tilex_sz   = pool_wave.ifmap.dim2d.x, 
                                        ifmap_tiley_sz   = pool_wave.ifmap.dim2d.y,
                                        ofmap_tilex_sz   = pool_wave.ofmap.dim2d.x, 
                                        ofmap_tiley_sz   = pool_wave.ofmap.dim2d.y)
                ofmap_tile.set_subtile_data_in_file2(pool_wave.ofmap, subtile_data)
        elif (layer_type == "Multiply" or layer_type == "ResAdd" or layer_type == "Maximum" or layer_type == "Minimum"):
            if ("mul_scalar" in first_op.data):
                assert (layer_type == "Multiply")
                tile_data_flatten = tpb.pool.scale(ifmaps_data_extract, first_op.data['mul_scalar'])
            else:
                residue_file_params = first_op.prev[first_op.residue_index].ofmaps_file_params
                residue_tile = ofmap_tile.copy()
                residue_tile.file_params = residue_file_params
                residue_ifmaps = residue_tile.get_tile_data_from_file(flatten=True)
                if (layer_type == "ResAdd"):
                    tile_data_flatten = tpb.pool.resadd(ifmaps_data_extract, residue_ifmaps)
                elif (layer_type == "Multiply"):                                    
                    tile_data_flatten = tpb.pool.multiply(ifmaps_data_extract, residue_ifmaps)
                elif (layer_type == "Maximum"):                                    
                    tile_data_flatten = tpb.pool.maximum(ifmaps_data_extract, residue_ifmaps)
                elif (layer_type == "Minimum"):                                    
                    tile_data_flatten = tpb.pool.minimum(ifmaps_data_extract, residue_ifmaps)
                else:
                    raise RuntimeError("ERROR: don't know how to handle vector op %s for layer %s"%(layer_type, first_op.data['layer_name']))
        elif (layer_type == "BiasAdd"):
            bias_chan_start = ofmap_tile.m_id * PEArray.NUM_COLS
            bias_chan_end = min(bias_chan_start + PEArray.NUM_COLS, first_op.M)
            bias_temp = self.biasadd_op.bias_file_params.dram_data
            bias = bias_temp.flatten()
            #x = DBG_DUMP_PSUM_COL("BiasAdd: ", ifmaps_data_extract, 0)
            tile_data_flatten = tpb.activate.biasadd(ifmaps_data_extract, bias[bias_chan_start : bias_chan_end])
            #x = DBG_DUMP_PSUM_COL("BiasAdd: ", tile_data_flatten, 0)
        elif (layer_type == "ClipByValue"):
            min_val = self.first_op.data['clip_value_min']
            max_val = self.first_op.data['clip_value_max']
            tile_data_flatten = tpb.pool.clipbyvalue(ifmaps_data_extract, min_val, max_val)
        elif re.search(self.act_ops_regex, layer_type):
            alpha = 1.0
            if layer_type == 'Lrelu':
                alpha = self.first_op.data['mul_scalar']
            tile_data_flatten = tpb.activate.act(layer_type, ifmaps_data_extract, alpha)
        else:
            raise RuntimeError("ERROR: cannot execute %s in execute_unfused_first_op"%layer_type)

        # Set resulting tile data into OFMAP (for pooling, we went ahead and do subtile pooling)
        if not (layer_type == "AvgPool" or layer_type == "MaxPool"):
            if first_op.dst_is_psum:
                tpb.pearray.write_psum(psum_bank_dst, 0, tile_data_flatten.shape[0], tile_data_flatten)
            else:
                ofmap_tile.set_tile_data_in_file(tile_data_flatten)

        for z in range(first_op.Tn):
            if (self.args.debug > 2): print("TRACE execute_unfused_first_op %s: tile %s done, ifmap_tile_lower_addr %d ifmap_tile_upper_addr %d psum_bank %d, ofmap_tile_lower_addr %d ofmap_tile_upper_addr %dx"\
                        %(first_op.data["layer_name"], ofmap_tile.id_string, ifmap_tile.lower_addr[z], ifmap_tile.upper_addr[z], -1, ofmap_tile.lower_addr[z], ofmap_tile.upper_addr[z]))

    """ Obtain producers of tile data for RAW dependencies; 
        if data doesn't exist in SB, generate SBAtomLoad waveops and return them
        Returns: (list of prev waveops, list of DRAM waveops)
    """
    def get_producers_for_subtile_region (self, tpb, fmap_subtile):
        # New 1
        # Start of RAW part
        tile = fmap_subtile.tile
        batch_item = tile.n_id * tile.Tn
        dram_fmaps_waveops = []
        list_of_accessors = []
        new_reader_morsels = []
        for z in range(tile.Tn):
            if 1: # (tile.m_id%2 == 0):
                (writers, readers, waveops, reader_morsels) \
                    = tpb.statebuffer.file_mapper.read_file_data_region(
                        tpb.waveop_stream.waveop_count,
                        tpb.waveop_stream,
                        tile.file_params,
                        batch_item + z,
                        tile.lower_addr[z], 
                        tile.lower_to_upper_len_bytes[z])
                dram_fmaps_waveops += waveops
                # don't include readers in dependencies since this pool is a 
                # reader, and we don't need to add RAR dependency
                list_of_accessors += writers
                new_reader_morsels += reader_morsels
                #print(last_writer, last_reader)
        prev_waveops = self.extract_predecessors(
                list_of_accessors   = list_of_accessors,
                waveop_list         = tpb.waveop_stream,
                dram_waveops        = dram_fmaps_waveops,
                relax_dependencies  = self.args.relax_dependencies,
                enable_cleanup      = self.args.enable_cleanup)

        if dram_fmaps_waveops != []  and self.args.no_inter_layer_load:
            if (not self.first_op.is_input and len(dram_fmaps_waveops) > 0):
                raise RuntimeError("There are DRAM loads when option \
                        no_inter_layer_load is set")
        # End of RAW part
        return (prev_waveops, dram_fmaps_waveops, new_reader_morsels)                        

    """ Mark producers within output tile's region in SB;
        extract WAW dependencies and attach to first waveop of current
        Tn batch items
    """
    def mark_producers_for_subtile_region (self, tpb, fmap_subtile, waveop):
        # New 3
        # Start of WAW and WAR part
        tile = fmap_subtile.tile
        batch_item = tile.n_id * tile.Tn
        start_at_mid_part = tile.m_id%2 == 1
        dram_output_waveops = []                            
        list_of_accessors = []
        for z in range(tile.Tn):
            # for scheduling, map resulting tile into portion of atom that is 
            # itself mapped to a portion in DRAM (file)
            # only record the writer to SB chunks in below code; 
            # use flush_file to write chunks to DRAM
            # (mark writer using ID of last waveop of current Tn batch items)
            (writers, readers, waveops) \
                = tpb.statebuffer.file_mapper.write_file_data_region(
                                        tpb.waveop_stream.waveop_count - 1,    # adjust since pool waveop already generated
                                        tpb.waveop_stream,
                                        tile.file_params,
                                        batch_item + z,
                                        tile.lower_addr[z], 
                                        tile.lower_to_upper_len_bytes[z], 
                                        start_at_mid_part)
            #assert(len(waveops) == 0)                            
            # TODO: roll this code into write_file_data_region
            list_of_accessors += writers + readers    # Include readers for WAR dependencies and writers for WAW dependencies
            dram_output_waveops += waveops

            if (self.args.debug > 3): print("TRACE execute_unfused_first_op %s:\
                    tile %s done, ofmap_tile_lower_addr %d\
                    ofmap_tile_upper_addr %dx"\
                        %(self.first_op.data["layer_name"], 
                            tile.id_string, 
                            tile.lower_addr[z], 
                            tile.upper_addr[z]))

        # Attach WAW dependency on the first waveop of current Tn batch items
        prev_waveops = waveop['previous_waveops']
        if (len(dram_output_waveops) > 0):
            for i in dram_output_waveops:
                try:
                    tpb.waveop_stream.append_check(i, -1)
                    prev = i['waveop_name']
                except:
                    prev = i
                if (prev not in prev_waveops and prev != waveop['waveop_name']):
                    prev_waveops.append(prev)
        predec_list_by_name = self.extract_predecessors(
                list_of_accessors   = list_of_accessors, 
                waveop_list         = tpb.waveop_stream,
                dram_waveops        = [],
                relax_dependencies  = False,
                enable_cleanup      = self.args.enable_cleanup)
        self.attach_predecessors(waveop, predec_list_by_name)

    # Extract list of predecessor waveop names from list of accessors
    def extract_predecessors(self, list_of_accessors, waveop_list, dram_waveops, relax_dependencies, enable_cleanup = False):
        predec_list_by_name = []
        if dram_waveops == []:
            if list_of_accessors != []:
                if enable_cleanup:
                    for latest_accessor in list_of_accessors:
                        if latest_accessor >= 0:
                            latest_accessor_waveop = waveop_list[latest_accessor]
                            if not relax_dependencies \
                                    or (latest_accessor_waveop['waveop_type'] != 'Activation' \
                                        and latest_accessor_waveop['waveop_type'] != 'Pool'):
                                predec_list_by_name.append(latest_accessor_waveop['waveop_name'])
                else:
                    latest_accessor = max(list_of_accessors)
                    if True: #for latest_accessor in list_of_accessors:
                        if latest_accessor >= 0:
                            latest_accessor_waveop = waveop_list[latest_accessor]
                            if not relax_dependencies \
                                    or (latest_accessor_waveop['waveop_type'] != 'Activation' \
                                        and latest_accessor_waveop['waveop_type'] != 'Pool'):
                                predec_list_by_name.append(latest_accessor_waveop['waveop_name'])
        return predec_list_by_name                        

    # Attach dependencies on waveop
    def attach_predecessors(self, waveop, predec_list_by_name):                
        prev_waveops_by_name = waveop['previous_waveops']
        for i in predec_list_by_name:
            if i not in prev_waveops_by_name and i != waveop['waveop_name']:
                prev_waveops_by_name.append(i)

    def emit_waveops_pool_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_dst):
        # wave loop ordering scheme: nmhw
        first_op = self.first_op
        batch_item = ofmap_tile.n_id * first_op.Tn
        start_at_mid_part = ofmap_tile.m_id%2 == 1
        if (first_op.data['layer_type'] == "AvgPool" 
                or first_op.data['layer_type'] == "MaxPool"):
            pool_decomposer = Pool.init_from_rectangles(
                                ifmap_tile.padded_tile_rect,
                                ifmap_tile.tile_rect,
                                ofmap_tile.tile_rect,
                                self.first_op.padWN,
                                self.first_op.pool_window,
                                self.first_op.stride)
            pool_waves = pool_decomposer.Decompose()
            for pool_wave in pool_waves:
                ifmap_tile_subtile \
                    = PoolSubtile(ifmap_tile, pool_wave.ifmap, pool_wave.window)
                ofmap_tile_subtile \
                    = PoolSubtile(ofmap_tile, pool_wave.ofmap, pool_wave.window)
                (prev_waveops, dram_ifmaps_waveops, _) \
                    = self.get_producers_for_subtile_region (
                            tpb          = tpb, 
                            fmap_subtile = ifmap_tile_subtile)
                waveop  = self.gen_unfused_pool_waveop_inline(
                        tpb               = tpb, 
                        ifmap_tile        = ifmap_tile_subtile, 
                        ofmap_tile        = ofmap_tile_subtile, 
                        dram_waveops      = dram_ifmaps_waveops, 
                        prev_waveops      = prev_waveops, 
                        start_at_mid_part = start_at_mid_part)
                self.mark_producers_for_subtile_region(
                        tpb           = tpb, 
                        fmap_subtile  = ofmap_tile_subtile, 
                        waveop        = waveop)
        else:   
            psum_bank_dst = -1
            bias_start = 0
            biasadd_op = None
            act_op = None
            ifmap_tile_subtile \
                    = PoolSubtile(ifmap_tile, ifmap_tile.tile_rect, Dim2D(1,1))
            ofmap_tile_subtile \
                    = PoolSubtile(ofmap_tile, ofmap_tile.tile_rect, Dim2D(1,1))
            (prev_waveops, dram_ifmaps_waveops, new_reader_morsels) \
                = self.get_producers_for_subtile_region (
                        tpb          = tpb, 
                        fmap_subtile = ifmap_tile_subtile)
            if (first_op.data['layer_type'] == "BiasAdd"): 
                bias_start = ofmap_tile.m_id * PEArray.NUM_COLS * first_op.item_sz
                (writers, readers, dram_bias_waveops, reader_morsels) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                tpb.waveop_stream.waveop_count,
                                                tpb.waveop_stream,
                                                first_op.bias_file_params,
                                                0,  # batch_item is not needed for bias
                                                bias_start,
                                                first_op.item_sz)
                #print(first_op.data['layer_name'], writers)
                dram_ifmaps_waveops += dram_bias_waveops
                new_reader_morsels += reader_morsels
                prev_wr_waveops = self.extract_predecessors(
                        list_of_accessors   = writers,
                        waveop_list         = tpb.waveop_stream,
                        dram_waveops        = [],
                        relax_dependencies  = self.args.relax_dependencies,
                        enable_cleanup      = self.args.enable_cleanup)
                for i in prev_wr_waveops:
                    if i not in prev_waveops:
                        prev_waveops.append(i)
                biasadd_op = first_op
            else:
                act_op = first_op
            if first_op.dst_is_psum:
                psum_bank_dst = first_op.get_psum_bank()
                first_op.set_psum_bank((psum_bank_dst+1)%4)
                tpb.pearray.last_psum_bank_used = first_op.get_psum_bank()
            if re.search("Multiply|ResAdd|ClipByValue", first_op.data['layer_type']):
                if ("mul_scalar" in first_op.data or first_op.data['layer_type'] == "ClipByValue"):
                    if first_op.data['layer_type'] == "ClipByValue":
                        scale_or_min_val = first_op.data['clip_value_min']
                        add_or_max_val = first_op.data['clip_value_max']
                    else:                        
                        scale_or_min_val = first_op.data['mul_scalar']
                        add_or_max_val = 0.0
                    waveop = self.gen_scaleadd_or_clip_waveop_inline(
                                tpb              = tpb,
                                op               = first_op, 
                                ifmap_tile       = ifmap_tile, 
                                ofmap_tile       = ofmap_tile,
                                src_is_psum      = False, 
                                psum_bank_src    = -1, 
                                dst_is_psum      = first_op.dst_is_psum, 
                                psum_bank_dst    = psum_bank_dst, 
                                dram_waveops     = dram_ifmaps_waveops, 
                                scale_or_min_val = scale_or_min_val,
                                add_or_max_val   = add_or_max_val)
                else:
                    residue_file_params = first_op.prev[first_op.residue_index].ofmaps_file_params
                    if residue_file_params.mapped_params.start_addr == ofmap_tile.file_params.mapped_params.start_addr:
                        if not first_op.is_input:
                            residue_file_params.mapped_params.modify_in_place = True
                    residue_tile = copy.copy(ifmap_tile)
                    residue_tile.file_params = residue_file_params
                    residue_subtile = PEWave(residue_tile, 0, 0, 0, first_op.stridedslice_chan_offset)
                    (_, dram_resadd_waveops, reader_morsels) = self.get_producers_for_subtile_region(tpb, residue_subtile)
                    new_reader_morsels += reader_morsels
                    residue_file_addr = residue_file_params.ravel_nchw(
                            ofmap_tile.n_id * ofmap_tile.Tn,
                            ofmap_tile.m_id // 2 * PEArray.NUM_ROWS,
                            ofmap_tile.tile_rect.lower.y,
                            ofmap_tile.tile_rect.lower.x)
                    residue_sb_addr = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(residue_tile.file_params, batch_item, residue_file_addr)
                    waveop = self.gen_join_waveop_inline(
                            tpb             = tpb, 
                            op              = first_op, 
                            ifmap_tile      = ifmap_tile,
                            ofmap_tile      = ofmap_tile,
                            src_is_psum     = False, 
                            psum_bank_src   = -1, 
                            dst_is_psum     = first_op.dst_is_psum, 
                            psum_bank_dst   = psum_bank_dst, 
                            dram_resadd_waveops = dram_ifmaps_waveops+dram_resadd_waveops,
                            residue_index   = first_op.residue_index,
                            residue_sb_addr   = residue_sb_addr,
                            new_reader_morsels = new_reader_morsels)
            else:                    
                waveop = self.gen_act_waveop_inline(
                        tpb               = tpb, 
                        biasadd_op        = biasadd_op, 
                        act_op            = act_op, 
                        conv_op           = None, 
                        ifmap_tile        = ifmap_tile, 
                        ofmap_tile        = ofmap_tile, 
                        src_is_psum       = False, 
                        psum_bank_src     = -1, 
                        dst_is_psum       = first_op.dst_is_psum, 
                        psum_bank_dst     = psum_bank_dst, 
                        dram_bias_waveops = dram_ifmaps_waveops, 
                        bias_start        = bias_start,
                        new_reader_morsels = new_reader_morsels)
            self.attach_predecessors(waveop, prev_waveops)
            self.mark_producers_for_subtile_region(
                    tpb           = tpb, 
                    fmap_subtile  = ofmap_tile_subtile, 
                    waveop        = waveop)

        #if self.args.abstract_mem:
        #    if len(dram_output_waveops) > 0:
        #        self.waveop_stream.last_main_waveop = None
    # Execute an unfused pooling operator
    def execute_unfused_pool_op(self, tpb, batch_item):
        first_op = self[0]
        n_id = batch_item // first_op.Tn
        for m_id in range(first_op.m):
            for h_id in range(first_op.h):
                for w_id in range(first_op.w):
                    tile_id = (n_id, m_id, h_id, w_id, first_op.n, first_op.m, first_op.h, first_op.w)
                    ifmap_tile = Tile(tile_id, self.first_op.ifmaps_file_params, self.first_op.Tn, is_pe_input=False,
                                        stridedslice_chan_offset = first_op.stridedslice_chan_offset)
                    ofmap_tile = Tile(tile_id, self.last_op.ofmaps_file_params, self.last_op.Tn, is_pe_input=False)
                    self.first_op.compute_ifmap_ofmap_tile_info(ifmap_tile, ofmap_tile)
                    psum_bank_dst = self.first_op.get_psum_bank()
                    self.execute_pool_tile(tpb, ifmap_tile, ofmap_tile, psum_bank_dst)
                    self.emit_waveops_pool_tile(tpb, ifmap_tile, ofmap_tile, psum_bank_dst)
                    # execute subsequent instructions
                    self.execute_postconv_tile_ops(tpb, ofmap_tile, ofmap_tile, psum_bank_dst)
                    self.first_op.set_psum_bank((psum_bank_dst + 1) % 4)
                    tpb.pearray.last_psum_bank_used = self.first_op.get_psum_bank()

    # Execute an unfused transpose operator
    def execute_unfused_transpose_op(self, tpb, batch_item):
        self.last_op.ofmaps_file_params.dram_data = \
            np.transpose(self.first_op.ifmaps_file_params.dram_data, self.first_op.data['perm'])
        (last_writer, last_reader, waveops) = tpb.statebuffer.file_mapper.write_file_data_region(
                                    -1,  # signifies a pass-through (keep old writers of region)
                                    tpb.waveop_stream,
                                    self.last_op.ofmaps_file_params,
                                    batch_item,
                                    0,
                                    self.first_op.ifmaps_file_params.file_sz,   # mark entire file
                                    False)

    def execute_unfused_reshape_op(self, tpb, batch_item):
        self.first_op.ofmaps_file_params.dram_data = self.first_op.ifmaps_file_params.dram_data.view()
        try:
            self.first_op.ofmaps_file_params.dram_data.shape = self.first_op.ofmaps_file_params.file_dims.shape_tuple
        except AttributeError as e:                            
            raise RuntimeError ("ERROR: Cannot reshape data without copying; please implement reshape with copy")
        (last_writer, last_reader, waveops) = tpb.statebuffer.file_mapper.write_file_data_region(
                                    -1,  # signifies a pass-through (keep old writers of region)
                                    tpb.waveop_stream,
                                    self.first_op.ofmaps_file_params,
                                    batch_item,
                                    0,
                                    self.first_op.ofmaps_file_params.file_sz,   # mark entire file
                                    False)

    def execute_unfused_slice_op(self, tpb, batch_item):
        ofp = self.first_op.ofmaps_file_params
        ifp = self.first_op.ifmaps_file_params
        slice_w_begin = self.first_op.slice_offset.x
        slice_w_stop = self.first_op.slice_offset.x + self.first_op.slice_size.x
        slice_h_begin = self.first_op.slice_offset.y
        slice_h_stop = self.first_op.slice_offset.y + self.first_op.slice_size.y
        #print("Slice: w axis [%d:%d], h axis [%d:%d]"%(slice_w_begin, slice_w_stop, slice_h_begin, slice_h_stop))
        if ifp.file_dims.format_str == 'HNWC':
            ofp.dram_data = ifp.dram_data[ slice_h_begin : slice_h_stop, :, slice_w_begin : slice_w_stop, : ]
            ofp.dram_data.shape = ofp.file_dims.shape_tuple
        elif ofp.file_dims.format_str == 'NCW':
            ofp.dram_data = ifp.dram_data[:, :, slice_w_begin : slice_w_stop]
        elif ofp.file_dims.format_str == 'NCHW':
            ofp.dram_data = ifp.dram_data[:, :, slice_h_begin : slice_h_stop, slice_w_begin : slice_w_stop]
        else:
            raise RuntimeError("Format %s is not supported for Slice operation"%(ofp.file_dims.format_str))

        if not self.first_op.is_input:
            (last_writer, last_reader, waveops) = tpb.statebuffer.file_mapper.write_file_data_region(
                                    -1,  # signifies a pass-through (keep old writers of region)
                                    tpb.waveop_stream,
                                    self.first_op.ofmaps_file_params,
                                    batch_item,
                                    0,
                                    self.first_op.ofmaps_file_params.file_sz,   # mark entire file
                                    False)
        else:
            self.no_writer_for_save = True  # pass-thru so there's no writer for SBAtomSave (so don't save this fused-op output)

    # weights for simple channel slice extraction 
    def gen_weights_for_chan_slicing(self, tpb, slice_start, slice_stop, m_id):
        op = self.first_op
        if slice_stop < slice_start:
            slice_stop += op.C
        slice_count         = slice_stop - slice_start
        assert(slice_count > 0 and slice_count < PEArray.NUM_COLS)
        weights_shape       = [op.C, 1, 1, slice_count]
        weights_shape_dims  = ShapeDims("CRSM", tuple(weights_shape))
        weights_data        = np.zeros(weights_shape, dtype = op.data_type)
        identity            = np.identity(slice_count, dtype = op.data_type)
        identity_exp        = np.expand_dims(np.expand_dims(identity, 1), 1)
        weights_data[slice_start:slice_stop, :, :, :] = identity_exp
        weights_file        = op.data['ref_file'].replace(".npy", "_concat_weight_CRSM_m%d.npy"%m_id)
        np.save(weights_file, weights_data)
        #if (op.weights_file_params == None):
        op.weights_file_params = FileParams(
            file_name       = weights_file, 
            file_dims       = weights_shape_dims, 
            data_type       = op.data_type, 
            op_params       = op, 
            args            = op.parent.args, 
            contain_weights = True)
        op.weights_file_params.layer_name = op.data['layer_name']
        op.weights_file_params.load_file()
        op.weights_file_params.consumers.append(op)
        op.weights_file_params_concat.append(op.weights_file_params)
        self.map_files_gen_during_exec(tpb, op.weights_file_params)
        self.live_mapped_file_params.append(op.weights_file_params)
        #if (op.data['layer_name'] == 'mixed10/concat'):
        self.print_SB_addr(op.weights_file_params)

    def execute_unfused_stridedslice_op (self, tpb, batch_item):
        #print ("Stridedslice node name = %s"%self.first_op.data['layer_name'])
        first_op = self[0]
        n_id = batch_item // first_op.Tn
        ifmap_file_params = self.first_op.ifmaps_file_params
        self.first_op.R = 1
        self.first_op.S = 1
        for m_id in range(first_op.m):
            first = True
            slice_start = self.first_op.data['channel_slice'][0]
            slice_stop  = self.first_op.data['channel_slice'][1]
            self.gen_weights_for_chan_slicing(tpb, slice_start, slice_stop, m_id)
            for h_id in range(first_op.h):
                for w_id in range(first_op.w):
                    ifmap_tile_id = (n_id, 0, h_id, w_id, first_op.n\
                                     , first_op.m, first_op.h, first_op.w)
                    ofmap_tile_id = (n_id, m_id, h_id, w_id, first_op.n\
                                     , first_op.m, first_op.h, first_op.w)
                    next_ofmap_start = 0
                    ifmap_tile =\
                      Tile(ifmap_tile_id\
                           #, self.first_op.ifmaps_file_params\
                           , ifmap_file_params\
                           , self.first_op.Tn\
                           , is_pe_input=True)
                    ofmap_tile =\
                        Tile(ofmap_tile_id\
                             , self.last_op.ofmaps_file_params\
                             , self.last_op.Tn\
                             , is_pe_input=False)
                    self.first_op.compute_ifmap_ofmap_tile_info(ifmap_tile
                                                                ,ofmap_tile)

                    pewave = PEWave(ifmap_tile, 0, 0, 0)
                    packed_ifmap = first_op.pack_wave_ifmaps(ifmap_file_params.dram_data, pewave, pewave, 1, False)
                    weight = first_op.weights_file_params.dram_data
                    pewave_for_weights = PEWave(ifmap_tile, 0, 0, 0)
                    ofmap_pewave = PEWave(ofmap_tile, 0, 0, 0)
                    packed_weights = self.first_op.pack_wave_conv_weights(weight
                                                                 , pewave_for_weights
                                                                 , ofmap_pewave
                                                                 , 1)

                    tpb.pearray.wave_fp16_mm(packed_ifmap
                                             , packed_weights
                                             , self.first_op.psum_bank_dst
                                             , False)

                    concat_collaterals = [ []
                             , ifmap_file_params 
                             , [slice_start, slice_stop-1]
                             , [0, first_op.weights_file_params.file_dims.M-1]]

                    self.emit_waveop_concat_tile(
                                                 tpb
                                                 , ifmap_tile
                                                 , ofmap_tile
                                                 , concat_collaterals
                                                 , next_ofmap_start
                                                 , 1
                                                 , 1
                                                 , first
                                                )
                    first = False
                    psum_bank_src = self.first_op.psum_bank_dst
                    MM_data = tpb.pearray.extract_psum(psum_bank_src, 0, ifmap_tile.tile_rect.get_tot_size())
                    ofmap_tile.set_tile_data_in_file(MM_data)
                    self.execute_postconv_tile_ops(tpb
                                                   , ofmap_tile
                                                   , ofmap_tile
                                                   , psum_bank_src)
                    tpb.pearray.last_psum_bank_used = self.first_op.get_psum_bank()
                    self.first_op.psum_bank_dst += 1
                    self.first_op.psum_bank_dst %= PEArray.PSUM_NUM_BANKS

    def execute_unfused_concat_op (self, tpb, batch_item):
        def move_psum_to_ofmap (tpb, ifmap_tile, ofmap_tile):
            MM_data = tpb.pearray.extract_psum(self.first_op.psum_bank_dst, 0\
                    , ifmap_tile.tile_rect.get_tot_size())
            ofmap_tile.set_tile_data_in_file(MM_data)
            return 
        def get_first_item(t):
            return t[0]

        first_op = self[0]
        n_id = batch_item // first_op.Tn
        ifmap_file_params = self.first_op.ifmaps_file_params_concat
        concat = Concat.init_from_file_params(ifmap_file_params)
        concat.PerformConcatDecomposition(forward_move = True\
                                          , generate_weight_files = True)
        keys = list(concat.subtile_infos.keys())
        keys = sorted(keys, key=get_first_item)
        for m_id in range(len(keys)):
            subtile_info = concat.subtile_infos[keys[m_id]]
            [mfilters,ifmap_file_params_tile,ifmap_channel_ranges]=subtile_info
            first = True
            for h_id in range(first_op.h):
                for w_id in range(first_op.w):
                    ifmap_tile_id = (n_id, 0, h_id, w_id, first_op.n
                                     , first_op.m, first_op.h, first_op.w)
                    ofmap_tile_id = (n_id, m_id, h_id, w_id, first_op.n
                                     , first_op.m, first_op.h, first_op.w)
                    next_ofmap_start = 0
                    for tile in range(len(ifmap_file_params_tile)):
                        ifmap_tile = Tile(ifmap_tile_id
                               , ifmap_file_params_tile[tile]
                               , self.first_op.Tn
                               , is_pe_input=True)
                        ofmap_tile = Tile(ofmap_tile_id
                                 , self.last_op.ofmaps_file_params
                                 , self.last_op.Tn
                                 , is_pe_input=False)
                        self.first_op.compute_ifmap_ofmap_tile_info(ifmap_tile
                                                                    ,ofmap_tile)
                        concat_collaterals = [mfilters[tile]
                                 , ifmap_file_params_tile[tile]
                                 , ifmap_channel_ranges[tile]
                                 , keys[m_id]]
                        next_ofmap_start = self.execute_concat_tile(
                                                         tpb
                                                         , ifmap_tile
                                                         , ofmap_tile
                                                         , concat_collaterals
                                                         , next_ofmap_start
                                                         , tile == 0
                                                         , first
                                                        )
                        self.emit_waveop_concat_tile(tpb
                                                     , ifmap_tile
                                                     , ofmap_tile
                                                     , concat_collaterals
                                                     , next_ofmap_start
                                                     , tile == 0
                                                     , 1
                                                     , first
                                                    )
                    if (first == True): first = False
                    move_psum_to_ofmap(tpb, ifmap_tile, ofmap_tile)
                    psum_bank_src = self.first_op.psum_bank_dst
                    self.execute_postconv_tile_ops(tpb
                                                   , ofmap_tile
                                                   , ofmap_tile
                                                   , psum_bank_src)
                    tpb.pearray.last_psum_bank_used = self.first_op.get_psum_bank()
                    self.first_op.psum_bank_dst += 1
                    self.first_op.psum_bank_dst %= PEArray.PSUM_NUM_BANKS

    def emit_waveop_concat_tile (self
                                 , tpb
                                 , ifmap_tile
                                 , ofmap_tile
                                 , concat_collaterals
                                 , ofmap_start
                                 , start_tensor_calc
                                 , num_weights
                                 , first = False
                                ):
        [_, ifmap_file_param, ifmap_channel_range, ofmap_range] =\
                concat_collaterals
        c_start_idx = int(math.floor(ifmap_channel_range[1] / PEArray.NUM_ROWS))
        pewave = PEWave(ifmap_tile, c_start_idx, 0, 0)
        pewave.ifmap_channel_count = PEArray.NUM_ROWS
        ofmap_pewave = PEWave(ofmap_tile, 0, 0, 0)
        self.first_op.compute_ifmap_ofmap_pewave_info(pewave, pewave, False)
        (prev_waveops, dram_ifmaps_waveops, _) =\
                self.get_producers_for_subtile_region (
                    tpb = tpb, fmap_subtile = pewave)
        # 9-10-2018
        if (first == True):
          total_size_weights =(
           self.first_op.weight_wave_upper_addr
             - self.first_op.weight_wave_lower_addr
           + self.first_op.item_sz) * num_weights
          (writers, readers, dram_weights_waveops, _) =\
                  tpb.statebuffer.file_mapper.read_file_data_region(
                      tpb.waveop_stream.waveop_count
                      , tpb.waveop_stream
                      , self.first_op.weights_file_params
                      , 0  # batch_item doesn't apply for weights
                      , self.first_op.weight_wave_lower_addr
                      , total_size_weights
                      , start_at_mid_part = False
                      , end_after_mid_part = True
                      , repl_multiple_of_C = 1
                      , dont_put_prev_ops = False
                  )
          self.concat_dram_weights_waveops = dram_weights_waveops
        dram_weights_waveops = self.concat_dram_weights_waveops
        #print ("len(dram_weights_waveops) = %d"%len(dram_weights_waveops))
        psum_add = start_tensor_calc == False
        mms = self.gen_matmul_waveop(tpb\
                                     , pewave\
                                     , ofmap_pewave\
                                     , psum_add\
                                     , dram_weights_waveops)
        for i in range(len(mms)):
            #print ("mm_i name = %s"%mm_i['waveop_name'])
            if (i == 0):
                weights = dram_weights_waveops
            else:
                weights = None
            if (first == True):
              tpb.waveop_stream.add_linked(
                  mms[i], weights, self.first_op.psum_bank_dst)
            else:
              tpb.waveop_stream.add_linked(
                  mms[i], [], self.first_op.psum_bank_dst)
            prev_record_in_waveop = mms[i]['previous_waveops']
            for prev in prev_waveops:
                if prev not in prev_record_in_waveop:
                    prev_record_in_waveop.append(prev)
            ofmap_pewave = PEWave(ofmap_tile, 0, 0, 0)
            self.mark_producers_for_subtile_region(
                tpb           = tpb, 
                fmap_subtile  = ofmap_pewave,
                waveop        = mms[i])

    def execute_concat_tile (self
                             , tpb
                             , ifmap_tile
                             , ofmap_tile
                             , concat_collaterals
                             , ofmap_start
                             , start_tensor_calc
                             , first = False
                            ):
        [mfilter, ifmap_file_param, ifmap_channel_range, ofmap_range] =\
                concat_collaterals
        # NCHW
        ifmap_data = ifmap_file_param.dram_data
        c_start_idx = int(math.floor(ifmap_channel_range[1] / PEArray.NUM_ROWS))
        self.first_op.ifmaps_file_params = ifmap_tile.file_params
        # Since Concat is executed using MatMul internally,
        # we need some parameters associated with MatMul such as
        # weight shape and weights_file_params. So we set them here
        # instead of calling populate_conv_params due to incompatibility.
        def set_weight_params_for_concat_knode(knode
                                               ,mfilter
                                               ,ifmap_file_param):
            knode.R = 1
            knode.S = 1
            knode.N, knode.C, knode.H, knode.W =\
                    ifmap_file_param.get_nchw_shape()
            #layer_info = knode.data
            weights_shape_dims = ShapeDims("CRSM", (128, 1, 1, 64))
            weights_file = mfilter.file_name
            #if (knode.weights_file_params == None):
            knode.weights_file_params = FileParams(
                file_name       = weights_file, 
                file_dims       = weights_shape_dims, 
                data_type       = knode.data_type, 
                op_params       = knode, 
                args            = knode.parent.args, 
                contain_weights = True)
            knode.weights_file_params.layer_name = knode.data['layer_name']
            knode.weights_file_params.load_file()
            knode.weights_file_params.consumers.append(knode)
            knode.weights_file_params_concat.append(knode.weights_file_params)
            self.map_files_gen_during_exec(tpb, knode.weights_file_params)
            self.live_mapped_file_params.append(knode.weights_file_params)
            self.print_SB_addr(knode.weights_file_params)
            return
        pewave = PEWave(ifmap_tile, c_start_idx, 0, 0)
        pewave.ifmap_channel_count = PEArray.NUM_ROWS
        ofmap_tile_channel_stop = ofmap_tile.channel_stop
        ofmap_tile_channel_count = ofmap_tile.channel_count
        ofmap_tile_channel_start = ofmap_tile.channel_start
        ofmap_pewave = PEWave(ofmap_tile, 0, 0, 0)
        ofmap_pewave.tile.channel_start = 0
        ofmap_pewave.tile.channel_stop = PEArray.NUM_COLS
        ofmap_pewave.tile.channel_count = PEArray.NUM_COLS
        if (first == True):
          print("weight is generated")
          set_weight_params_for_concat_knode(self.first_op\
                                             , mfilter, ifmap_file_param)
          self.concat_dram_weights_file_params[mfilter] =\
            self.first_op.weights_file_params
        else:
          self.first_op.weights_file_params =\
            self.concat_dram_weights_file_params[mfilter]
        packed_ifmap =\
                self.first_op.pack_wave_ifmaps(ifmap_data,pewave,pewave,1,False)
        weight = np.load(mfilter.file_name)
        pewave_for_weights = PEWave(ifmap_tile, 0, 0, 0)
        pewave_for_weights.ifmap_channel_count = PEArray.NUM_ROWS
        pewave_for_weights.ifmap_channel_start = 0
        pewave_for_weights.ifmap_channel_stop = PEArray.NUM_ROWS
        pewave_for_weights.ifmap_channel_end = PEArray.NUM_ROWS - 1
        packed_weights =\
                self.first_op.pack_wave_conv_weights(weight, pewave_for_weights\
                                                     , ofmap_pewave, 1)
        # FIXME : psum_bank needs to be specificed correctly
        tpb.pearray.wave_fp16_mm(packed_ifmap\
                                 , packed_weights\
                                 , self.first_op.psum_bank_dst
                                 , start_tensor_calc == False)
        next_ofmap_start = ofmap_start\
                + (ifmap_channel_range[1] - ifmap_channel_range[0] + 1)
        ofmap_tile.channel_stop = ofmap_tile_channel_stop
        ofmap_tile.channel_count = ofmap_tile_channel_count
        ofmap_tile.channel_start = ofmap_tile_channel_start
        return next_ofmap_start

    # Execute conv and other operations in list: for each op, load parameters and perform op with input
    def execute_conv_ops(self, tpb, batch_item):
        inputs = self.first_op.ifmaps_file_params.dram_data
        weights = self.conv_op.weights_file_params.dram_data

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
                        tile_id = (n_id, m_id, h_id, w_id, self.conv_op.n, self.conv_op.m, self.conv_op.h, self.conv_op.w)
                        ifmap_tile = Tile(tile_id, self.conv_op.ifmaps_file_params, self.conv_op.Tn, is_pe_input=True,
                                            stridedslice_chan_offset = self.conv_op.stridedslice_chan_offset)
                        ofmap_tile = Tile(tile_id, self.last_op.ofmaps_file_params, self.last_op.Tn, is_pe_input=False)
                        self.conv_op.compute_ifmap_ofmap_tile_info(ifmap_tile, ofmap_tile, self.conv_op.is_conv_transpose)
                        # loops for constructing a tile
                        for c_id in range(self.conv_op.c):
                            r_id = 0
                            s_id = 0
                            remaining_filter_elems = self.conv_op.RS
                            while r_id < self.conv_op.weights_file_params.file_dims.R:
                                while s_id < self.conv_op.weights_file_params.file_dims.S:
                                    ifmap_pewave = PEWave(ifmap_tile, c_id, r_id, s_id)
                                    ofmap_pewave = PEWave(ofmap_tile, c_id, r_id, s_id)
                                    # execute PEArray matrix multiply, and add to PSUM after first wave
                                    repl_multiple_per_wave = 1
                                    if self.conv_op.repl_multiple_of_C > 1:
                                        repl_multiple_per_wave = min(remaining_filter_elems, self.conv_op.repl_multiple_of_C)
                                        remaining_filter_elems -= self.conv_op.repl_multiple_of_C
                                    if (self.execute_matmul_waveop(tpb, ifmap_pewave, ofmap_pewave, inputs, weights, psum_add, repl_multiple_per_wave)):
                                        psum_add = True
                                    s_id += self.conv_op.repl_multiple_of_C
                                r_id += s_id//self.conv_op.S
                                s_id = s_id%self.conv_op.S
                        # tile is done                                   
                        # extract PSUM data
                        psum_bank_src = self.conv_op.get_psum_bank()
                        #x = DBG_DUMP_PSUM_COL("PSUM after PEArray: ", psum_temp, 0)
                        # go through the remaining operations, using ofmap_tile as ifmap_tile (TODO: compute new shapes per operation)
                        psum_temp = self.execute_postconv_tile_ops(tpb, ofmap_tile, ofmap_tile, psum_bank_src)
                        #x = DBG_DUMP_PSUM_COL("PSUM after PEArray: ", psum_temp, 0)
                        # if operation is the last one, dump current result into a portion of final result
                        output_params_op = self.conv_op
                        if (self.has_pool):
                            output_params_op = self.pool_op
                        if self.conv_op.is_conv_transpose:
                            ofmap_tile.set_tile_data_in_file(psum_temp)
                        else:
                            # TODO: to simplify fix regular convolution so that we use cropped tile (like conv transpose)
                            # instead of padded tile
                            ofmap_tile.set_padded_tile_data_in_file(psum_temp)
                        waveop = tpb.waveop_stream.last_psum_waveop[psum_bank_src]
                        self.mark_producers_for_subtile_region(tpb, ofmap_tile.make_pewave(), waveop)

                        # Advance to new bank (ping-pong between 0 and 1) for PEArray, while the old bank is being processed by other engines
                        self.conv_op.set_psum_bank((self.conv_op.get_psum_bank()+1)%4)
                        tpb.pearray.last_psum_bank_used = self.conv_op.get_psum_bank()
                        psum_add = False

