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
from me_utils import align_addr_NB
from me_utils import Coord
from me_utils import Dim2D
from me_utils import Rect
from me_utils import ShapeDims
from me_utils import FileParams
from me_utils import EngineEnum
from me_utils import extract_predecessors
from me_utils import attach_predecessors
from me_pool import Pool
from me_concat import Concat
from me_quantization_utils import mem_pattern_uint8_perf, reorder_waveops_uint8_perf

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
    def __init__(self, tile, c_id, r_id, s_id):
        self.format = "nmhwcrs"
        self.tile = tile
        self.c_id, self.r_id, self.s_id = c_id, r_id, s_id
        self.id_array = self.tile.id_array  + [self.c_id, self.r_id, self.s_id]
        self.id_string = self.tile.id_string + "_c%d_r%d_s%d"%(self.c_id, self.r_id, self.s_id)
        self.ifmap_channel_start = self.c_id * PEArray.NUM_ROWS
        self.ifmap_channel_stop = min(self.tile.file_params.file_dims.C,
                                      self.ifmap_channel_start + PEArray.NUM_ROWS)
        self.ifmap_channel_count = self.ifmap_channel_stop - self.ifmap_channel_start
        self.subtile_psum_offset = 0
        # Default rectangle info, works if subtile=tile; 
        # compute_ifmap_ofmap_pewave_info will overwrite these with correct values
        self.subtile_rect = tile.tile_rect
        (self.lower_addr, self.upper_addr) = self.get_subtile_file_addrs()
        (self.tile.lower_addr, self.tile.upper_addr) = (self.lower_addr, self.upper_addr) 
        self.lower_to_upper_len_bytes = []
        self.tile.lower_to_upper_len_bytes = []
        for i in range(len(self.lower_addr)):
            size = self.upper_addr[i] - self.lower_addr[i] + self.tile.file_params.item_sz
            self.lower_to_upper_len_bytes.append(size)
            self.tile.lower_to_upper_len_bytes.append(size)

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

    def set_waveop_pattern(self, file_mapper, waveop, prefix, psum_bank_id, sb_address, start_at_mid_part, sb_dtype=None):
        prefix2 = prefix.replace("src","in").replace("dst","out")
        x_step = 1
        x_num = self.subtile_rect.dim2d.x
        y_num = self.subtile_rect.dim2d.y
        z_num = 1 if waveop['waveop_type'] == 'Pool' else self.tile.Tn 
        batch_item = self.tile.n_id * self.tile.Tn
        if psum_bank_id >= 0:
            dtype  = "float32"
            # note (HC): if state buffer dtype is specified, we select
            # the "smallest" psum dtype in the case of fused op
            if sb_dtype is None:
                pass
            elif sb_dtype in {"float16", "bfloat16", "float32"}:
                dtype = "float32"
            elif sb_dtype in {"uint8", "uint16", "int32"}:
                dtype = "int32"
            elif sb_dtype in {"int64"}:
                dtype = "int64"
            else:
                raise NotImplementedError('sb_dtype %s is not supported' % sb_dtype)
            y_step = self.subtile_rect.dim2d.x
            z_step = y_step * y_num if z_num > 1 else 1
            waveop[prefix + "_psum_bank_id"]      = psum_bank_id
            waveop[prefix + "_psum_bank_offset"]  = self.subtile_psum_offset
        else:   # negative psum_bank_id indicates SB
            dtype  = self.tile.file_params.data_type if sb_dtype is None else sb_dtype
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
        self.lower_to_upper_len_bytes = []
        for i in range(len(self.lower_addr)):
            self.lower_to_upper_len_bytes.append(self.upper_addr[i] - self.lower_addr[i] + self.tile.file_params.item_sz)
        assert(tile.is_pe_input == False)

""" FMAP tile object
"""
class Tile:

    def __init__(self, tile_id, file_params, Tn, is_pe_input=False):
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
        self.channel_start = self.m_id * PEArray.NUM_COLS
        self.channel_stop = min(self.file_params.file_dims.C, self.channel_start + PEArray.NUM_COLS)
        self.channel_count = self.channel_stop - self.channel_start
        self.start_at_mid_part = (self.m_id % 2) == 1
        self.n_start = self.n_id * self.Tn
        self.n_stop  = min(self.file_params.file_dims.N, self.n_start + self.Tn)
        #FIXME
        self.mepoolspec = None

    def copy(self):
        tile_id = (self.n_id, self.m_id, self.h_id, self.w_id, self.n, self.m, self.h, self.w)
        new_tile = Tile(tile_id, 
                        self.file_params, 
                        self.Tn, 
                        is_pe_input=self.is_pe_input)
        new_tile.tile_rect = self.tile_rect
        new_tile.padded_tile_rect = self.padded_tile_rect
        new_tile.mepoolspec = self.mepoolspec
        return new_tile

    def make_pewave(self):
        return PEWave(self, 0, 0, 0)

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
    act_ops_regex = "Relu|Softplus|Sigmoid|Tanh|^Exp$|Identity|Lrelu|Prelu|Sqrt"
    bias_ops_regex = "BiasAdd"
    pool_ops_regex = ".*Pool|Sub|Add|Multiply|ResAdd|Maximum|Minimum|QuantizeV2|Dequantize"
    identity_ops_regex = "Reshape|Squeeze|ExpandDims"
    next_is_fusable_regex = bias_ops_regex + "|" + act_ops_regex + "|" + pool_ops_regex # + "|" + identity_ops_regex
    next_is_fusable = {
            'Conv'     : next_is_fusable_regex,
            'ConvTranspose': next_is_fusable_regex,
            'QuantizedConv': next_is_fusable_regex,
            'Dequantize': next_is_fusable_regex,
            'MatMul'   : next_is_fusable_regex,
            'BiasAdd'  : next_is_fusable_regex,
            'Add'      : next_is_fusable_regex,
            'Sub'      : next_is_fusable_regex,
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

    def translate_isa_op_name(self, op_type):
        new_type = op_type
        if op_type == "resadd":
            new_type = "add"
        elif op_type == "sub":
            new_type = "subtract"
        elif op_type == "maximum":
            new_type = "max"
        elif op_type == "minimum":
            new_type = "min"
        elif op_type == "multiply":
            new_type = "mult"
        return new_type            

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
        #self.concat_dram_weights_waveops = []
        self.concat_dram_weights_waveops = dict()
        self.concat_dram_weights_file_params = dict()

    def set_live_mapped_file_params(self, live_mapped_file_params):
        self.live_mapped_file_params = live_mapped_file_params 

    # Add operation to list of fused operations.
    # Returns True if successful; False if cannot add (i.e. Pool cannot be fused)
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
        elif (op.data['layer_type'] == 'Conv' or op.data['layer_type'] == 'ConvTranspose' or op.data['layer_type'] == 'MatMul' or op.data['layer_type'] == 'Softmax2' or op.data['layer_type'] == 'QuantizedConv'):
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
                elif len(self) == 0 and op.data['layer_type'] == 'Sub':
                    # If Sub is not in the middle of fused ops, its residue_index must be one.
                    # since ME accepts only sub(ifmap, residue_index).
                    # See Kaena-1097
                    self.join_op.residue_index = 1                    
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
        bias_region_sz = self.last_op.parent.combined_bias_file_params.mapped_params.region_sz
        bias_region_sz = tpb.statebuffer.batcher.sb_bias_sz[0]
        if file_start_addr < bias_region_sz:
            file_start_addr = bias_region_sz
        if file_start_addr + file_sz >= tpb.statebuffer.file_mapper.sb_partition_sz:
            file_start_addr = bias_region_sz
        # Check overlap with IFMAP
        #for i in self.first_op.ifmaps_file_params_concat:
        #    file_start_addr = check_overlap(i, file_start_addr)
        # Check overlap with OFMAP
        ofmap_file_params = self.last_op.ofmaps_file_params
        #assert(ofmap_file_params.mapped_params != None)
        #file_start_addr = check_overlap(ofmap_file_params, file_start_addr)
        ifmap_ofmap_file_params = self.first_op.ifmaps_file_params_concat + [ofmap_file_params]
        (file_start_addr, wrap_around, file_sz) = tpb.statebuffer.file_mapper.move_addr_to_first_free_section(
                                  st_addr                 = file_start_addr 
                                , region_sz               = file_sz
                                , min_region_start        = bias_region_sz
                                , live_mapped_file_params = self.live_mapped_file_params + ifmap_ofmap_file_params)
        tpb.statebuffer.file_mapper.map_file(\
                                             file_params\
                                             , file_start_addr\
                                             , wrap_around=wrap_around\
                                             , region_sz=file_sz\
                                            )
        tpb.statebuffer.next_nonbias_file_start = file_start_addr\
                + align_addr_NB(file_params.tot_partition_usage_sz_padded, 64)
        return

    def print_SB_addr (self, file_params):
        print ("file_name = %s"%file_params.file_name)
        print ("\tstart_addr = %d"%file_params.mapped_params.start_addr)
        print ("\tend_addr = %d"%file_params.mapped_params.end_addr)

    """Map tensors to SB
        args:
            tpb: TPB object (TODO: pass only needed objects: statebuffer batcher/file_mapper)
            batch_item: batch item number of the start of the partial batch group  
            live_mapped_file_params: current list of SB mapped tensor files that should remain (since they are still being used, and we don't want to evict unless we have to) 
    """
    def map_files(self, 
                tpb, 
                batch_item, 
                live_mapped_file_params):
        map_file = tpb.statebuffer.file_mapper.map_file

        # select SB region sizing index (only relevant for ResNet50 but we also use for BiasAdd)
        # This maybe overwritten later if next batch count doubles (pairup)
        sb_size_set_index = tpb.statebuffer.batcher.sb_size_set_index[(self.current_batch_count, self.partial_batch_pre_pairup)]

        # Bias values are now folded into CNcHW, combined with constants and loaded in chunks (kaena-356)
        # They are premapped by call to kgraph.map_constants_file in me_main

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
        ofmaps_region_sz = single_ofmap_sz * self.last_op.N

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
                                                + self.first_op.slice_offset.x * self.first_op.ifmaps_file_params.item_sz \
                                                + self.first_op.slice_offset.y * self.first_op.ifmaps_file_params.item_sz * self.first_op.W
                else:    
                    ofmaps_region_sz = tpb.statebuffer.batcher.sb_scratch_sz[sb_size_set_index]
                    ofmaps_region_start_addr = tpb.statebuffer.file_mapper.sb_partition_sz - ofmaps_region_sz

            # If OFMAP region overlaps IFMAP region, and numober of channels > 64 or stride/filter-size > 1, offset it (to lower address) by OFMAP * Tn 
            # (stride is only relevant to Conv/Pool, and filter-size is only relevant to Conv)
            if ofmaps_region_start_addr >= ifmaps_region_start_addr \
                    and ofmaps_region_start_addr < ifmaps_region_start_addr + ifmaps_region_sz \
                    and not self.first_op.is_nop:
                if self.first_op.data["layer_type"] == "TransposeVector":                                            
                    ofmaps_region_start_addr = ifmaps_region_start_addr + ifmaps_region_sz
                    if ofmaps_region_start_addr + ofmaps_region_sz >= tpb.statebuffer.file_mapper.sb_partition_sz:
                        ofmaps_region_start_addr = tpb.statebuffer.batcher.sb_bias_sz[sb_size_set_index]
                elif (ofmaps_file_params.file_dims.C > 64) \
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
                # stagger weight chunks to relax dependency on the first chunk
                gap_at_region_end = (weights_region_start_addr + weights_region_sz) - (weights_file_start_addr + weights_file_sz)
                chunk_id_start_offset = -(gap_at_region_end // weights_file_params.chunk_sz_padded)
                if chunk_id_start_offset > 0:
                    chunk_id_start_offset = 0
                elif chunk_id_start_offset < 0 \
                        and weights_file_start_addr == weights_region_start_addr \
                        and weights_region_sz >= weights_file_sz:
                    weights_file_start_addr += weights_region_sz - weights_file_sz
                # map file to region                
                map_file(weights_file_params, weights_file_start_addr, wrap_around=False, region_sz=weights_file_sz, chunk_id_start_offset=chunk_id_start_offset)
                # obtain the adjusted region size
                weights_region_sz = weights_file_params.mapped_params.region_sz
                # also in case that file is already mapped, keep the mapped values
                weights_file_start_addr = weights_file_params.mapped_params.start_addr
        # Simple networks: use simple mapping                
        else:
            bias_region_sz = self.first_op.parent.combined_bias_file_params.mapped_params.region_sz

            # Get start for IFMAPs
            start_addr = tpb.statebuffer.next_nonbias_file_start
            if start_addr < bias_region_sz:
                start_addr = bias_region_sz
            # IFMAPs regions                    
            # Input/residue uses upper portion of the shared space
            is_graph_input = self.first_op.is_input or ifmaps_file_params.produce_op.data['layer_type'] == 'ConstLoad'
            is_pool_type = self.first_op.data["layer_type"] == "AvgPool" or self.first_op.data["layer_type"] == "MaxPool"
            for ifp in self.first_op.ifmaps_file_params_concat:
                if is_graph_input and ifp.mapped_params is None:
                    if start_addr + ifmaps_region_sz >= tpb.statebuffer.file_mapper.sb_partition_sz:                    
                        start_addr = bias_region_sz
                    ifmaps_region_start_addr = start_addr
                    ifmaps_region_sz = ifp.tot_partition_usage_sz_padded
                    (ifmaps_region_start_addr, wrap_around, ifmaps_region_sz) \
                            = tpb.statebuffer.file_mapper.move_addr_to_first_free_section(
                                          st_addr                 = ifmaps_region_start_addr
                                        , region_sz               = ifmaps_region_sz
                                        , min_region_start        = bias_region_sz
                                        , live_mapped_file_params = live_mapped_file_params
                                        , allow_wrap              = not is_pool_type
                                        , max_num_chunks          = 4
                                        , chunk_sz                = ifp.chunk_sz_padded)
                    map_file(ifp, ifmaps_region_start_addr, wrap_around=wrap_around, region_sz=ifmaps_region_sz)
                    live_mapped_file_params.append(ifp)
                    # obtain the adjusted region size
                    start_addr       = ifmaps_region_start_addr + align_addr_NB(ifmaps_region_sz, 64)
                # For overlap checking later, only use region info for the first IFMAP (do we still need this?)
                if ifp == self.first_op.ifmaps_file_params_concat[0]: 
                    ifmaps_region_start_addr = ifp.mapped_params.start_addr
                    ifmaps_region_sz  = ifp.mapped_params.region_sz
                    single_ifmap_start = ifmaps_region_start_addr

            # Weights region, align to 8B
            if self.has_conv:
                if weights_file_params.mapped_params is None:
                    wrap_around = False
                    weights_file_start_addr = start_addr
                    (weights_file_start_addr, wrap_around, weights_file_sz) \
                            = tpb.statebuffer.file_mapper.move_addr_to_first_free_section(
                                                      st_addr                 = weights_file_start_addr 
                                                    , region_sz               = weights_file_sz
                                                    , min_region_start        = bias_region_sz
                                                    , live_mapped_file_params = live_mapped_file_params
                                                    , allow_wrap              = True
                                                    , chunk_sz                = weights_file_params.chunk_sz_padded)
                    map_file(weights_file_params, weights_file_start_addr, wrap_around=wrap_around, region_sz=weights_file_sz)
                    live_mapped_file_params.append(weights_file_params)
                    # obtain the adjusted region size
                    weights_region_sz = weights_file_params.mapped_params.region_sz
                    start_addr = weights_file_start_addr + align_addr_NB(weights_region_sz, 64)
                else:                    
                    # also in case that file is already mapped, keep the mapped values
                    weights_file_start_addr = weights_file_params.mapped_params.start_addr
                    # For dynamic weights, weights_file_params points to the ofmap
                    # of a predecessor of this matmul. It should be marked as live-in.
                    # JeffH: already taken care off by the above IFMAP loop
                    #if not (weights_file_params in live_mapped_file_params):
                    #    live_mapped_file_params.append(weights_file_params)
                weights_region_start_addr = weights_file_start_addr

            # get residue file params (to check if OFMAP can be mapped to same space as residue)
            # map it if it's not mapped yet (case with two inputs to a join)
            residue_file_params = None
            if self.has_join and not self.join_op.is_concat:
                residue_file_params = self.join_op.prev[self.join_op.residue_index].ofmaps_file_params
                if residue_file_params.mapped_params is None:
                    residue_region_sz = residue_file_params.tot_partition_usage_sz_padded
                    if start_addr + residue_region_sz >= tpb.statebuffer.file_mapper.sb_partition_sz:                    
                        start_addr = bias_region_sz
                    if residue_region_sz > (tpb.statebuffer.file_mapper.sb_partition_sz/4):
                        residue_region_sz = 4 * residue_file_params.chunk_sz_padded * residue_file_params.fmap_channels_folds
                    (start_addr, wrap_around, residue_region_sz) \
                            = tpb.statebuffer.file_mapper.move_addr_to_first_free_section(
                                          st_addr                 = start_addr 
                                        , region_sz               = residue_region_sz
                                        , min_region_start        = bias_region_sz
                                        , live_mapped_file_params = live_mapped_file_params)
                    map_file(residue_file_params, start_addr, wrap_around=True, region_sz=residue_region_sz)
                    live_mapped_file_params.append(residue_file_params)
                    residue_region_sz = residue_file_params.mapped_params.region_sz
                    start_addr += align_addr_NB(residue_region_sz, 64)

            # OFMAPs region
            if ofmaps_file_params.mapped_params is None:
                wrap_around = False
                if self.first_op.is_nop:
                    if self.first_op.ifmaps_file_params.file_dims.format_str == 'HNWC':
                        assert(self.first_op.slice_offset.x == 0), "Unstacking HNWC: only slicing in H dimension allowed"
                        ofmaps_region_start_addr = ifmaps_region_start_addr \
                                                    + self.first_op.slice_offset.y \
                                                        * self.first_op.ofmaps_file_params.tot_partition_usage_sz_padded
                    else:
                        ofmaps_region_start_addr = ifmaps_region_start_addr \
                                                    + self.first_op.slice_offset.x * self.first_op.ifmaps_file_params.item_sz \
                                                    + self.first_op.slice_offset.y * self.first_op.ifmaps_file_params.item_sz * self.first_op.W
                    if self.first_op.data["layer_type"] == "StridedSlice":
                        ofmaps_region_start_addr += ceildiv(self.first_op.data['channel_slice'][0], PEArray.NUM_ROWS) \
                                                    * ifmaps_file_params.fmap_data_len_padded
                # ReviewMe: Reusing residue memory region doesn't work in the folllowing example.
                #
                # a = ..
                # b = multiply(a, a)  <-- b should not reuse a's memory address
                # ...
                #   = conv(b)
                # ..
                # c = multiply(a)
                elif self.has_join and len(self.last_op.ofmaps_file_params.writers_of_shared_fmap) > 1 \
                        and len(residue_file_params.readers_of_shared_fmap) == 1 \
                        and not self.join_op.is_input \
                        and (ofmaps_file_params.chunk_sz_padded == ifmaps_file_params.chunk_sz_padded) \
                        and (residue_file_params != ifmaps_file_params):
                    ofmaps_region_start_addr = residue_file_params.mapped_params.start_addr
                else:    
                    ofmaps_region_start_addr = start_addr
                    (ofmaps_region_start_addr, wrap_around, ofmaps_region_sz) \
                            = tpb.statebuffer.file_mapper.move_addr_to_first_free_section(
                                  st_addr                 = ofmaps_region_start_addr
                                , region_sz               = ofmaps_region_sz
                                , min_region_start        = bias_region_sz
                                , live_mapped_file_params = live_mapped_file_params
                                , allow_wrap              = False
                                , chunk_sz                = ofmaps_file_params.chunk_sz_padded)
                map_file(ofmaps_file_params\
                         , ofmaps_region_start_addr\
                         , wrap_around=wrap_around, region_sz=ofmaps_region_sz)
                live_mapped_file_params.append(ofmaps_file_params)
                # obtain the adjusted region size
                ofmaps_region_sz = ofmaps_file_params.mapped_params.region_sz
                single_ofmap_start = ofmaps_region_start_addr 
                start_addr = single_ofmap_start + align_addr_NB(ofmaps_region_sz, 64)
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
            tpb.statebuffer.next_nonbias_file_start = ofmaps_region_start_addr + align_addr_NB(ofmaps_region_sz, 64)

        # Trace printout
        if (self.args.debug > 2 and not tpb.statebuffer.printed_map_trace_header): 
            print("SB MAP TRACE, fused op, fused op ID, batch elem, Tn, current_batch_count, next_batch_count, partial_batch_pre_pairup, partial_batch_pairup, residue_in_scratch,, \
 weights file end_addr, weights region end_addr, weights region start_addr, weights file start_addr,,\
 ifmap single end_addr, ifmaps region end_addr, ifmaps region start_addr, ifmap single start_addr,,\
 ofmap single end_addr, ofmaps region end_addr, ofmaps region start_addr, ofmap single start_addr,, ifmap file, ofmap file")                    
            tpb.statebuffer.printed_map_trace_header = True
        if (self.args.debug > 2): print("SB MAP TRACE, %s, %d, %d, %d, %d, %d, %d, %d, %d,, %d, %d, %d, %d,, %d, %d, %d, %d,, %d, %d, %d, %d,, %s, %s"%(
                                self.last_op.data['layer_name'], 
                                self.fused_op_id, 
                                batch_item,
                                self.first_op.Tn,
                                self.current_batch_count,
                                self.next_batch_count,
                                self.partial_batch_pre_pairup,
                                self.partial_batch_pairup,
                                self.residue_in_scratch,
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
            if overlap_some_percent != overlap_100_percent:
                raise RuntimeError("There's overlap between IFMAP region (start %s size %s) and OFMAP region (start %s size %s). (This error should not raise if OFMAP overlaps IFMAP region 100 percent for SB space reuse purpose)"%(single_ifmap_start, min(single_ifmap_sz, ifmaps_region_sz), single_ofmap_start, single_ofmap_sz))

    def mark_ifmaps_are_consumed(self, live_mapped_file_params):
        # Note that each of ifmaps of current knode is ofmap of its predecessor.
        # That's why ofmap is retrieved from the predecessor knode.
        if (self.has_conv):
            weight = self.conv_op.weights_file_params
            weight.mapped_params.mark_consumed(self.first_op)
            if weight.mapped_params.is_consumed_by_all_readers():
                live_mapped_file_params.remove(weight)
        for i in self:
            if len(i.weights_file_params_concat) > 0:
                for w in i.weights_file_params_concat:
                    w.mapped_params.mark_consumed(i)
                    if w.mapped_params.is_consumed_by_all_readers():
                        live_mapped_file_params.remove(w)
            for pred in i.prev:
                ofmap = pred.ofmaps_file_params
                if ofmap.mapped_params is not None:
                    ofmap.mapped_params.mark_consumed(i)
                if ofmap in live_mapped_file_params and\
                        ofmap.mapped_params.is_consumed_by_all_readers():
                    live_mapped_file_params.remove(ofmap)
        return

    def execute(self, tpb, batch_item):
        assert (batch_item >= 0)
        assert (batch_item < self.first_op.N)
        # Check conv fused op
        first_op_type = self.first_op.data['layer_type']
        if (first_op_type == "Conv" or first_op_type == "ConvTranspose" or first_op_type == "MatMul"):
            self.execute_conv_ops(tpb, batch_item)
        elif first_op_type == "QuantizedConv":
            self.execute_conv_ops(tpb, batch_item)
        elif first_op_type in {"AvgPool", "MaxPool", "ClipByValue", "QuantizeV2", "Dequantize"}:
            self.execute_unfused_pool_op(tpb, batch_item)
        elif (first_op_type == "Softmax2"):
            self.execute_softmax2(tpb, batch_item)
        elif (first_op_type == "Multiply" 
                or first_op_type == "Maximum" 
                or first_op_type == "Minimum" 
                or first_op_type == "Add" 
                or first_op_type == "Sub" 
                or first_op_type == "ResAdd"): # TODO: handle the scalar 
            if (len(self.first_op.data['previous_layers']) <= 2):
                results = self.execute_unfused_pool_op(tpb, batch_item)
            else:                
                raise RuntimeError("ERROR: cannot handle more than two inputs for first operation %s, layer %s"%(first_op_type, first_op.data["layer_name"]))
        elif re.search(self.act_ops_regex, first_op_type):
            self.execute_unfused_pool_op(tpb, batch_item)
        elif (first_op_type == "BiasAdd" or first_op_type == "Reciprocal"):
            self.execute_unfused_pool_op(tpb, batch_item)
        elif (first_op_type == "Concat"):
            self.execute_unfused_concat_op(tpb, batch_item)
        elif (first_op_type == "StridedSlice"):
            self.execute_unfused_stridedslice_op(tpb, batch_item)
        elif (first_op_type == "Transpose" or first_op_type == "TransposeVector"):
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
                            expected_ofmaps_extracted = expected_ofmaps[i, :]
                            assert(expected_ofmaps_extracted.flags.c_contiguous == True)
                            diff = ofmaps - expected_ofmaps_extracted
                            if (self.args.debug > 1): print("========= Compare for: " + self.last_op.data['layer_name'])
                            if (self.args.debug > 2): print("\nInput IFMAPS:\n", ifmaps)
                            if (self.args.debug > 1): print("\nComputed OFMAPS:\n", ofmaps)
                            if (self.args.debug > 1): print("\nExpected OFMAPS:\n", expected_ofmaps_extracted)
                            if (self.args.debug > 1): print("\nDiffed   OFMAPS:\n", diff)
                            if (self.args.debug > 1): print("========= End ====================")
                            if (not npu.allclose(ofmaps, expected_ofmaps_extracted, 1/100, 1e-5, verbose=3)):
                                print("\nERROR: layer %s batch item %d computed OFMAPS is not equal to expected OFMAPS!\n"%(self.last_op.data['layer_name'], i))
                                tpb.num_mismatches += 1

            # Save results for network output or we want to save debug intermediate results
            save_layer = False
            if self.args.save_layer_output > 0:
                save_layer = self.fused_op_id < self.args.save_layer_output 
            elif self.args.save_layer_output < 0:
                save_layer = self.fused_op_id >= (len(tpb.fused_ops_list) + self.args.save_layer_output)
            if self.args.save_layer_regex != "":
                save_layer = re.search(self.args.save_layer_regex, self.last_op.data['layer_name'])

            if self.last_op.next == [] or save_layer and not self.no_writer_for_save:
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

    def gen_regload_vector_waveop(self,
            tpb,
            layer_name,
            subtile,
            parallel_mode):
        batch_item = subtile.tile.n_id * subtile.tile.Tn
        for z in range(subtile.tile.Tn):
            file_addr = subtile.lower_addr[z]
            sb_addr = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(subtile.tile.file_params, batch_item + z, file_addr)
            (prev_waveops, dram_ifmaps_waveops, new_reader_morsels) =\
                    self.get_producers_for_subtile_region (
                        tpb = tpb, 
                        fmap_subtile = subtile, 
                        reader_engine = EngineEnum.POOL,
                        subbatch_item = z)
            waveop = {
                  'previous_waveops'        : prev_waveops,
                  'waveop_type'             : 'RegLoad',
                  'waveop_name'             : layer_name + "/RegLoad",
                  'layer_name'              : layer_name,
                  'parallel_mode'           : parallel_mode,
                  'in_dtype'                : subtile.tile.file_params.data_type,
                  'src_is_psum'             : False,
                  'src_x_step'              : 1,
                  'src_x_num'               : 1,
                  'src_y_step'              : 1,
                  'src_y_num'               : 1,
                  'src_z_step'              : 1,
                  'src_z_num'               : 1,
                  'src_sb_address'          : sb_addr,
                  'src_start_at_mid_part'   : (subtile.tile.m_id % 2) == 1,
                  'num_partitions'          : subtile.tile.channel_count,
                }
            if not parallel_mode:
                waveop['num_partitions']  = 1
                waveop['src_x_num']       = subtile.subtile_rect.upper.x - subtile.subtile_rect.lower.x + 1
                waveop['src_start_at_mid_part'] = False
            tpb.waveop_stream.add_linked(waveop, dram_ifmaps_waveops, -1, new_reader_morsels)
        return waveop

    def gen_regstore_vector_waveop(self,
            tpb,
            layer_name,
            subtile,
            parallel_mode):
        batch_item = subtile.tile.n_id * subtile.tile.Tn
        waveop_list = []
        for z in range(subtile.tile.Tn):
            file_addr = subtile.lower_addr[z]
            sb_addr = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(subtile.tile.file_params, batch_item + z, file_addr)
            waveop = {
                  'previous_waveops'        : [],
                  'waveop_type'             : 'RegStore',
                  'waveop_name'             : layer_name + "/RegStore",
                  'layer_name'              : layer_name,
                  'parallel_mode'           : parallel_mode,
                  'out_dtype'               : subtile.tile.file_params.data_type,
                  'dst_is_psum'             : False,
                  'dst_x_step'              : 1,
                  'dst_x_num'               : 1,
                  'dst_y_step'              : 1,
                  'dst_y_num'               : 1,
                  'dst_z_step'              : 1,
                  'dst_z_num'               : 1,
                  'dst_sb_address'          : sb_addr,
                  'dst_start_at_mid_part'   : (subtile.tile.m_id % 2) == 1,
                  'num_partitions'          : subtile.tile.channel_count,
                }
            if not parallel_mode:
                waveop['num_partitions']  = 1
                waveop['dst_x_num']       = subtile.subtile_rect.upper.x - subtile.subtile_rect.lower.x + 1
                waveop['dst_start_at_mid_part'] = False
            tpb.waveop_stream.add_linked(waveop, [], -1, [])
            self.mark_producers_for_subtile_region(
                    tpb          = tpb, 
                    fmap_subtile = subtile,
                    waveop       = waveop,
                    subbatch_item = z)
        return waveop

    # generate MatMul waveop to clear PSUM
    def gen_psum_clr_waveop(self, 
            tpb, 
            ifmap_pewave, 
            ofmap_pewave, 
            psum_bank_id):

        # obtain input/output data types
        in_dtype = self.first_op.ifmaps_file_params.data_type
        out_dtype = self.first_op.ofmaps_file_params.data_type
        # PSUM has more precision bits (also, allow for int32 to remain int32)
        # TODO: handle uint16 -> int64
        if out_dtype in {'float16', 'bfloat16', 'float32'}:
            out_dtype = 'float32'

        # Need to clear the entire full tile, not just one subtile
        fmap_x_num = ofmap_pewave.tile.tile_rect.dim2d.x
        fmap_y_num = ofmap_pewave.tile.tile_rect.dim2d.y

        # grab dependencies for zeros
        (writers, _, dram_bias_waveops, zeros_reader_morsels) = tpb.statebuffer.file_mapper.read_file_data_region(
                                        tpb.waveop_stream.waveop_count,
                                        tpb.waveop_stream,
                                        self.first_op.parent.combined_bias_file_params,
                                        0,  # batch_item is not needed for bias
                                        0,  # zeros are located in first fold of bias-constants file
                                        self.first_op.parent.combined_bias_file_params.item_sz,
                                        EngineEnum.PEARRAY,
                                        )
        prev_wr_waveops = extract_predecessors(
                list_of_accessors_wr = writers, 
                list_of_accessors_rd = [[]],
                waveop_list         = tpb.waveop_stream)

        zeros_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.first_op.parent.combined_bias_file_params, 0, 0)
        waveop_name = self.first_op.data['layer_name']+"/MatMul_"+ofmap_pewave.id_string+"__psum_clr"

        instr = {
              'is_dynamic_weights'      : False,
              'previous_waveops'        : [],
              'waveop_type'             : 'MatMul',
              'waveop_name'             : waveop_name,
              'layer_name'              : self.first_op.data['layer_name'],
              'weights_sb_address'      : zeros_sb_address,
              'in_dtype'                : in_dtype,
              'out_dtype'               : out_dtype,
              'start_tensor_calc'       : True,
              'stop_tensor_calc'        : False,
              'src_is_psum'             : False,
              'src_sb_address'          : zeros_sb_address,
              'src_start_at_mid_part'   : False,
              'src_x_step'              : 0,
              'src_x_num'               : fmap_x_num * fmap_y_num * self.first_op.Tn,
              'src_y_step'              : 0,
              'src_y_num'               : 1,
              'src_z_step'              : 0,
              'src_z_num'               : 1,
              'num_row_partitions'      : ifmap_pewave.ifmap_channel_count,
              'dst_is_psum'             : True,
              'dst_psum_bank_id'        : psum_bank_id,
              'dst_psum_bank_offset'    : 0,
              'dst_x_step'              : 1,
              'dst_x_num'               : fmap_x_num,
              'dst_y_step'              : fmap_x_num,
              'dst_y_num'               : fmap_y_num,
              'dst_z_step'              : self.first_op.ofmap_full_tile_sz,
              'dst_z_num'               : self.first_op.Tn,
              'num_column_partitions'   : ofmap_pewave.tile.channel_count,
              'ifmap_replication_resolution' : 0,
              'ifmap_replication_num_rows'   : 0,
              'ifmap_replication_shift_amnt' : 0,
              'quant_offset_ifmaps'     : 0,
              'quant_offset_weights'    : 0,
              'pe_perf_opt_mode'        : 'none',
        }

        attach_predecessors(instr, prev_wr_waveops)
        tpb.waveop_stream.add_linked(instr, dram_bias_waveops, psum_bank_id, zeros_reader_morsels)
        if (self.args.debug > 2): print("DBG %s: Generating MatMul waveop %s to clear PSUM, weights_sb_address %d, fmap_sb_address %d"%(self.first_op.data['layer_name'], waveop_name, zeros_sb_address, zeros_sb_address))
        return [instr]

    # generate MatMul waveop and add it to waveop stream
    def gen_matmul_waveop(self, 
            tpb, 
            ifmap_pewave, 
            ofmap_pewave, 
            psum_add, 
            psum_bank_id, 
            dram_weights_waveops, 
            load_weights_into_pearray=True, 
            repl_multiple_of_C=1, 
            conv_transpose=False):
        batch_item = ofmap_pewave.tile.n_id * ofmap_pewave.tile.Tn
        # Note (HC): cannot use ofmap_pewave to determine out_dtype because
        # in quantized cases ofmap_pewave.tile.file_params.data_type == float32
        # Assumes this function is always called due to the first op self[0]
        in_dtype = self.first_op.ifmaps_file_params.data_type
        assert in_dtype in {'float16', 'bfloat16', 'float32', 'uint8', 'uint16'}
        out_dtype = self.first_op.ofmaps_file_params.data_type
        assert out_dtype in {'float16', 'bfloat16', 'float32', 'int32'}
        if 'int32' == out_dtype: # quantized conv; needs offsets
            quant_offset_ifmaps = int(self.first_op.data['zero_point_input'])
            quant_offset_weights = int(self.first_op.data['zero_point_filter'])
        elif out_dtype in {'float16', 'bfloat16', 'float32'}:
            out_dtype = 'float32'
            quant_offset_ifmaps = 0
            quant_offset_weights = 0
        else:
            raise NotImplementedError('out_dtype %s is not supported' % out_dtype)

        weights_sb_address = -1
        has_dynamic_weights = False
        if load_weights_into_pearray:
            # find the weights offset within atom; -1 means don't load new weights
            weights_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(
                                        self.first_op.weights_file_params, 
                                        0, 
                                        self.first_op.weight_wave_lower_addr)
            has_dynamic_weights =  self.first_op.weights_file_params.is_dynamic_weights

        # If wave crosses atom boundaries, break it into multiple waves
        # The following assumes noodle tile (width is equal to FMAP width)
        current_chunk_id = -10000   # force the first break at start address
        current_atom_id = -10000
        break_at_y = []
        break_addr = []
        subtile_rect_dim2d = ifmap_pewave.subtile_rect.dim2d if conv_transpose else ofmap_pewave.subtile_rect.dim2d
        addr_step_y = self.first_op.W * self.first_op.stride.y * self.first_op.ifmaps_file_params.item_sz
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
            dst_psum_bank_offset = break_at_y[i] * dst_y_step
            dst_psum_bank_offset += ofmap_pewave.subtile_psum_offset
            assert(dst_psum_bank_offset < PEArray.MAX_WAVE_SIZE)
            fmap_sb_address = break_addr[i]
            if ifmap_replication_resolution > 1:
                assert((fmap_sb_address%8) == 0), "Kaena-593: IFMAP start address must by 8B aligned for replication to work"

            # get dram waveops (weights) for the first piece of broken matmul
            dram_waveop_names = []
            if i==0:
                for j in dram_weights_waveops:
                    #print("tamke:gen_matmuls:dram_weights_waveops = %s"%j["waveop_name"])
                    dram_waveop_names.append(j["waveop_name"])
            else:
                # reuse weights loaded with first piece of broken matmul
                weights_sb_address = -1

            for z in range(self.first_op.Tn):                    
                lower_file_address = ifmap_pewave.lower_addr[z] + break_at_y[i] * addr_step_y
                upper_file_address = min(ifmap_pewave.lower_addr[z] + next_break * addr_step_y - self.first_op.ifmaps_file_params.item_sz, ifmap_pewave.upper_addr[z])
                list_of_names = tpb.statebuffer.file_mapper.get_dram_waveop_names(self.first_op.ifmaps_file_params, batch_item + z, lower_file_address, upper_file_address)
                for name in list_of_names:
                    if name not in dram_waveop_names:
                        dram_waveop_names.append(name)

            waveop_name = self.first_op.data['layer_name']+"/MatMul_"+ofmap_pewave.id_string+"__"+str(i)
            if (self.args.debug > 2): print("DBG %s: MatMul wave %s subwave %d weights_sb_address %d, fmap_sb_address %d, fmap_y_num %d"%(self.first_op.data['layer_name'], waveop_name, i, weights_sb_address, fmap_sb_address, fmap_y_num))

            instr = {
                  'is_dynamic_weights'      : has_dynamic_weights,
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
                  'dst_psum_bank_id'        : psum_bank_id,
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
                  'quant_offset_ifmaps'     : quant_offset_ifmaps,
                  'quant_offset_weights'    : quant_offset_weights,
                  'pe_perf_opt_mode'        : 'none',
            }
            # uint8 performance mode adjustments
            if self.args.uint8_performance_mode and in_dtype == 'uint8':
                instr = mem_pattern_uint8_perf(self.conv_op, instr, ifmap_pewave.tile.tile_rect.dim2d)
            matmul_waveop.append(instr)
            start_tensor_calc = False   # this is only true for the first MatMul, even when there's a break
        return matmul_waveop

    # generate Pool waveop and add it to waveop stream
    # TODO: currently, always go to SB after Pooling
    # TODO: currently, cannot process multiple batch items in one instruction
    def gen_pool_waveop(self, tpb, ifmap_tile, ofmap_tile, src_is_psum, src_psum_bank_id, start_at_mid_part, partial_batch_item):
        batch_item = ofmap_tile.tile.n_id * self.pool_op.Tn
        window_x = ifmap_tile.window.x
        window_y = ifmap_tile.window.y
        # TODO: currently assumes that the pool op is always the first op
        # but it might be better to pass the op in
        in_dtype = self[0].ifmaps_file_params.data_type
        assert in_dtype in {'float16', 'bfloat16', 'float32', 'uint8', 'int32', 'uint16', 'int64'}
        if src_is_psum:
            if in_dtype in {'float16', 'bfloat16'}:
                in_dtype = 'float32'
            elif in_dtype == 'uint8':
                in_dtype = 'int32'
            elif in_dtype == 'uint16':
                in_dtype = 'int64'
        out_dtype = self[0].ofmaps_file_params.data_type
        assert out_dtype in {'float16', 'bfloat16', 'float32', 'uint8', 'int32', 'uint16'}
        if (src_is_psum):
            src_ifmap_width = ifmap_tile.subtile_rect.dim2d.x
            if self.conv_op != None and self.conv_op.repl_multiple_of_C > 1:
                src_ifmap_width = self.conv_op.W // self.conv_op.stride.x
            src_ifmap_height = ifmap_tile.subtile_rect.dim2d.y
            src_sb_address = 0
        else:
            src_ifmap_width = self.pool_op.W
            src_ifmap_height = self.pool_op.H
            src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.pool_op.ifmaps_file_params, batch_item + partial_batch_item, ifmap_tile.lower_addr[partial_batch_item])
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
              'pool_func'               : self.pool_op.data['layer_type'],
              'in_dtype'                : in_dtype,
              'out_dtype'               : out_dtype,
              'src_is_psum'             : src_is_psum,
              'src_x_step'              : 1,
              'src_x_num'               : window_x,
              'src_y_step'              : src_ifmap_width,
              'src_y_num'               : window_y,
              'src_z_step'              : self.pool_op.stride.x,
              'src_z_num'               : ofmap_tile.subtile_rect.dim2d.x,
              'src_w_step'              : src_ifmap_width * self.pool_op.stride.y,
              'src_w_num'               : ofmap_tile.subtile_rect.dim2d.y,
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
    def execute_matmul_waveop(self, tpb, ifmap_pewave, ofmap_pewave, inputs, weights, psum_add, psum_bank_id, repl_multiple_of_C, is_const_weights = True):
        batch_item = ofmap_pewave.tile.n_id * self.conv_op.Tn
        self.conv_op.compute_ifmap_ofmap_pewave_info(ifmap_pewave, ofmap_pewave, self.conv_op.is_conv_transpose)
        pearray_packed_weights = self.conv_op.pack_wave_conv_weights(
                                        weights, 
                                        ifmap_pewave,
                                        ofmap_pewave, 
                                        repl_multiple_of_C,
                                        is_const_weights
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
            length = self.conv_op.ifmap_wave_upper_addr[0] - self.conv_op.ifmap_wave_lower_addr[0] + self.conv_op.ifmaps_file_params.item_sz
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
                        , tpb.pearray.psum_buf[psum_bank_id]
                        , not psum_add
                        , self.conv_op.stride)
            else:
                #assert(self.conv_op.psum_bank_offset == ofmap_pewave.subtile_psum_offset)
                ifmap_data_type = ifmap_pewave.tile.file_params.data_type
                if ifmap_data_type in {'uint8', 'uint16'}:
                    tpb.pearray.wave_uint8_mm(pearray_packed_ifmaps, pearray_packed_weights, psum_bank_id, psum_add,
                        offset_ifmaps=self.conv_op.data['zero_point_input'],
                        offset_weights=self.conv_op.data['zero_point_filter'])
                elif ifmap_data_type in {'float16', 'bfloat16', 'float32'}:
                    tpb.pearray.wave_fp16_mm(pearray_packed_ifmaps, pearray_packed_weights, psum_bank_id, psum_add)
                else:
                    raise NotImplementedError('ifmap_pewave has wrong data type %s' % ifmap_data_type)
            # Generate weights waveops
            load_weights_into_pearray = not tpb.pearray.check_loaded_weights(
                                            self.conv_op.weights_file_params, 
                                            self.conv_op.weight_wave_lower_addr,
                                            batch_item) \
                                        or self.first_op.is_concat   # always load weights for concat
            (writers, readers, dram_weights_waveops, new_reader_morsels) = ([], [], [], [])
            
            # if weights are dynamic, it will be collected to weight_writer_waveop.
            weight_writer_waveops = []
            if load_weights_into_pearray:
                (writers, readers, dram_weights_waveops, new_reader_morsels) = tpb.statebuffer.file_mapper.read_file_data_region(
                                        tpb.waveop_stream.waveop_count,
                                        tpb.waveop_stream,
                                        self.conv_op.weights_file_params,
                                        0,  # batch_item doesn't apply for weights
                                        self.conv_op.weight_wave_lower_addr, 
                                        self.conv_op.weight_wave_upper_addr - self.conv_op.weight_wave_lower_addr + self.conv_op.weights_file_params.item_sz,
                                        EngineEnum.PEARRAY,
                                        start_at_mid_part = False,
                                        end_after_mid_part = True,
                                        repl_multiple_of_C = repl_multiple_of_C)
                weight_writer_waveops = [ tpb.waveop_stream[waveop_id] for waveop_id in weight_writer_waveops ]
                if (self.args.debug > 2): print("DBG %s: MatMul weight_wave_lower_addr %d weight_wave_upper_addr %d"%(self.conv_op.data['layer_name'], self.conv_op.weight_wave_lower_addr, self.conv_op.weight_wave_upper_addr))                
                for i in dram_weights_waveops: tpb.waveop_stream.add_linked(i, [], -1)

            dram_ifmaps_waveops = []
            list_of_accessors = writers
            for z in range(self.conv_op.Tn):
                # TODO: move the following into gen_matmul_waveop to handle breaking wave into two
                (writers, readers, waveops, reader_morsels) = tpb.statebuffer.file_mapper.read_file_data_region(
                                            tpb.waveop_stream.waveop_count,
                                            tpb.waveop_stream,
                                            ifmap_pewave.tile.file_params,
                                            batch_item + z,
                                            ifmap_pewave.lower_addr[z],
                                            ifmap_pewave.lower_to_upper_len_bytes[z],
                                            EngineEnum.PEARRAY,
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
            
            for i in dram_ifmaps_waveops: 
                tpb.waveop_stream.add_linked(i, [], -1)

            # kaena-1144: conv transpose OFMAP have holes (unwritten data) if stride > filter size; 
            # so clear PSUM tile explicitly before first MatMul of tensor computation (start_tensor_calc=True).
            if self.conv_op.is_conv_transpose \
                    and not psum_add \
                    and self.conv_op.stride > self.conv_op.filter:
                self.gen_psum_clr_waveop(
                        tpb, 
                        ifmap_pewave, 
                        ofmap_pewave, 
                        psum_bank_id)

            matmul_waveop = self.gen_matmul_waveop(
                    tpb, 
                    ifmap_pewave, 
                    ofmap_pewave, 
                    psum_add, 
                    psum_bank_id, 
                    dram_weights_waveops + weight_writer_waveops, 
                    load_weights_into_pearray,
                    repl_multiple_of_C, 
                    self.conv_op.is_conv_transpose)

            #print("DBG list_of_accessors ", list_of_accessors, len(dram_ifmaps_waveops))
            prev_waveops = extract_predecessors(
                    list_of_accessors_wr = list_of_accessors, 
                    list_of_accessors_rd = [[]],
                    waveop_list          = tpb.waveop_stream)
            #print("DBG prev_waveops ", prev_waveops)

            for i in range(len(matmul_waveop)):
                tpb.waveop_stream.add_linked(matmul_waveop[i], [], psum_bank_id, new_reader_morsels)
                if i==0:    # only need to satisfy the first in group of matmul waveops
                    attach_predecessors(matmul_waveop[i], prev_waveops)
            return True
        
    # execute remaining fused ops
    def execute_postconv_tile_ops (self, tpb, ifmap_tile, ofmap_tile, psum_bank_id):
        assert(psum_bank_id >= 0)
        assert(psum_bank_id < PEArray.PSUM_NUM_BANKS)
        psum_temp = tpb.pearray.extract_psum(psum_bank_id, 0, self.first_op.ofmap_full_tile_sz * ofmap_tile.Tn)
        op_list_iter = iter(range(1, len(self)))
        op_list = self
        batch_item = ofmap_tile.n_id * ofmap_tile.Tn
        prev_waveops = []
        for i in op_list_iter:
            layer_type = self[i].data['layer_type'] 
            layer_data = self[i].data
            if (re.search(self.act_ops_regex, layer_type)):
                dram_bias_waveops = []
                list_of_accessors_wr = []
                list_of_accessors_rd = [[] for l in range(EngineEnum.COUNT)]
                if 1: # (ofmap_tile.m_id%2 == 0):
                    (writers, readers, dram_bias_waveops, new_reader_morsels) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                    tpb.waveop_stream.waveop_count,
                                                    tpb.waveop_stream,
                                                    self[i].parent.combined_bias_file_params,
                                                    0,  # batch_item is not needed for bias
                                                    0,  # zeros start at address 0 of combined bias file
                                                    self[i].parent.combined_bias_file_params.item_sz,
                                                    EngineEnum.ACT,
                                                    )
                    list_of_accessors_wr += writers # writers for WAW dependencies

                alpha = 1.0
                if layer_type == 'Lrelu':
                    alpha = self[i].data['scalar_val']
                scale = self[i].data['scale'] if 'scale' in self[i].data else 1.0
                psum_temp = tpb.activate.act(op_list[i].data['layer_type'], psum_temp, alpha, scale)
                dst_is_psum = self[i].dst_is_psum 
                if dst_is_psum:
                    tpb.pearray.write_psum(psum_bank_id, 0, psum_temp.shape[0], psum_temp)
                waveop = self.gen_act_waveop_inline(tpb, None, op_list[i], self.first_op, ifmap_tile, ofmap_tile, 
                                          True, psum_bank_id, dst_is_psum, psum_bank_id, dram_bias_waveops, 
                                          0,  # zeros start at address 0 of combined bias file
                                          new_reader_morsels)
                prev_waveops = extract_predecessors(
                        list_of_accessors_wr = list_of_accessors_wr, 
                        list_of_accessors_rd = list_of_accessors_rd,
                        waveop_list         = tpb.waveop_stream)
                attach_predecessors(waveop, prev_waveops)
            elif (layer_type == 'BiasAdd'):
                # load bias values
                bias_data_temp          = self[i].parent.combined_bias_file_params.dram_data
                bias_chan_start         = self[i].bias_file_c_fold_offset * PEArray.NUM_ROWS + ofmap_tile.m_id * PEArray.NUM_COLS
                if self[i].parent.combined_bias_file_params.file_dims.has_c:
                    bias_addr           = (bias_chan_start // PEArray.NUM_ROWS) * self[i].parent.combined_bias_file_params.item_sz
                else:                    
                    bias_addr           = bias_chan_start * self[i].parent.combined_bias_file_params.item_sz
                bias_chan_mid_part      = (ofmap_tile.m_id%2) == 1
                bias_chan_end           = min(bias_chan_start + PEArray.NUM_COLS, self[i].bias_file_c_fold_offset * PEArray.NUM_ROWS + ofmap_tile.file_params.file_dims.C)
                bias_extracted          = np.zeros(bias_chan_end - bias_chan_start)
                for chan in range(bias_chan_start, bias_chan_end):
                    fold = chan // PEArray.NUM_ROWS
                    part = chan % PEArray.NUM_ROWS
                    bias_extracted[chan - bias_chan_start] = self[i].parent.combined_bias_file_params.dram_data[part, 0, fold, 0, 0]
                    #print("c_fold_offset %d chan %d fold %d partition %d value %d"%(self[i].bias_file_c_fold_offset, chan, fold, part, bias_extracted[chan - bias_chan_start]))
                dram_bias_waveops   = []
                list_of_accessors_wr = []
                list_of_accessors_rd = [[] for l in range(EngineEnum.COUNT)]
                if 1: #(ofmap_tile.m_id%2 == 0):
                    (writers, readers, dram_bias_waveops, new_reader_morsels) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                    tpb.waveop_stream.waveop_count,
                                                    tpb.waveop_stream,
                                                    self[i].parent.combined_bias_file_params,
                                                    0,  # batch_item is not needed for bias
                                                    bias_addr,
                                                    self[i].parent.combined_bias_file_params.item_sz,
                                                    EngineEnum.ACT,
                                                    )
                    list_of_accessors_wr += writers # writers for WAW dependencies 
                #x = DBG_DUMP_PSUM_COL("PSUM col0 before BiasAdd (FP32): ", psum_temp, 0)
                scale = self[i].data['scale'] if 'scale' in self[i].data else 1.0
                psum_temp = tpb.activate.biasadd(psum_temp, bias_extracted, scale)
                #y = DBG_DUMP_PSUM_COL("PSUM col0 after BiasAdd: ", psum_temp, 0)
                dst_is_psum = False
                if (i+1 < len(op_list) and re.search(self.act_ops_regex, op_list[i+1].data['layer_type'])):
                    act_type = op_list[i+1].data['layer_type']
                    alpha = 1.0
                    if act_type == 'Lrelu':
                        alpha = op_list[i+1].data['scalar_val']
                    psum_temp = tpb.activate.act(act_type, psum_temp, alpha)
                    if (i+1 != len(op_list)-1):
                        dst_is_psum = True
                        tpb.pearray.write_psum(psum_bank_id, 0, self.first_op.ofmap_full_tile_sz * ofmap_tile.Tn, psum_temp)
                    waveop = self.gen_act_waveop_inline(tpb, op_list[i], op_list[i+1], self.first_op, ifmap_tile, ofmap_tile, 
                                              True, psum_bank_id, dst_is_psum, psum_bank_id, dram_bias_waveops, bias_addr, new_reader_morsels)
                    next(op_list_iter)
                else:                                    
                    if (i != len(op_list)-1):
                        dst_is_psum = True
                        tpb.pearray.write_psum(psum_bank_id, 0, self.first_op.ofmap_full_tile_sz * ofmap_tile.Tn, psum_temp)
                    waveop = self.gen_act_waveop_inline(tpb, op_list[i], None, self.first_op, ifmap_tile, ofmap_tile, 
                                              True, psum_bank_id, dst_is_psum, psum_bank_id, dram_bias_waveops, bias_addr, new_reader_morsels)
                prev_waveops = extract_predecessors(
                        list_of_accessors_wr = list_of_accessors_wr, 
                        list_of_accessors_rd = list_of_accessors_rd,
                        waveop_list         = tpb.waveop_stream)
                attach_predecessors(waveop, prev_waveops)
            elif (self[i].is_join):
                residue_file_params = self[i].prev[self[i].residue_index].ofmaps_file_params
                if residue_file_params.mapped_params.start_addr == ofmap_tile.file_params.mapped_params.start_addr:
                    if not self[i].is_input:
                        residue_file_params.mapped_params.modify_in_place = True
                dram_resadd_waveops = []
                list_of_accessors = []
                new_reader_morsels = []

                if 1: # (ofmap_tile.m_id%2 == 0):
                    for z in range(op_list.first_op.Tn):
                        (writers, readers, waveops, reader_morsels) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                    tpb.waveop_stream.waveop_count,
                                                    tpb.waveop_stream,
                                                    residue_file_params,
                                                    batch_item + z,
                                                    ofmap_tile.lower_addr[z], 
                                                    ofmap_tile.upper_addr[z] - ofmap_tile.lower_addr[z] + ofmap_tile.file_params.item_sz,
                                                    EngineEnum.POOL,
                                                    )
                        #if self.args.no_inter_layer_load:
                        #    if (not self.first_op.is_input and len(waveops) > 0):
                        #        raise RuntimeError("There are DRAM loads when option no_inter_layer_load is set")
                        if (self.args.debug > 2): print("DBG %s: ResAdd/Mult ofmaps_tile_lower_addr %d ofmap_tile_upper_addr %d"%(self.first_op.data['layer_name'], ofmap_tile.lower_addr[z], ofmap_tile.upper_addr[z]))                
                        dram_resadd_waveops += waveops
                        list_of_accessors += writers    # only include writers for RAW dependency
                        new_reader_morsels += reader_morsels

                # Do the actual math
                residue_tile = ofmap_tile.copy()
                residue_tile.file_params = residue_file_params
                residue_ifmaps = residue_tile.get_tile_data_from_file(flatten=True)
                num_cols = residue_ifmaps.shape[1]
                num_elems = residue_ifmaps.shape[0]
                #x1 = DBG_DUMP_PSUM_COL("PSUM col0 before ResAdd (FP32): ", psum_temp, 0)
                #x2 = DBG_DUMP_PSUM_COL("Residue col0 before ResAdd (FP32): ", residue_ifmaps, 0)
                if (layer_type == 'ResAdd' or layer_type == "Add"):
                    psum_temp[0:num_elems, 0:num_cols] = tpb.pool.resadd(psum_temp[0:num_elems, 0:num_cols], residue_ifmaps)
                elif (layer_type == 'Sub'):
                    psum_temp[0:num_elems, 0:num_cols] = tpb.pool.sub(psum_temp[0:num_elems, 0:num_cols], residue_ifmaps)
                elif (layer_type == 'Multiply'):    
                    psum_temp[0:num_elems, 0:num_cols] = tpb.pool.multiply(psum_temp[0:num_elems, 0:num_cols], residue_ifmaps)
                elif (layer_type == 'Maximum'):    
                    psum_temp[0:num_elems, 0:num_cols] = tpb.pool.maximum(psum_temp[0:num_elems, 0:num_cols], residue_ifmaps)
                elif (layer_type == 'Minimum'):    
                    psum_temp[0:num_elems, 0:num_cols] = tpb.pool.minimum(psum_temp[0:num_elems, 0:num_cols], residue_ifmaps)
                else:
                    raise RuntimeError("ERROR: don't know how to handle vector op %s for layer %s"%(layer_type, self[i].data["layer_name"]))
                #y1 = DBG_DUMP_PSUM_COL("PSUM col0 after RessAdd (FP32): ", psum_temp, 0)
                dst_is_psum = self[i].dst_is_psum 
                if dst_is_psum:
                    tpb.pearray.write_psum(psum_bank_id, 0, self.first_op.ofmap_full_tile_sz * ofmap_tile.Tn, psum_temp)
                residue_file_addr = residue_file_params.ravel_nchw(
                        ofmap_tile.n_id * ofmap_tile.Tn,
                        ofmap_tile.m_id // 2 * PEArray.NUM_ROWS,
                        ofmap_tile.tile_rect.lower.y,
                        ofmap_tile.tile_rect.lower.x)
                residue_sb_addr =  tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(residue_file_params, batch_item, residue_file_addr)
                waveop = self.gen_join_waveop_inline(
                        tpb                 = tpb, 
                        op                  = op_list[i], 
                        ifmap_tile          = ifmap_tile, 
                        ofmap_tile          = ofmap_tile, 
                        src_is_psum         = True,
                        psum_bank_src       = psum_bank_id, 
                        dst_is_psum         = dst_is_psum, 
                        psum_bank_dst       = psum_bank_id if dst_is_psum else -1, 
                        dram_resadd_waveops = dram_resadd_waveops, 
                        residue_index       = self[i].residue_index,
                        psum_bank_residue   = -1,
                        residue_sb_addr     = residue_sb_addr,
                        new_reader_morsels  = new_reader_morsels)
                prev_waveops = extract_predecessors(
                        list_of_accessors_wr = list_of_accessors, 
                        list_of_accessors_rd = [[]],
                        waveop_list          = tpb.waveop_stream)
                attach_predecessors(waveop, prev_waveops)
            elif re.search("Multiply|ClipByValue|^Add$|Minimum|Maximum|^Sub$|QuantizeV2|Dequantize", layer_type):
                if layer_type == "ClipByValue":
                    scalar_val = [layer_data['clip_value_min'], layer_data['clip_value_max']]
                    scalar_op = [
                        self.translate_isa_op_name("maximum"),
                        self.translate_isa_op_name("minimum")
                    ]  # clip: max for lower bound, and min for the upper bound
                    scalar_op_reverse = [False, False]
                    psum_temp = tpb.pool.clipbyvalue(psum_temp, scalar_val[0], scalar_val[1])
                elif layer_type == 'QuantizeV2':
                    assert('quant_scale' in layer_data)
                    assert('zero_point' in layer_data)
                    scalar_val = [layer_data['quant_scale'], layer_data['zero_point']]
                    scalar_op = [
                        self.translate_isa_op_name("multiply"),
                        self.translate_isa_op_name("add"),
                    ]
                    scalar_op_reverse = [False, False]
                    psum_temp = tpb.pool.tensor_scalar_op(layer_type, psum_temp, scalar_val)
                elif layer_type == 'Dequantize':
                    assert('dequant_scale' in layer_data)
                    assert('zero_point' in layer_data)
                    scalar_val = [-layer_data['zero_point'], layer_data['dequant_scale']]
                    scalar_op = [
                        self.translate_isa_op_name("add"),
                        self.translate_isa_op_name("multiply"),
                    ]
                    scalar_op_reverse = [False, False]
                    psum_temp = tpb.pool.tensor_scalar_op(layer_type, psum_temp, scalar_val)
                else:
                    assert('scalar_val' in layer_data)
                    assert('reverse_operands' in layer_data)
                    scalar_val = [layer_data['scalar_val'], 0.0]
                    scalar_op = [layer_data['layer_type'].lower(), "bypass"]
                    scalar_op_reverse = [layer_data['reverse_operands'], False]
                    psum_temp = tpb.pool.tensor_scalar_op(layer_type, psum_temp, scalar_val[0], reverse=scalar_op_reverse[0])
                if self[i].dst_is_psum:                    
                    tpb.pearray.write_psum(psum_bank_id, 0, psum_temp.shape[0], psum_temp)
                waveop = self.gen_tensor_scalar_waveop_inline(
                            mapper        = tpb.statebuffer.file_mapper, 
                            waveop_stream = tpb.waveop_stream,
                            op            = self[i],
                            scalar_op     = scalar_op,
                            scalar_val    = scalar_val,
                            scalar_op_reverse = scalar_op_reverse,
                            ifmap_tile    = ifmap_tile, 
                            ofmap_tile    = ofmap_tile, 
                            psum_bank_id  = psum_bank_id,
                            dram_waveops  = [], 
                            new_reader_morsels = []
                            )
                attach_predecessors(waveop, prev_waveops)
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
                waveop = self.gen_fused_pool_waveop_inline(tpb, ifmap_subtile, ofmap_subtile, psum_bank_id, (ofmap_tile.m_id%2) == 1)
                attach_predecessors(waveop, prev_waveops)
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
        assert(psum_bank_dst >= 0)
        assert(psum_bank_src >= 0)
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
        tpb.waveop_stream.add_linked(instr, [], psum_bank_src)
        return instr

    # generate activation instruction and add it to instruction stream
    def gen_act_waveop_inline(self, tpb, biasadd_op, act_op, conv_op, ifmap_tile, ofmap_tile, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_start, new_reader_morsels=[]):
        layer_name = ""
        # kaena-452: load zeros into start of SB and use it for Activation instruction when there's no BiasAdd
        bias_sb_address = 0
        act_or_biasadd_op = None
        alpha = 1.0
        if biasadd_op is not None:
            act_or_biasadd_op = biasadd_op
            bias_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(biasadd_op.parent.combined_bias_file_params, 0, bias_start)
            layer_name = biasadd_op.data['layer_name']
        act_type = "Identity"    
        if act_op is not None:
            act_or_biasadd_op = act_op
            act_type = act_op.data['layer_type']
            if not re.search(self.act_ops_regex, act_type):
                act_type = "Identity"
            layer_name = act_op.data['layer_name']
            if act_type == 'Lrelu':
                alpha = act_op.data['scalar_val']
        first_op = biasadd_op if biasadd_op is not None else act_or_biasadd_op
        # Note (HC): cannot use ifmap_tile to determine in_dtype because
        # in quantized cases of fused conv->act we will operate on
        # the ofmap_tile that corresponds to the last op in this fused op list,
        # where this ofmap_tile is of float32 but what we need is int32
        in_dtype = first_op.ifmaps_file_params.data_type
        assert in_dtype in {'float16', 'bfloat16', 'float32', 'int32'}
        if src_is_psum and in_dtype in {'float16', 'bfloat16'}:
            in_dtype = 'float32'
        out_dtype = act_or_biasadd_op.ofmaps_file_params.data_type
        assert out_dtype in {'float16', 'bfloat16', 'float32'}
        if dst_is_psum and out_dtype in {'float16', 'bfloat16'}:
            out_dtype = 'float32'
        if biasadd_op is not None and 'scale' in biasadd_op.data: # pass scale to act op
            act_or_biasadd_op.data['scale'] = biasadd_op.data['scale']
        scale = act_or_biasadd_op.data['scale'] if 'scale' in act_or_biasadd_op.data else 1.0
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
            src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(ifmap_tile.file_params, batch_item, ifmap_tile.lower_addr[0])
        if dst_is_psum:
            dst_sb_address = 0
        else:            
            dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(ofmap_tile.file_params, batch_item, ofmap_tile.lower_addr[0])
        waveop_name = layer_name+"/Activation_"+ofmap_tile.id_string            
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : 'Activation',
              'waveop_name'             : waveop_name,
              'layer_name'              : layer_name,
              'activation_func'         : act_type,
              'alpha'                   : alpha,
              'in_dtype'                : in_dtype,
              'bias_dtype'              : act_or_biasadd_op.parent.combined_bias_file_params.data_type,
              'out_dtype'               : out_dtype,
              'scale'                   : scale,
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
              'bias_sb_address'         : bias_sb_address,
              'bias_start_at_mid_part'  : ofmap_tile.m_id%2 == 1,
            }
        if src_is_psum:
            assert(psum_bank_src >= 0)
            instr['src_psum_bank_id'] = psum_bank_src
            instr['src_psum_bank_offset'] = 0 
        else:                
            instr['src_sb_address'] = src_sb_address
            instr['src_start_at_mid_part'] = ofmap_tile.m_id%2 == 1
        if dst_is_psum:
            assert(psum_bank_dst >= 0)
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
    def gen_join_waveop_inline(self, tpb, op, ifmap_tile, ofmap_tile, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_resadd_waveops, residue_index, psum_bank_residue, residue_sb_addr, new_reader_morsels=[]):
        batch_item = ofmap_tile.batch_item_start
        num_partitions = PEArray.NUM_COLS
        num_partitions = ofmap_tile.channel_count
        start_at_mid_part = ofmap_tile.start_at_mid_part
        sb_addr = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(ifmap_tile.file_params, batch_item, ifmap_tile.lower_addr[0])
        assert(not(psum_bank_src >= 0 and psum_bank_residue >= 0)), "gen_join_waveop_inline for %s: cannot source both tensors from PSUM banks"%op.data['layer_name'] 
        # SB start addresses
        if residue_index == 0:
            # Source-B is PSUM if fused, or SB if unfused
            src_b_sb_address = 0 if src_is_psum else sb_addr
            # Source-A (Residue) is SB
            src_a_sb_address = 0 if psum_bank_residue >= 0 else residue_sb_addr
        else:
            # Source-A is PSUM if fused, or SB if unfused
            src_a_sb_address = 0 if src_is_psum else sb_addr
            # Source-B (Residue) is SB
            src_b_sb_address = 0 if psum_bank_residue >= 0 else residue_sb_addr

        # Destination SB address
        if dst_is_psum:
            dst_sb_address = 0
        else:            
            dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(ofmap_tile.file_params, batch_item, ofmap_tile.lower_addr[0])
        waveop_name = op.data['layer_name'] + "/" + op.data['layer_type'] + "_" + ofmap_tile.id_string
        op_type = op.data['layer_type'].lower()
        op_type = self.translate_isa_op_name(op_type)

        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : 'TensorTensor',
              'op'                      : op_type,
              'waveop_name'             : waveop_name,
              'layer_name'              : op.data['layer_name'],
              'num_partitions'          : num_partitions,
            }

        ofmap_subtile = ofmap_tile.make_pewave()
        ifmap_subtile = ifmap_tile.make_pewave()
        ofmap_subtile.set_waveop_pattern(tpb.statebuffer.file_mapper, instr, 'dst', psum_bank_dst, dst_sb_address, start_at_mid_part)
        # Note (HC): When ResAdd is fused with a later QuantizeV2 and when
        # the QuantizeV2 is also the last op in the fused op list, ifmap_subtile
        # will be coming from the output of QuantizeV2 and the data type is
        # therefore uint8. However the correct state buffer data type should be
        # float32. Here we assume that op.ifmaps_file_params.data_type has
        # the correct state buffer data type.
        ifmap_sb_dtype = op.ifmaps_file_params.data_type
        if residue_index == 0:
            ifmap_subtile.set_waveop_pattern(tpb.statebuffer.file_mapper, instr, 'src_b', psum_bank_src, src_b_sb_address, start_at_mid_part, sb_dtype=ifmap_sb_dtype)
            ifmap_subtile.set_waveop_pattern(tpb.statebuffer.file_mapper, instr, 'src_a', psum_bank_residue, src_a_sb_address, start_at_mid_part, sb_dtype=ifmap_sb_dtype)
        else:            
            ifmap_subtile.set_waveop_pattern(tpb.statebuffer.file_mapper, instr, 'src_a', psum_bank_src, src_a_sb_address, start_at_mid_part, sb_dtype=ifmap_sb_dtype)
            ifmap_subtile.set_waveop_pattern(tpb.statebuffer.file_mapper, instr, 'src_b', psum_bank_residue, src_b_sb_address, start_at_mid_part, sb_dtype=ifmap_sb_dtype)
        psum_bank_used = -1
        if psum_bank_src >= 0:
            psum_bank_used = psum_bank_src
        elif psum_bank_residue >= 0:
            psum_bank_used = psum_bank_residue
        tpb.waveop_stream.add_linked(instr, dram_resadd_waveops, psum_bank_used, new_reader_morsels)
        return instr

    # generate tensor-scalar waveop and add to waveop stream
    def gen_tensor_scalar_waveop_inline(self, 
            mapper, 
            waveop_stream, 
            op, 
            scalar_op,
            scalar_val,
            scalar_op_reverse,
            ifmap_tile, 
            ofmap_tile, 
            psum_bank_id, 
            dram_waveops=[], 
            new_reader_morsels=[]):
        batch_item = ofmap_tile.batch_item_start
        num_partitions = ofmap_tile.channel_count
        start_at_mid_part = ofmap_tile.start_at_mid_part
        # Source SB address or PSUM bank
        if op.src_is_psum:
            src_sb_address = 0
            src_psum_bank = psum_bank_id
        else:            
            src_sb_address = mapper.get_sb_addr_from_file_addr(ifmap_tile.file_params, batch_item, ifmap_tile.lower_addr[0])
            src_psum_bank = -1

        # Destination SB address or PSUM bank
        if op.dst_is_psum:
            dst_sb_address = 0
            dst_psum_bank = psum_bank_id
        else:            
            dst_sb_address = mapper.get_sb_addr_from_file_addr(ofmap_tile.file_params, batch_item, ofmap_tile.lower_addr[0])
            dst_psum_bank = -1

        waveop_name = op.data['layer_name'] + "/" + op.data['layer_type'] + "_" + ofmap_tile.id_string
        waveop_type = op.data['layer_type']

        for i in range(len(scalar_op)):
            scalar_op[i] = self.translate_isa_op_name(scalar_op[i])

        waveop = {
              'previous_waveops'        : [],
              'waveop_type'             : "TensorScalar",
              'waveop_name'             : waveop_name,
              'layer_name'              : op.data['layer_name'],
              'num_partitions'          : num_partitions,
              'op0'                     : scalar_op[0],
              'op1'                     : scalar_op[1],
              'reverse0'                : scalar_op_reverse[0],
              'reverse1'                : scalar_op_reverse[1],
              'imm_val0'                : scalar_val[0],
              'imm_val1'                : scalar_val[1],
            }

        ifmap_subtile = ifmap_tile.make_pewave()
        ofmap_subtile = ofmap_tile.make_pewave()
        ifmap_subtile.set_waveop_pattern(mapper, waveop, 'src', src_psum_bank, src_sb_address, start_at_mid_part,
            sb_dtype=op.ifmaps_file_params.data_type)
        ofmap_subtile.set_waveop_pattern(mapper, waveop, 'dst', dst_psum_bank, dst_sb_address, start_at_mid_part,
            sb_dtype=op.ofmaps_file_params.data_type)
        waveop_stream.add_linked(waveop, dram_waveops, src_psum_bank, new_reader_morsels)
        return waveop
        
    def gen_unary_tensor_waveop_inline(self, mapper, waveop_stream, op, ifmap_tile, ofmap_tile, psum_bank_id, dram_waveops=[], new_reader_morsels=[]):
        batch_item = ofmap_tile.batch_item_start
        num_partitions = ofmap_tile.channel_count
        start_at_mid_part = ofmap_tile.start_at_mid_part
        # Source SB address or PSUM bank
        if op.src_is_psum:
            src_sb_address = 0
            src_psum_bank = psum_bank_id
        else:            
            src_sb_address = mapper.get_sb_addr_from_file_addr(ifmap_tile.file_params, batch_item, ifmap_tile.lower_addr[0])
            src_psum_bank = -1

        # Destination SB address or PSUM bank
        if op.dst_is_psum:
            dst_sb_address = 0
            dst_psum_bank = psum_bank_id
        else:            
            dst_sb_address = mapper.get_sb_addr_from_file_addr(ofmap_tile.file_params, batch_item, ofmap_tile.lower_addr[0])
            dst_psum_bank = -1

        waveop_name = op.data['layer_name'] + "/" + op.data['layer_type'] + "_" + ofmap_tile.id_string
        waveop_type = op.data['layer_type']

        waveop = {
              'previous_waveops'        : [],
              'waveop_type'             : waveop_type,
              'waveop_name'             : waveop_name,
              'layer_name'              : op.data['layer_name'],
              'num_partitions'          : num_partitions,
            }

        ifmap_subtile = ifmap_tile.make_pewave()
        ofmap_subtile = ofmap_tile.make_pewave()
        ifmap_subtile.set_waveop_pattern(mapper, waveop, 'src', src_psum_bank, src_sb_address, start_at_mid_part)
        ofmap_subtile.set_waveop_pattern(mapper, waveop, 'dst', dst_psum_bank, dst_sb_address, start_at_mid_part)
        waveop_stream.add_linked(waveop, dram_waveops, src_psum_bank, new_reader_morsels)
        return waveop        

    def gen_fused_pool_waveop_inline (self, tpb, ifmap_tile, ofmap_tile, psum_bank_src, start_at_mid_part, new_reader_morsels=[]):
        first_waveop = None
        for z in range(self.pool_op.Tn):
            pool_waveop = self.gen_pool_waveop(tpb, ifmap_tile, ofmap_tile, True, psum_bank_src, start_at_mid_part, z)
            tpb.waveop_stream.add_linked(pool_waveop, [], psum_bank_src, new_reader_morsels)
            if z==0:
                first_waveop = pool_waveop                        
        return first_waveop

    def gen_unfused_pool_waveop_inline (self, tpb, ifmap_tile, ofmap_tile, dram_waveops, prev_waveops, start_at_mid_part, new_reader_morsels=[]):
        first_waveop = None
        for z in range(self.pool_op.Tn):
            pool_waveop = self.gen_pool_waveop(tpb, ifmap_tile, ofmap_tile, False, 0, start_at_mid_part, z)
            tpb.waveop_stream.add_linked(pool_waveop, dram_waveops if z==0 else [], -1, new_reader_morsels)
            # Add non-DRAM waveops to a previouse_waveops list
            existing_prev_waveops = pool_waveop['previous_waveops']
            for i in prev_waveops:
                if i not in existing_prev_waveops and i != pool_waveop['waveop_name']:
                    existing_prev_waveops.append(i)
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

        # start tensor computation by clearing psum bank
        psum_add = False                               

        # use conv to sum the exponential results
        # wave loop ordering scheme: nmhwcRS
        for n_id in range(self.conv_op.n):
            for m_id in range(self.conv_op.m):
                for h_id in range(self.conv_op.h):
                    for w_id in range(self.conv_op.w):
                        tile_id = (n_id, m_id, h_id, w_id, self.conv_op.n, self.conv_op.m, self.conv_op.h, self.conv_op.w)
                        ifmap_tile = Tile(tile_id, self.conv_op.ifmaps_file_params, self.conv_op.Tn, is_pe_input=True)
                        ofmap_tile = Tile(tile_id, self.conv_op.ofmaps_file_params, self.conv_op.Tn, is_pe_input=False)
                        self.conv_op.compute_ifmap_ofmap_tile_info(ifmap_tile, ofmap_tile)
                        # Pick psum bank to use, and advance pointer to new bank (cycle through 4 banks), 
                        psum_bank_id = tpb.pearray.use_psum_bank_and_adv_ptr(dst_is_psum = True)
                        # compute ofmap tile information (tile startx, starty, height, width)
                        # loops for constructing a tile
                        for c_id in range(self.conv_op.c):
                            for r_id in range(self.conv_op.R):
                                for s_id in range(self.conv_op.S):
                                    ifmap_pewave = PEWave(ifmap_tile, c_id, r_id, s_id)
                                    ofmap_pewave = PEWave(ofmap_tile, c_id, r_id, s_id)
                                    if (self.parent.args.debug > 2): print (ofmap_pewave.id_array)
                                    # execute PEArray matrix multiply, and add to PSUM after first wave
                                    if (self.execute_matmul_waveop(self, ifmap_pewave, ofmap_pewave, inputs, weights, psum_add, psum_bank_id, 1)):
                                        psum_add = True
                        # tile is done                                   
                        #self.waveop_stream[-1]['stop_tensor_calc'] = True  # This field is for debug only
                        # extract PSUM data
                        psum_temp = tpb.pearray.extract_psum(psum_bank_id, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn)
                        # go through the remaining operations
                        psum_temp = tpb.pool.reciprocate(psum_temp, self.conv_op.M)
                        tpb.pearray.write_psum(psum_bank_id, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, psum_temp)
                        tpb.gen_recip_waveop_inline(tpb, self.conv_op, psum_bank_id, True, psum_bank_id)
                        # loops for final scaling
                        for c_id in range(ceildiv(self.conv_op.C, PEArray.NUM_COLS)):
                            wave_id = ofmap_tile.make_pewave()
                            pearray_packed_ifmaps = self.conv_op.pack_wave_ifmaps(inputs, wave_id, 1, for_softmax=True)
                            scale_val = tpb.pearray.extract_psum(psum_bank_id, 0, 1)[0,0]
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
                                #                            self.waveop_stream[-1])
                       # The scale_add destination need to be adjusted after the above writes to data region
                        if (self.waveop_stream[-1]['waveop_type'] == "ScaleAdd"):
                            sb_addr = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.conv_op.ofmaps_file_params, 0, ofmap_tile.lower_addr[0])
                            self.waveop_stream[-1]['dst_sb_address'] = sb_addr
                        self.waveop_stream.add_outputs(dram_output_waveops)
                        # Get PSUM bank ready for the next tensor computation
                        psum_add = False

    def execute_pool_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id):
        first_op = self.first_op
        ifmap_subtile = ifmap_tile.make_pewave()
        ifmaps_data = first_op.pack_wave_ifmaps_unfused_pooling(ifmap_tile.file_params.dram_data, ifmap_subtile)
        input_tilex = ifmap_tile.tile_rect.dim2d.x
        input_tiley = ifmap_tile.tile_rect.dim2d.y
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
            #print("execute_pool_tile::pool_waves size = %d"%len(pool_waves))
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
        elif first_op.is_join:
            residue_file_params = first_op.prev[first_op.residue_index].ofmaps_file_params
            residue_tile = ofmap_tile.copy()
            residue_tile.file_params = residue_file_params
            residue_ifmaps = residue_tile.get_tile_data_from_file(flatten=True)
            if (layer_type == "ResAdd" or layer_type == "Add"):
                tile_data_flatten = tpb.pool.resadd(ifmaps_data_extract, residue_ifmaps)
            elif (layer_type == "Sub"):                                    
                tile_data_flatten = tpb.pool.sub(ifmaps_data_extract, residue_ifmaps)
            elif (layer_type == "Multiply"):                                    
                tile_data_flatten = tpb.pool.multiply(ifmaps_data_extract, residue_ifmaps)
            elif (layer_type == "Maximum"):                                    
                tile_data_flatten = tpb.pool.maximum(ifmaps_data_extract, residue_ifmaps)
            elif (layer_type == "Minimum"):                                    
                tile_data_flatten = tpb.pool.minimum(ifmaps_data_extract, residue_ifmaps)
            else:
                raise RuntimeError("ERROR: don't know how to handle tensor-tensor op %s for layer %s"%(layer_type, first_op.data['layer_name']))
        elif re.search("Multiply|ClipByValue|^Add$|Minimum|Maximum|^Sub$|QuantizeV2|Dequantize", layer_type):
            if (layer_type == "ClipByValue"):
                scalar_val = [first_op.data['clip_value_min'], first_op.data['clip_value_max']]
                tile_data_flatten = tpb.pool.clipbyvalue(ifmaps_data_extract, scalar_val[0], scalar_val[1])
            elif layer_type == 'QuantizeV2':
                assert('quant_scale' in first_op.data)
                assert('zero_point' in first_op.data)
                scalar_val = [first_op.data['quant_scale'], first_op.data['zero_point']]
                tile_data_flatten = tpb.pool.tensor_scalar_op(layer_type, ifmaps_data_extract, scalar_val)
            elif layer_type == 'Dequantize':
                assert('dequant_scale' in first_op.data)
                assert('zero_point' in first_op.data)
                scalar_val = [-first_op.data['zero_point'], first_op.data['dequant_scale']]
                tile_data_flatten = tpb.pool.tensor_scalar_op(layer_type, ifmaps_data_extract, scalar_val)
            else:
                assert('scalar_val' in first_op.data)
                assert('reverse_operands' in first_op.data)
                scalar_val = [first_op.data['scalar_val'], 0.0]
                scalar_op_reverse = [first_op.data['reverse_operands'], False]
                tile_data_flatten = tpb.pool.tensor_scalar_op(layer_type, ifmaps_data_extract, scalar_val[0], reverse=scalar_op_reverse[0])
        elif (layer_type == "BiasAdd"):
            bias_chan_start = first_op.bias_file_c_fold_offset * PEArray.NUM_ROWS + ofmap_tile.m_id * PEArray.NUM_COLS
            bias_chan_end = min(bias_chan_start + PEArray.NUM_COLS, first_op.bias_file_c_fold_offset * PEArray.NUM_ROWS + first_op.M)
            bias_chan_mid_part = (ofmap_tile.m_id%2) == 1
            #x = DBG_DUMP_PSUM_COL("BiasAdd: ", ifmaps_data_extract, 0)
            bias_extracted = np.zeros(bias_chan_end - bias_chan_start)
            for chan in range(bias_chan_start, bias_chan_end):
                fold = chan // PEArray.NUM_ROWS
                part = chan % PEArray.NUM_ROWS
                bias_extracted[chan - bias_chan_start] = first_op.parent.combined_bias_file_params.dram_data[part, 0, fold, 0, 0]
            tile_data_flatten = tpb.activate.biasadd(ifmaps_data_extract, bias_extracted)
            #x = DBG_DUMP_PSUM_COL("BiasAdd: ", tile_data_flatten, 0)
        elif re.search(self.act_ops_regex, layer_type):
            alpha = 1.0
            if layer_type == 'Lrelu':
                alpha = self.first_op.data['scalar_val']
            tile_data_flatten = tpb.activate.act(layer_type, ifmaps_data_extract, alpha)
        elif (layer_type == "Reciprocal"):
            tile_data_flatten = tpb.pool.reciprocate(ifmaps_data_extract, ifmap_tile.channel_count)
        else:
            raise RuntimeError("ERROR: cannot execute %s in execute_unfused_first_op"%layer_type)

        # Set resulting tile data into OFMAP (for pooling, we went ahead and do subtile pooling)
        if not (layer_type == "AvgPool" or layer_type == "MaxPool"):
            if first_op.dst_is_psum:
                tpb.pearray.write_psum(psum_bank_id, 0, tile_data_flatten.shape[0], tile_data_flatten)
            else:
                ofmap_tile.set_tile_data_in_file(tile_data_flatten)

        for z in range(first_op.Tn):
            if (self.args.debug > 2): print("TRACE execute_unfused_first_op %s: tile %s done, ifmap_tile_lower_addr %d ifmap_tile_upper_addr %d psum_bank %d, ofmap_tile_lower_addr %d ofmap_tile_upper_addr %dx"\
                        %(first_op.data["layer_name"], ofmap_tile.id_string, ifmap_tile.lower_addr[z], ifmap_tile.upper_addr[z], psum_bank_id, ofmap_tile.lower_addr[z], ofmap_tile.upper_addr[z]))

    """ Obtain producers of tile data for RAW dependencies; 
        if data doesn't exist in SB, generate SBAtomLoad waveops and return them
        Returns: (list of prev waveops, list of DRAM waveops)
    """
    def get_producers_for_subtile_region (self, tpb, fmap_subtile, reader_engine, subbatch_item = -1):
        # New 1
        # Start of RAW part
        tile = fmap_subtile.tile
        batch_item = tile.n_id * tile.Tn
        dram_fmaps_waveops = []
        list_of_accessors = []
        new_reader_morsels = []
        if subbatch_item >= 0:
            subbatch_range = range(subbatch_item, subbatch_item+1)
        else:
            subbatch_range = range(tile.Tn)
        for z in subbatch_range:
            if 1: # (tile.m_id%2 == 0):
                (writers, readers, waveops, reader_morsels) \
                    = tpb.statebuffer.file_mapper.read_file_data_region(
                        tpb.waveop_stream.waveop_count,
                        tpb.waveop_stream,
                        tile.file_params,
                        batch_item + z,
                        fmap_subtile.lower_addr[z], 
                        fmap_subtile.lower_to_upper_len_bytes[z],
                        reader_engine,
                        )
                dram_fmaps_waveops += waveops
                # don't include readers in dependencies since this pool is a 
                # reader, and we don't need to add RAR dependency
                list_of_accessors += writers
                new_reader_morsels += reader_morsels
                #print(last_writer, last_reader)
        prev_waveops = extract_predecessors(
                list_of_accessors_wr = list_of_accessors, 
                list_of_accessors_rd = [[]],
                waveop_list         = tpb.waveop_stream)

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
    def mark_producers_for_subtile_region (self, tpb, fmap_subtile, waveop, subbatch_item = -1):
        # New 3
        # Start of WAW and WAR part
        tile = fmap_subtile.tile
        batch_item = tile.n_id * tile.Tn
        start_at_mid_part = tile.m_id%2 == 1
        dram_output_waveops = []                            
        list_of_accessors_wr = []
        list_of_accessors_rd = [[] for l in range(EngineEnum.COUNT)]
        if subbatch_item >= 0:
            subbatch_range = range(subbatch_item, subbatch_item+1)
        else:
            subbatch_range = range(tile.Tn)
        for z in subbatch_range:
            # for scheduling, map resulting tile into portion of atom that is 
            # itself mapped to a portion in DRAM (file)
            # only record the writer to SB chunks in below code; 
            # use flush_file to write chunks to DRAM
            # (mark writer using ID of last waveop of current Tn batch items)
            (writers, readers, dram_waveops) \
                = tpb.statebuffer.file_mapper.write_file_data_region(
                                        tpb.waveop_stream.waveop_count - 1,    # adjust since pool waveop already generated
                                        tpb.waveop_stream,
                                        tile.file_params,
                                        batch_item + z,
                                        tile.lower_addr[z], 
                                        tile.lower_to_upper_len_bytes[z], 
                                        start_at_mid_part)
            #assert(len(waveops) == 0)                            
            list_of_accessors_wr += writers # Include readers for WAR dependencies and writers for WAW dependencies
            for l in range(EngineEnum.COUNT):
                list_of_accessors_rd[l] += readers[l]    # Include readers for WAR dependencies and writers for WAW dependencies
            dram_output_waveops += dram_waveops

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
        predec_list_by_name = extract_predecessors(
                list_of_accessors_wr  = list_of_accessors_wr, 
                list_of_accessors_rd  = list_of_accessors_rd, 
                waveop_list         = tpb.waveop_stream)
        attach_predecessors(waveop, predec_list_by_name)

    def emit_waveops_pool_tile(self, tpb, ifmap_tile, ofmap_tile, psum_bank_id):
        # wave loop ordering scheme: nmhw
        first_op = self.first_op
        layer_type = first_op.data['layer_type']
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
                (prev_waveops, dram_ifmaps_waveops, new_reader_morsels) \
                    = self.get_producers_for_subtile_region (
                            tpb          = tpb, 
                            fmap_subtile = ifmap_tile_subtile,
                            reader_engine = EngineEnum.POOL,
                            )
                #print(first_op.data['layer_type'], ifmap_tile_subtile.subtile_rect, ofmap_tile_subtile.subtile_rect, " prev: ", prev_waveops, " dram waveops: ", dram_ifmaps_waveops, psum_bank_id)
                waveop  = self.gen_unfused_pool_waveop_inline(
                        tpb               = tpb, 
                        ifmap_tile        = ifmap_tile_subtile, 
                        ofmap_tile        = ofmap_tile_subtile, 
                        dram_waveops      = dram_ifmaps_waveops, 
                        prev_waveops      = prev_waveops, 
                        start_at_mid_part = start_at_mid_part,
                        new_reader_morsels = new_reader_morsels)
                #print(waveop)
                self.mark_producers_for_subtile_region(
                        tpb           = tpb, 
                        fmap_subtile  = ofmap_tile_subtile, 
                        waveop        = waveop)
                #print(waveop)
        else:   
            bias_addr = 0 # zeros start at address 0 of combined bias file
            biasadd_op = None
            act_op = None
            ifmap_tile_subtile \
                    = PoolSubtile(ifmap_tile, ifmap_tile.tile_rect, Dim2D(1,1))
            ofmap_tile_subtile \
                    = PoolSubtile(ofmap_tile, ofmap_tile.tile_rect, Dim2D(1,1))
            layer_type = first_op.data['layer_type']
            if re.search(self.act_ops_regex, layer_type) or layer_type == "BiasAdd":
                reader_engine = EngineEnum.ACT
            else:                
                reader_engine = EngineEnum.POOL
            (prev_waveops, dram_ifmaps_waveops, new_reader_morsels) \
                = self.get_producers_for_subtile_region (
                        tpb          = tpb, 
                        fmap_subtile = ifmap_tile_subtile,
                        reader_engine = reader_engine,
                        )
            first_waveop = None
            last_waveop = None
            if (re.search(self.act_ops_regex, first_op.data['layer_type'])):
                (writers, readers, dram_bias_waveops, reader_morsels) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                tpb.waveop_stream.waveop_count + len(dram_ifmaps_waveops),
                                                tpb.waveop_stream,
                                                first_op.parent.combined_bias_file_params,
                                                0,  # batch_item is not needed for bias
                                                0,  # zeros start at address 0 of combined bias file
                                                first_op.parent.combined_bias_file_params.item_sz,
                                                EngineEnum.ACT,
                                                )
                dram_ifmaps_waveops += dram_bias_waveops
                new_reader_morsels += reader_morsels
                prev_wr_waveops = extract_predecessors(
                        list_of_accessors_wr = writers, 
                        list_of_accessors_rd = [[]],
                        waveop_list         = tpb.waveop_stream)
                for i in prev_wr_waveops:
                    if i not in prev_waveops:
                        prev_waveops.append(i)
                act_op = first_op
            elif (first_op.data['layer_type'] == "BiasAdd"): 
                bias_chan_start         = first_op.bias_file_c_fold_offset * PEArray.NUM_ROWS + ofmap_tile.m_id * PEArray.NUM_COLS
                if first_op.parent.combined_bias_file_params.file_dims.has_c:
                    bias_addr           = (bias_chan_start // PEArray.NUM_ROWS) * first_op.parent.combined_bias_file_params.item_sz
                else:                    
                    bias_addr           = bias_chan_start * first_op.parent.combined_bias_file_params.item_sz
                (writers, readers, dram_bias_waveops, reader_morsels) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                tpb.waveop_stream.waveop_count + len(dram_ifmaps_waveops),
                                                tpb.waveop_stream,
                                                first_op.parent.combined_bias_file_params,
                                                0,  # batch_item is not needed for bias
                                                bias_addr,
                                                first_op.parent.combined_bias_file_params.item_sz,
                                                reader_engine)
                #print(first_op.data['layer_name'], writers)
                dram_ifmaps_waveops += dram_bias_waveops
                new_reader_morsels += reader_morsels
                prev_wr_waveops = extract_predecessors(
                        list_of_accessors_wr = writers,
                        list_of_accessors_rd = [[]],
                        waveop_list         = tpb.waveop_stream)
                for i in prev_wr_waveops:
                    if i not in prev_waveops:
                        prev_waveops.append(i)
                biasadd_op = first_op

            if first_op.is_join:
                if 1:
                    residue_file_params = first_op.prev[first_op.residue_index].ofmaps_file_params
                    if residue_file_params.mapped_params.start_addr == ofmap_tile.file_params.mapped_params.start_addr:
                        if not first_op.is_input:
                            residue_file_params.mapped_params.modify_in_place = True
                    residue_tile = copy.copy(ifmap_tile)
                    residue_tile.file_params = residue_file_params
                    residue_subtile = PEWave(residue_tile, 0, 0, 0)
                    (residue_prev_waveops, dram_resadd_waveops, residue_reader_morsels) = \
                        self.get_producers_for_subtile_region(tpb, residue_subtile, reader_engine)
                    residue_file_addr = residue_file_params.ravel_nchw(
                            ofmap_tile.n_id * ofmap_tile.Tn,
                            ofmap_tile.m_id // 2 * PEArray.NUM_ROWS,
                            ofmap_tile.tile_rect.lower.y,
                            ofmap_tile.tile_rect.lower.x)
                    residue_sb_addr = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(residue_tile.file_params, batch_item, residue_file_addr)
                    main_sb_addr = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(ifmap_tile.file_params, batch_item, ifmap_tile.lower_addr[0])
                    psum_bank_src = psum_bank_id 
                    # If doing tensor-tensor op without PSUM, but distance between tensors is too far, or operation is non-commutativ, then move tile to PSUM
                    if abs(residue_sb_addr - main_sb_addr) > (2**15 - 1):
                        if psum_bank_src < 0:
                            psum_bank_residue = tpb.pearray.use_psum_bank_and_adv_ptr(True)
                        else:
                            psum_bank_residue = psum_bank_src
                        residue_tile.lower_addr[0] = residue_file_addr
                        # grab dependencies for zeros
                        (writers, readers, dram_bias_waveops, zeros_reader_morsels) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                        tpb.waveop_stream.waveop_count + len(dram_ifmaps_waveops),
                                                        tpb.waveop_stream,
                                                        first_op.parent.combined_bias_file_params,
                                                        0,  # batch_item is not needed for bias
                                                        0,  # zeros are located in first fold of bias-constants file
                                                        first_op.parent.combined_bias_file_params.item_sz,
                                                        EngineEnum.ACT,
                                                        )
                        first_waveop = self.gen_act_waveop_inline(
                                tpb               = tpb, 
                                biasadd_op        = None, 
                                act_op            = first_op, 
                                conv_op           = None, 
                                ifmap_tile        = residue_tile, 
                                ofmap_tile        = ofmap_tile, 
                                src_is_psum       = False,
                                psum_bank_src     = -1, 
                                dst_is_psum       = psum_bank_residue >= 0,
                                psum_bank_dst     = psum_bank_residue, 
                                dram_bias_waveops = dram_resadd_waveops + dram_bias_waveops, 
                                bias_start        = 0,  # zeros are located in first fold of bias-constants file
                                new_reader_morsels = residue_reader_morsels + zeros_reader_morsels)
                        prev_wr_waveops = extract_predecessors(
                                list_of_accessors_wr = writers, 
                                list_of_accessors_rd = [[]],
                                waveop_list         = tpb.waveop_stream)
                        for i in prev_wr_waveops:
                            if i not in residue_prev_waveops:
                                residue_prev_waveops.append(i)
                        attach_predecessors(first_waveop, residue_prev_waveops)
                    else:                       
                        act_waveop = None
                        new_reader_morsels += residue_reader_morsels
                        prev_waveops += residue_prev_waveops
                        dram_ifmaps_waveops += dram_resadd_waveops
                        psum_bank_residue = -1
                    last_waveop = self.gen_join_waveop_inline(
                            tpb                 = tpb, 
                            op                  = first_op, 
                            ifmap_tile          = ifmap_tile,
                            ofmap_tile          = ofmap_tile,
                            src_is_psum         = first_op.src_is_psum, 
                            psum_bank_src       = psum_bank_src if first_op.src_is_psum else -1, 
                            dst_is_psum         = first_op.dst_is_psum, 
                            psum_bank_dst       = psum_bank_src if first_op.dst_is_psum else -1, 
                            dram_resadd_waveops = dram_ifmaps_waveops,
                            residue_index       = first_op.residue_index,
                            psum_bank_residue   = psum_bank_residue,
                            residue_sb_addr     = residue_sb_addr,
                            new_reader_morsels  = new_reader_morsels)

            elif re.search("Multiply|ClipByValue|^Add$|Minimum|Maximum|^Sub$|QuantizeV2|Dequantize", layer_type):
                if layer_type == "ClipByValue":
                    scalar_val = [first_op.data['clip_value_min'], first_op.data['clip_value_max']]
                    scalar_op = [
                        self.translate_isa_op_name("maximum"),
                        self.translate_isa_op_name("minimum")
                    ]  # clip: max for lower bound, and min for the upper bound
                    scalar_op_reverse = [False, False]
                elif layer_type == 'QuantizeV2':
                    assert('quant_scale' in first_op.data)
                    assert('zero_point' in first_op.data)
                    scalar_val = [first_op.data['quant_scale'], first_op.data['zero_point']]
                    scalar_op = [
                        self.translate_isa_op_name("multiply"),
                        self.translate_isa_op_name("add"),
                    ]
                    scalar_op_reverse = [False, False]
                elif layer_type == 'Dequantize':
                    assert('dequant_scale' in first_op.data)
                    assert('zero_point' in first_op.data)
                    scalar_val = [-first_op.data['zero_point'], first_op.data['dequant_scale']]
                    scalar_op = [
                        self.translate_isa_op_name("add"),
                        self.translate_isa_op_name("multiply"),
                    ]
                    scalar_op_reverse = [False, False]
                else:
                    assert('scalar_val' in first_op.data)
                    assert('reverse_operands' in first_op.data)
                    scalar_val = [first_op.data['scalar_val'], 0.0]
                    scalar_op = [first_op.data['layer_type'].lower(), "bypass"]
                    scalar_op_reverse = [first_op.data['reverse_operands'], False]
                last_waveop = self.gen_tensor_scalar_waveop_inline(
                            mapper        = tpb.statebuffer.file_mapper, 
                            waveop_stream = tpb.waveop_stream,
                            op            = self[0],
                            scalar_op     = scalar_op,
                            scalar_val    = scalar_val,
                            scalar_op_reverse = scalar_op_reverse,
                            ifmap_tile    = ifmap_tile, 
                            ofmap_tile    = ofmap_tile, 
                            psum_bank_id  = psum_bank_id,
                            dram_waveops  = dram_ifmaps_waveops, 
                            new_reader_morsels = new_reader_morsels
                            )
            elif first_op.data['layer_type'] == 'Reciprocal':
                ifmap_tile_subtile \
                        = PoolSubtile(ifmap_tile, ifmap_tile.tile_rect, Dim2D(1,1))
                ofmap_tile_subtile \
                        = PoolSubtile(ofmap_tile, ofmap_tile.tile_rect, Dim2D(1,1))

                last_waveop = self.gen_unary_tensor_waveop_inline(
                    mapper             = tpb.statebuffer.file_mapper, 
                    waveop_stream      = tpb.waveop_stream,
                    op                 = self[0],
                    ifmap_tile         = ifmap_tile, 
                    ofmap_tile         = ofmap_tile, 
                    psum_bank_id       = psum_bank_id if first_op.dst_is_psum else -1,
                    dram_waveops       = dram_ifmaps_waveops, 
                    new_reader_morsels = new_reader_morsels)

            else:                    
                last_waveop = self.gen_act_waveop_inline(
                        tpb               = tpb, 
                        biasadd_op        = biasadd_op, 
                        act_op            = act_op, 
                        conv_op           = None, 
                        ifmap_tile        = ifmap_tile, 
                        ofmap_tile        = ofmap_tile, 
                        src_is_psum       = False, 
                        psum_bank_src     = -1, 
                        dst_is_psum       = first_op.dst_is_psum, 
                        psum_bank_dst     = psum_bank_id if first_op.dst_is_psum else -1, 
                        dram_bias_waveops = dram_ifmaps_waveops, 
                        bias_start        = bias_addr,
                        new_reader_morsels = new_reader_morsels)
            attach_predecessors(last_waveop, prev_waveops)
            self.mark_producers_for_subtile_region(
                    tpb           = tpb, 
                    fmap_subtile  = ofmap_tile_subtile, 
                    waveop        = last_waveop)

    # Execute an unfused pooling operator
    def execute_unfused_pool_op(self, tpb, batch_item):
        first_op = self[0]
        n_id = batch_item // first_op.Tn
        for m_id in range(first_op.m):
            for h_id in range(first_op.h):
                for w_id in range(first_op.w):
                    tile_id = (n_id, m_id, h_id, w_id, first_op.n, first_op.m, first_op.h, first_op.w)
                    ifmap_tile = Tile(tile_id, self.first_op.ifmaps_file_params, self.first_op.Tn, is_pe_input=False)
                    ofmap_tile = Tile(tile_id, self.last_op.ofmaps_file_params, self.last_op.Tn, is_pe_input=False)
                    self.first_op.compute_ifmap_ofmap_tile_info(ifmap_tile, ofmap_tile)
                    psum_bank_id = tpb.pearray.use_psum_bank_and_adv_ptr(dst_is_psum = self.first_op.dst_is_psum)
                    self.execute_pool_tile(tpb, ifmap_tile, ofmap_tile, psum_bank_id)
                    self.emit_waveops_pool_tile(tpb, ifmap_tile, ofmap_tile, psum_bank_id)
                    # execute subsequent instructions
                    if len(self) > 1:
                        self.execute_postconv_tile_ops(tpb, ofmap_tile, ofmap_tile, psum_bank_id)

    # Execute an unfused transpose operator
    def execute_unfused_transpose_op(self, tpb, batch_item):
        first_op = self.first_op
        last_op = self.last_op
        layer_name = first_op.data["layer_name"]
        # For ME behavior verification
        last_op.ofmaps_file_params.dram_data = \
            np.transpose(first_op.ifmaps_file_params.dram_data, first_op.data['perm'])
        # For wavegraph:    
        # for each group of 64 input channels
        # regload parallel -> regstore serial
        transpose_vec = False
        if first_op.data["layer_type"] == "TransposeVector":
            transpose_vec = True
            n_id = batch_item // first_op.Tn
            num_64_chan_folds = ceildiv(first_op.C, PEArray.NUM_COLS)            
            for m_id in range(num_64_chan_folds):
                ifmap_tile_id = (n_id, m_id, 0, 0, first_op.n, first_op.m, 1, 1)
                ifmap_tile = Tile(ifmap_tile_id, first_op.ifmaps_file_params, first_op.Tn, is_pe_input=False)
                ifmap_tile.tile_rect = Rect(Coord(0,0), Coord(0,0))
                ifmap_subtile = PEWave(ifmap_tile, 0, 0, 0)
                ofmap_tile_id = (n_id, 0, 0, m_id, first_op.n, 1, 1, first_op.m)
                ofmap_tile = Tile(ofmap_tile_id, last_op.ofmaps_file_params, last_op.Tn, is_pe_input=False)
                ofmap_tile.tile_rect = Rect(Coord(ifmap_tile.channel_start, 0), Coord(ifmap_tile.channel_stop - 1, 0))
                ofmap_subtile = PEWave(ofmap_tile, 0, 0, 0)
                self.gen_regload_vector_waveop(tpb, layer_name, ifmap_subtile, parallel_mode=True)
                self.gen_regstore_vector_waveop(tpb, layer_name, ofmap_subtile, parallel_mode=False)

        # Mark the entire output file for dependency checking
        #(last_writer, last_reader, waveops) = tpb.statebuffer.file_mapper.write_file_data_region(
        #                            tpb.waveop_stream.waveop_count - 1 if transpose_vec else -1,  # -1 signifies a pass-through (keep old writers of region)
        #                            tpb.waveop_stream,
        #                            last_op.ofmaps_file_params,
        #                            batch_item,
        #                            0,
        #                            last_op.ofmaps_file_params.file_sz,   # mark entire file
        #                            False)

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
        assert(slice_count > 0 and slice_count <= PEArray.NUM_COLS)
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
        #self.print_SB_addr(op.weights_file_params)

    def execute_unfused_stridedslice_op (self, tpb, batch_item):
        #print ("Stridedslice node name = %s"%self.first_op.data['layer_name'])
        first_op = self[0]
        n_id = batch_item // first_op.Tn
        ifmap_file_params = first_op.ifmaps_file_params
        first_op.R = 1
        first_op.S = 1
        slice_start = first_op.data['channel_slice'][0]
        slice_stop  = first_op.data['channel_slice'][1]
        if (slice_start % PEArray.NUM_ROWS) == 0:
            ofp = first_op.ofmaps_file_params
            ifp = first_op.ifmaps_file_params
            if ofp.file_dims.format_str == 'NCHW':
                ofp.dram_data = ifp.dram_data[:, slice_start : slice_stop]
                ofp.dram_data.shape = ofp.file_dims.shape_tuple
            else:
                raise RuntimeError("Format %s is not supported for StridedSlice operation"%(ofp.file_dims.format_str))
            if not first_op.is_input:
                (last_writer, last_reader, waveops) = tpb.statebuffer.file_mapper.write_file_data_region(
                                        -1,  # signifies a pass-through (keep old writers of region)
                                        tpb.waveop_stream,
                                        first_op.ofmaps_file_params,
                                        batch_item,
                                        0,
                                        first_op.ofmaps_file_params.file_sz,   # mark entire file
                                        False)
            else:
                self.no_writer_for_save = True  # pass-thru so there's no writer for SBAtomSave (so don't save this fused-op output)
            return

        # We get here since slice channel start is not multiple of 128, so cross-partition movement is required
        for m_id in range(first_op.m):
            first = True
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

                    psum_bank_id = tpb.pearray.use_psum_bank_and_adv_ptr(dst_is_psum = first_op.dst_is_psum)
                    tpb.pearray.wave_fp16_mm(packed_ifmap
                                             , packed_weights
                                             , psum_bank_id
                                             , False)

                    concat_collaterals = [ []
                             , ifmap_file_params 
                             , [slice_start, slice_stop-1]
                             , [0, first_op.weights_file_params.file_dims.M-1]]

                    self.emit_waveop_concat_tile(
                                                 tpb                  = tpb
                                                 , ifmap_tile         = ifmap_tile
                                                 , ofmap_tile         = ofmap_tile
                                                 , concat_collaterals = concat_collaterals
                                                 , ofmap_start        = next_ofmap_start
                                                 , start_tensor_calc  = 1
                                                 , psum_bank_id       = psum_bank_id
                                                 , num_weights        = 1
                                                 , first              = first
                                                )
                    first = False
                    MM_data = tpb.pearray.extract_psum(psum_bank_id, 0, ifmap_tile.tile_rect.get_tot_size())
                    ofmap_tile.set_tile_data_in_file(MM_data)
                    self.execute_postconv_tile_ops(tpb
                                                   , ofmap_tile
                                                   , ofmap_tile
                                                   , psum_bank_id)

    def execute_unfused_concat_op (self, tpb, batch_item):
        def move_psum_to_ofmap (tpb, ifmap_tile, ofmap_tile, psum_bank_id):
            MM_data = tpb.pearray.extract_psum(psum_bank_id, 0\
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
                    psum_bank_id = tpb.pearray.use_psum_bank_and_adv_ptr(dst_is_psum = self.first_op.dst_is_psum)
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
                                                         tpb                  = tpb
                                                         , ifmap_tile         = ifmap_tile
                                                         , ofmap_tile         = ofmap_tile
                                                         , concat_collaterals = concat_collaterals
                                                         , ofmap_start        = next_ofmap_start
                                                         , start_tensor_calc  = tile == 0
                                                         , psum_bank_id       = psum_bank_id
                                                         , first              = first
                                                        )
                        self.emit_waveop_concat_tile(tpb                  = tpb
                                                     , ifmap_tile         = ifmap_tile
                                                     , ofmap_tile         = ofmap_tile
                                                     , concat_collaterals = concat_collaterals
                                                     , ofmap_start        = next_ofmap_start
                                                     , start_tensor_calc  = tile == 0
                                                     , psum_bank_id       = psum_bank_id
                                                     , num_weights        = 1
                                                     , first              = first
                                                    )
                    if (first == True): first = False
                    move_psum_to_ofmap(tpb, ifmap_tile, ofmap_tile, psum_bank_id)
                    self.execute_postconv_tile_ops(tpb
                                                   , ofmap_tile
                                                   , ofmap_tile
                                                   , psum_bank_id)

    def emit_waveop_concat_tile (self
                                 , tpb
                                 , ifmap_tile
                                 , ofmap_tile
                                 , concat_collaterals
                                 , ofmap_start
                                 , start_tensor_calc
                                 , psum_bank_id
                                 , num_weights
                                 , first = False
                                ):
        [_, ifmap_file_param, ifmap_channel_range, ofmap_range] =\
                concat_collaterals
        c_start_idx = int(math.floor(ifmap_channel_range[1] / PEArray.NUM_ROWS))
        pewave = PEWave(ifmap_tile, c_start_idx, 0, 0)
        # Don't need to force IFMAP channel count because we need to emit
        # instruction with proper num_partitions value, so keep previously computed ifmap_channel_count
        #pewave.ifmap_channel_count = PEArray.NUM_ROWS
        ofmap_pewave = PEWave(ofmap_tile, 0, 0, 0)
        self.first_op.compute_ifmap_ofmap_pewave_info(pewave, ofmap_pewave, False)
        (prev_waveops, dram_ifmaps_waveops, new_reader_morsels) =\
                self.get_producers_for_subtile_region (
                    tpb = tpb, fmap_subtile = pewave, reader_engine = EngineEnum.PEARRAY)
        (writers, readers, dram_weights_waveops) = ([], [], [])
        # 9-10-2018
        total_size_weights =(self.first_op.weight_wave_upper_addr
                            - self.first_op.weight_wave_lower_addr
                            + self.first_op.weights_file_params.item_sz) * num_weights
        (writers, readers, dram_weights_waveops, reader_morsels) =\
                tpb.statebuffer.file_mapper.read_file_data_region(
                    tpb.waveop_stream.waveop_count
                    , tpb.waveop_stream
                    , self.first_op.weights_file_params
                    , 0  # batch_item doesn't apply for weights
                    , self.first_op.weight_wave_lower_addr
                    , total_size_weights
                    , reader_engine = EngineEnum.PEARRAY
                    , start_at_mid_part = False
                    , end_after_mid_part = True
                    , repl_multiple_of_C = 1
                )
        new_reader_morsels += reader_morsels
        psum_add = start_tensor_calc == False
        mms = self.gen_matmul_waveop(tpb\
                                     , pewave\
                                     , ofmap_pewave\
                                     , psum_add\
                                     , psum_bank_id \
                                     , dram_weights_waveops)
        prev_waveops += extract_predecessors(
                list_of_accessors_wr = writers, 
                list_of_accessors_rd = [[]],
                waveop_list          = tpb.waveop_stream)
        for i in range(len(mms)):
            #print ("mm_i name = %s"%mm_i['waveop_name'])
            if (i == 0):
                weights = dram_weights_waveops
                ifmaps = dram_ifmaps_waveops
            else:
                weights = None
                ifmaps = None
            tpb.waveop_stream.add_linked(
                  mms[i], weights + ifmaps, psum_bank_id, new_reader_morsels)
            attach_predecessors(mms[i], prev_waveops)

    def execute_concat_tile (self
                             , tpb
                             , ifmap_tile
                             , ofmap_tile
                             , concat_collaterals
                             , ofmap_start
                             , start_tensor_calc
                             , psum_bank_id
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
            #self.print_SB_addr(knode.weights_file_params)
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
          #print("weight is generated")
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
        tpb.pearray.wave_fp16_mm(packed_ifmap\
                                 , packed_weights\
                                 , psum_bank_id
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

        is_const_weights = not self.conv_op.weights_file_params.is_dynamic_weights

        # start tensor computation by clearing psum bank
        psum_add = False                               

        # wave loop ordering scheme: nmhwcRS
        n_id = batch_item // self.conv_op.Tn
        if True:
            for m_id in range(self.conv_op.m):
                for h_id in range(self.conv_op.h):
                    for w_id in range(self.conv_op.w):
                        tile_id = (n_id, m_id, h_id, w_id, self.conv_op.n, self.conv_op.m, self.conv_op.h, self.conv_op.w)
                        ifmap_tile = Tile(tile_id, self.conv_op.ifmaps_file_params, self.conv_op.Tn, is_pe_input=True)
                        ofmap_tile = Tile(tile_id, self.last_op.ofmaps_file_params, self.last_op.Tn, is_pe_input=False)
                        self.conv_op.compute_ifmap_ofmap_tile_info(ifmap_tile, ofmap_tile, self.conv_op.is_conv_transpose)
                        # Pick psum bank to use, and advance pointer to new bank (cycle through 4 banks), 
                        psum_bank_id = tpb.pearray.use_psum_bank_and_adv_ptr(dst_is_psum = True)
                        # loops for constructing a tile
                        for c_id in range(self.conv_op.c):
                            r_id = 0
                            s_id = 0
                            remaining_filter_elems = self.conv_op.RS
                            old_num_waveops = len(tpb.waveop_stream)
                            while r_id < self.conv_op.weights_file_params.file_dims.R:
                                while s_id < self.conv_op.weights_file_params.file_dims.S:
                                    ifmap_pewave = PEWave(ifmap_tile, c_id, r_id, s_id)
                                    ofmap_pewave = PEWave(ofmap_tile, c_id, r_id, s_id)
                                    # execute PEArray matrix multiply, and add to PSUM after first wave
                                    repl_multiple_per_wave = 1
                                    if self.conv_op.repl_multiple_of_C > 1:
                                        repl_multiple_per_wave = min(remaining_filter_elems, self.conv_op.repl_multiple_of_C)
                                        remaining_filter_elems -= self.conv_op.repl_multiple_of_C
                                    if (self.execute_matmul_waveop(tpb, ifmap_pewave, ofmap_pewave, inputs, weights, psum_add, psum_bank_id, repl_multiple_per_wave, is_const_weights)):
                                        psum_add = True
                                    s_id += self.conv_op.repl_multiple_of_C
                                r_id += s_id//self.conv_op.S
                                s_id = s_id%self.conv_op.S
                            if self.args.uint8_performance_mode:
                                tpb.waveop_stream[old_num_waveops:] = reorder_waveops_uint8_perf(self.conv_op, tpb.waveop_stream[old_num_waveops:])

                        # tile is done                                   
                        # go through the remaining operations, using ofmap_tile as ifmap_tile (TODO: compute new shapes per operation)
                        psum_temp = self.execute_postconv_tile_ops(tpb, ofmap_tile, ofmap_tile, psum_bank_id)
                        if self.conv_op.is_conv_transpose:
                            ofmap_tile.set_tile_data_in_file(psum_temp)
                        else:
                            # TODO: to simplify fix regular convolution so that we use cropped tile (like conv transpose)
                            # instead of padded tile
                            ofmap_tile.set_padded_tile_data_in_file(psum_temp)
                        #waveop = tpb.waveop_stream.last_psum_waveop[psum_bank_id]
                        #self.mark_producers_for_subtile_region(tpb, ofmap_tile.make_pewave(), waveop)
                        psum_add = False
