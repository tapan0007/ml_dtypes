import json
import os
import math
import re
import numpy as np
import copy
import argparse
import inspect
from layeropt_utils import CircbufPtrs 
from layeropt_utils import ShapeDims
from layeropt_utils import FileParams
from layeropt_utils import FileMapper
from layeropt_batch_recursion import BatchMachine
from skimage.util.shape import view_as_windows
from graphviz import Digraph
from enum import Enum

import sys
sys.path.insert(0, os.environ["KAENA_PATH"] + "/compiler/tffe")
from NpUtils import NpUtils as npu

DEBUG_LEVEL_DEFAULT=2

#np.set_printoptions(precision=14)

#kgraph_file = os.environ['KAENA_PATH'] + "/compiler/tffe/rundir/0-1conv0/trivnet_compiler.json"

# TODO: use datatype from K-Graph to cast everything to that datatype
# TODO: multiple atoms within waveop (batching)

def ceildiv(a,b):
    return (a//b) + (a%b != 0)

def DBG_DUMP_ARRAY(msg, a):
    if (args.debug > 3): print (msg, "\n" , a)
    return a

def DBG_DUMP_PSUM_COL(msg, psum, col):
    x = psum[:, col]
    if (args.debug > 3): print (msg, "\n" , x)
    return x

##################################################################################
# Collection of statistics
class Stats:
    def __init__(self):
        self.circbuf = {}
        self.circbuf["scratch"] = CircbufStats()
        self.circbuf["ifmaps"] = CircbufStats()
        self.circbuf["weights"] = CircbufStats()
        self.circbuf["bias"] = CircbufStats()
        self.circbuf["residue"] = CircbufStats()

class CircbufStats:
    def __init__(self):
        self.sb_all_channels_memcpys_in = 0
        self.sb_all_channels_memcpys_out = 0

##################################################################################
# ID of a PE array wave
class WaveID:

    def __init__(self, n_id, m_id, h_id, w_id, c_id, r_id, s_id):
        self.format = "nmhwcrs"
        self.n_id, self.m_id, self.h_id, self.w_id = n_id, m_id, h_id, w_id
        self.c_id, self.r_id, self.s_id = c_id, r_id, s_id

    def show(self):
        return [self.n_id, self.m_id, self.h_id, self.w_id, self.c_id, self.r_id, self.s_id]

    def id_string(self):
        return "n%d_m%d_h%d_w%d_c%d_r%d_s%d"%(self.n_id, self.m_id, self.h_id, self.w_id, self.c_id, self.r_id, self.s_id)

##################################################################################
# ID of completed OFMAP tile
class TileID:

    def __init__(self, n_id, m_id, h_id, w_id, n, m, h, w):
        self.format = "nmhw"
        self.n_id, self.m_id, self.h_id, self.w_id = n_id, m_id, h_id, w_id
        self.n, self.m, self.h, self.w = n, m, h, w

    def show(self):
        return [self.n_id, self.m_id, self.h_id, self.w_id]
    
    def id_string(self):
        return "n%d_m%d_h%d_w%d"%(self.n_id, self.m_id, self.h_id, self.w_id)

##################################################################################
# PE Array properties and methods

class PEArray:

    NUM_ROWS = 128
    NUM_COLS = 64
    PSUM_NUM_BANKS = 4
    MAX_WAVE_SIZE = 256
    num_wave_fp16_mm = 0
    num_of_ops_executed = 0
    total_pearray_latency_cycles = 0
    total_pearray_wave_elems = 0
    batching_in_wave = 0

    def __init__(self):
        self.psum_buf = np.zeros((self.PSUM_NUM_BANKS, self.MAX_WAVE_SIZE, self.NUM_COLS), dtype=np.float32)
        self.Tn = 0
        self.last_psum_bank_used = 0

    def trig_tile_done(self, tile_id):
        if (args.debug > 2): print("Tile done %s"%tile_id.id_string())
        pass

    def extract_psum (self, psum_bank, start_entry, num_entries):
        assert(start_entry < self.MAX_WAVE_SIZE)
        #assert((start_entry+num_entries) < self.MAX_WAVE_SIZE)
        return self.psum_buf[psum_bank, start_entry:start_entry+num_entries, :]

    def write_psum (self, psum_bank, start_entry, num_entries, psum_temp):
        assert(start_entry < self.MAX_WAVE_SIZE)
        #assert((start_entry+num_entries) < self.MAX_WAVE_SIZE)
        self.psum_buf[psum_bank, start_entry:start_entry+num_entries, :] = psum_temp

    # Do wave fp16->fp32 matrix-multiply        
    #   packed_ifmaps: must be 256x128 matrix, float16
    #   packet_weights: must be 128x64 matrix, float16
    #   psum_bank: the PSUM bank number to write result to
    #   psum_add: if True, add to PSUM value in buffer; if False, replace with new value
    def wave_fp16_mm(self, packed_ifmaps, packet_weights, psum_bank, psum_add):
        assert (packed_ifmaps.shape == (self.MAX_WAVE_SIZE, self.NUM_ROWS))
        assert (packet_weights.shape == (self.NUM_ROWS, self.NUM_COLS))
        assert (psum_bank < self.PSUM_NUM_BANKS)
        a = packed_ifmaps.astype(np.float32)
        b = packet_weights.astype(np.float32)
        #print(a.dtype, b.dtype, self.psum_buf.dtype)
        self.matmul_result = np.matmul(a, b)
        if (psum_add):
            self.psum_buf[psum_bank] += self.matmul_result
        else:            
            self.psum_buf[psum_bank] = self.matmul_result
        self.num_wave_fp16_mm += 1

##################################################################################
# Pooling properties and methods
class Pool:

    def wait_tile_done(self, tile_id):
        pass

    def resadd(self, array_a, array_b):
        return array_a + array_b

    def multiply(self, array_a, array_b):
        return array_a * array_b

    def pool(self, type, in_array, stride, pool_window_size, Tn, ifmap_tilex_sz, ifmap_tiley_sz, ofmap_tilex_sz, ofmap_tiley_sz):
        num_cols = in_array.shape[1]
        # view_as_windows needs in_array to be in the same dimension as window_shape
        # need to make sure the third dimension of stride_shape to be '1' since that is the column direction
        #print("ifmap_tilex_sz ", ifmap_tilex_sz, " ifmap_tiley_sz ", ifmap_tiley_sz)
        input_tilex_with_pad = ofmap_tilex_sz * stride + pool_window_size - stride
        input_tiley_with_pad = ofmap_tiley_sz * stride + pool_window_size - stride
        input_tile_with_pad_sz = input_tilex_with_pad*input_tiley_with_pad
        tile_array = np.empty((input_tiley_with_pad, input_tilex_with_pad))
        tile_array[:] = -np.inf  # set all padding values to -inf to allow only actual tile values to be analyzed
        ifmap_tile_sz = ifmap_tilex_sz*ifmap_tiley_sz
        ofmap_tile_sz = ofmap_tilex_sz*ofmap_tiley_sz
        pool_result = np.zeros((ofmap_tile_sz * Tn, num_cols))
        for j in range(Tn):
            for i in range(num_cols):
                tile_array[0:ifmap_tiley_sz, 0:ifmap_tilex_sz] = in_array[j*ifmap_tile_sz : (j+1)*ifmap_tile_sz, i].reshape(ifmap_tiley_sz, ifmap_tilex_sz) # ignoring Tn for now
                window_shape = (pool_window_size, pool_window_size)
                stride_shape = (stride, stride)
                pool_result_temp = view_as_windows(tile_array, window_shape, stride_shape)
                if (type == "MaxPool"):
                    pool_result[j*ofmap_tile_sz : (j+1)*ofmap_tile_sz, i] = pool_result_temp.max(axis=(2,3)).reshape(-1)
                elif (type == "AvgPool"):                    
                    pool_result[j*ofmap_tile_sz : (j+1)*ofmap_tile_sz, i] = pool_result_temp.mean(axis=(2,3)).reshape(-1)
                else:                    
                    print("ERROR: unknown type %s Pool.pool"%type)
                    exit(-1)
        return pool_result

    def reciprocate(self, in_array, num_active_channels):
        reciprocate_result = np.zeros(in_array.shape)
        reciprocate_result[:, 0:num_active_channels] = 1/in_array[:, 0:num_active_channels]
        return reciprocate_result

    def scale(self, in_array, scale_value):
        return in_array*scale_value

##################################################################################
# Bias-Add and Activate properties and methods
class BiasAddAct:

    def wait_tile_done(self, tile_id):
        pass

    def biasadd(self, in_array, bias_array):
        return in_array + bias_array

    def act(self, type, in_array):
        if (type == 'Relu'):
            return self.__relu(in_array)
        elif (type == 'Sigmoid'):
            return 1/(1 + np.exp(-in_array))
        elif (type == 'Exp'):
            return np.exp(in_array)
        elif (type == 'Tanh'):
            return np.tanh(in_array)
        else:
            print("ERROR BiasAddAct.act: unrecognized activation type %s"%type)
            exit(-1)

    def __relu(self, in_array):
        return np.maximum(np.zeros(in_array.shape, dtype=in_array.dtype), in_array)

##################################################################################
# State buffer memory manager
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
        self.next_bias_file_start = 0
        self.next_weights_file_start = 0
        self.printed_map_trace_header = False

##################################################################################
# Neural network node, containing data read from JSON
class KNode:
    def __init__(self, parent, data, item_sz, data_type, node_number):
        self.prev = []
        self.next = []
        self.parent = parent
        self.data = data
        self.psum_bank_dst = 0
        self.item_sz = item_sz
        self.data_type = data_type
        self.ofmap_wave_total_elems = 0
        self.node_number = node_number
        self.stridedslice_chan_offset = 0
        self.unstack_h_offset = 0
        self.result_file = None
        self.pool_window_y = 1
        self.pool_window_x = 1
        self.stride_y = 1
        self.stride_x = 1
        self.eigenlib_offset_x = 0
        self.eigenlib_offset_y = 0
        self.is_nop = False
        self.is_placeholder = False
        self.is_input = False
        self.is_output = False
        self.is_const = False
        self.is_join = False
        self.is_fork = False
        self.residue_index = -1
        self.src_is_psum = True
        self.dst_is_psum = True
        self.src_circbuf = None
        self.dst_circbuf = None
        self.ifmaps_file_params = None
        self.ofmaps_file_params = None
        self.weights_file_params = None
        self.bias_file_params = None
        self.fused_op = None
        self.replicate_multiple = 1

    def add_prev(self, prev_node):
        self.prev.append(prev_node)
    def add_next(self, next_node):
        if (not self.in_next(next_node)):
            self.next.append(next_node)
    def in_next(self, node):
        for i in self.next:
            if (i == node):
                return True
        return False   
    # Returns number of missing input results
    def count_missing_input_results(self):
        count = 0
        for i in self.prev:
            if (not i.is_const) and (i.result_file is None):
                count += 1
        return count                

    # set/get dest PSUM bank
    def set_psum_bank(self, dest):
        self.psum_bank_dst = dest
    def get_psum_bank(self):
        return self.psum_bank_dst

    def populate_ofmaps_file_params(self):
        layer_info = self.data
        ofmaps_shape_dims = ShapeDims(layer_info['ofmap_format'], layer_info['ofmap_shape'])            
        if self.is_placeholder:
            file_name = layer_info['ref_file']
        else:
            file_name = layer_info['ref_file'].replace(".npy", "-midout.npy")
        self.ofmaps_file_params = FileParams(
                                    self.parent.current_file_id, 
                                    file_name,
                                    ofmaps_shape_dims, 
                                    self.parent.data_type, 
                                    2048, 
                                    PEArray, 
                                    self,
                                    args)
        self.ofmaps_file_params.layer_name =  layer_info['layer_name']
        self.parent.current_file_id += 1
        self.N, self.M, self.E, self.F = self.ofmaps_file_params.get_nchw_shape()

    # populate common parameters for Conv and Pool
    def populate_common_params(self, adjust_for_pool):
        # get bias/scale info
        for i in range(len(self.prev)):
            prev_node = self.prev[i]
            # Const node indicates bias/scale values
            if prev_node.data['layer_type'] == "Const":
                bias_shape_dims = ShapeDims(prev_node.data['ofmap_format'], prev_node.data['ofmap_shape'])           
                self.bias_file_params = FileParams(
                                            self.parent.current_file_id, 
                                            prev_node.data['ref_file'], 
                                            bias_shape_dims, 
                                            self.parent.data_type, 
                                            2048, 
                                            PEArray, 
                                            self,
                                            args)
                self.bias_file_params.layer_name =  prev_node.data['layer_name']
                self.parent.current_file_id += 1
                self.bias_file_params.load_file()
                del self.prev[i]
                break
        # get output shape from current layer's data
        self.populate_ofmaps_file_params()
        # get input shape from previous layer's data
        input_layer = None
        prev_index = 0
        if self.is_join:
            prev_index = 1-self.residue_index
            assert(self.residue_index <= 1)
        if len(self.prev) > 0: # should work for 2-input cases (like ResAdd/Multiply)
            #raise RuntimeError("no more input to choose from when trying to decided the main input for layer %s"%(self.data['layer_name']))
            input_layer = self.prev[prev_index].data
            self.ifmaps_file_params = self.prev[prev_index].ofmaps_file_params
            self.prev[prev_index].ofmaps_file_params.readers_of_shared_fmap.append(self)
            self.N, self.C, self.H, self.W = self.ifmaps_file_params.get_nchw_shape()
        else:
            print("Layer % has no input"%(self.data['layer_name']))
        # since M is set to C above, but for Softmax2, we just need one output channel (sum across input channels)
        layer_info = self.data
        if (layer_info['layer_type'] == 'Softmax2'): self.M = 1
        # get padding and stride information
        if ('padding' in layer_info):            
            self.pad_north, self.pad_south = layer_info['padding'][2]
            self.pad_west, self.pad_east = layer_info['padding'][3]
        else:
            self.pad_north, self.pad_south = 0, 0
            self.pad_west, self.pad_east = 0, 0
        if ('stride' in layer_info):            
            self.stride_y = layer_info['stride'][2]
            self.stride_x = layer_info['stride'][3]
        if (args.eigenlib_stride):
            self.eigenlib_offset_x = ceildiv(self.stride_x, 2) - 1
            self.eigenlib_offset_y = ceildiv(self.stride_y, 2) - 1
        # OFMAP total areas
        self.EF = self.E * self.F
        # compute batch folding and batching within wave, Tn cannot be greater than batch size N
        self.Tn = PEArray.MAX_WAVE_SIZE // self.EF
        if (self.Tn < 1):
            self.Tn = 1
        elif (self.Tn > self.N):
            self.Tn = self.N
        # Heuristic: for simplicity, cap at 4 instead of allowing Tn to reach max of 5 for 7x7           
        if (self.Tn > 4):
            self.Tn = 4
        self.n = ceildiv(self.N, self.Tn)
        # per kaena-85, use noodle shapes for tiles
        # need to guard against small EF and build noodle tile to enable higher state buffer efficiency
        self.ofmap_full_tilex_sz = min(self.F, PEArray.MAX_WAVE_SIZE)
        self.ofmap_full_tiley_sz = min(self.E, PEArray.MAX_WAVE_SIZE // self.ofmap_full_tilex_sz)
        # kaena-202: prevent crossing atom gaps by making tiley even across full FMAP
        fmap_rows = self.E
        while (fmap_rows % 2 == 0 and fmap_rows > self.ofmap_full_tiley_sz):
            fmap_rows = fmap_rows//2
        if (fmap_rows < self.ofmap_full_tiley_sz):            
            self.ofmap_full_tiley_sz  = fmap_rows
        # If the EF is large, we need to make sure tiley is at least the same size as the pool_window
        #if ((self.EF > PEArray.MAX_WAVE_SIZE) and adjust_for_pool):
        if (adjust_for_pool and self.ofmap_full_tiley_sz < self.pool_window_y):
            self.ofmap_full_tiley_sz = min(self.E, self.pool_window_y)
            self.ofmap_full_tilex_sz = min(self.F, PEArray.MAX_WAVE_SIZE // self.ofmap_full_tiley_sz)
        self.ofmap_full_tile_sz = self.ofmap_full_tilex_sz * self.ofmap_full_tiley_sz
        self.ifmap_wave_lower_addr = [-1 for i in range(self.Tn)]
        self.ifmap_wave_upper_addr = [-1 for i in range(self.Tn)]
        # compute the IFMAP folds
        self.c = ceildiv(self.C, PEArray.NUM_ROWS)
        # compute the OFMAP folds
        self.m = ceildiv(self.M, PEArray.NUM_COLS)
        # computing the input map tiling       
        self.h, self.w, self.e, self.f = 1, 1, 1, 1
        # compute ofmap folding
        if (self.EF >= PEArray.MAX_WAVE_SIZE):
            self.e = ceildiv(self.E, self.ofmap_full_tiley_sz)
            self.f = ceildiv(self.F, self.ofmap_full_tilex_sz)
        # heigh/width folding is the same for IFMAP and OFMAP            
        self.h = self.e
        self.w = self.f
        print("Common params1 for layer %s:  N=%d, M=%d, H=%d, W=%d, C=%d, E=%d, F=%d"
                %(self.data['layer_name'], self.N, self.M, self.H, self.W, self.C, self.E, self.F))
        print("Common params2 for layer %s:  n=%d, m=%d, h=%d, w=%d, c=%d, Tn=%d"
                %(self.data['layer_name'], self.n, self.m, self.h, self.w, self.c, self.Tn))
        print("Common params3 for layer %s:  stride_x=%d, stride_y=%d, ofmap_full_tilex_sz=%d, ofmap_full_tiley_sz=%d, ofmap_full_tile_sz=%d"
                %(self.data['layer_name'], self.stride_x, self.stride_y, self.ofmap_full_tilex_sz, self.ofmap_full_tiley_sz, self.ofmap_full_tile_sz))

    # Compute Conv looping params
    def populate_conv_params(self):
        # convolution kernel shape
        layer_info = self.data
        if (layer_info['layer_type'] == 'Softmax2'):
            weights_shape_dims = ShapeDims(layer_info['ofmap_format'], layer_info['ofmap_shape'])            
            weights_file = self.data['ref_file'].replace(".npy", "-ones.npy")
            if (not args.inference):
                ones_tensor = np.ones(weights_shape_dims.shape_tuple, dtype=self.parent.data_type)
                np.save(weights_file, ones_tensor)
        else:
            weights_shape_dims = ShapeDims(layer_info['kernel_format'], layer_info['kernel_shape'])            
            weights_file = self.data['kernel_file']
        # kaena-141: replicate IFMAP a number of times.
        # The number is determined by S, multiplied by a portion of R to match r*S*C <= 128
        # In the case of 1st layer ResNet50, R=7, S=7, C=3 so R can be broken a number of ways. 
        # For now, split evenly among two waves.
        self.replicate_multiple = 1
        if args.enable_replication:
            num_replicated_waves = ceildiv(weights_shape_dims.R * weights_shape_dims.S * weights_shape_dims.C,  PEArray.NUM_ROWS)
            self.replicate_multiple = ceildiv(weights_shape_dims.R, num_replicated_waves) * weights_shape_dims.S
        self.weights_file_params = FileParams(self.parent.current_file_id, weights_file, weights_shape_dims, self.data_type, 2048, PEArray, self, args)
        self.weights_file_params.layer_name =  self.data['layer_name']
        self.parent.current_file_id += 1
        self.weights_file_params.load_file()
        self.R = self.weights_file_params.file_dims.R
        self.S = self.weights_file_params.file_dims.S
        print("Conv params for layer %s: R=%d, S=%d, replicate_multiple=%d"%(self.data['layer_name'], self.weights_file_params.file_dims.R, self.weights_file_params.file_dims.S, self.replicate_multiple))

    # Compute pooling params
    def populate_pooling_params(self):
        # are the dimensions from layer info correct?
        layer_info = self.data
        self.pool_window_y = layer_info['kernel_shape'][2]
        self.pool_window_x = layer_info['kernel_shape'][3]
        self.stride_y = layer_info['stride'][2]
        self.stride_x = layer_info['stride'][3]
        if (args.eigenlib_stride):
            self.eigenlib_offset_x = ceildiv(self.stride_x, 2) - 1
            self.eigenlib_offset_y = ceildiv(self.stride_y, 2) - 1
        print("Pooling params for layer %s: pool_window_x=%d, pool_window_y=%d, stride_x=%d, stride_y=%d"
                %(self.data['layer_name'], self.pool_window_x, self.pool_window_y, self.stride_x, self.stride_y))

    # Recompute conv tile params due to fused pooling
    def recompute_conv_params(self, pool_window_x, pool_window_y):        
        # For pooling using PSUM (fused), max tile size must be a multiple of pooling window
        self.ofmap_full_tiley_sz = (self.ofmap_full_tiley_sz // pool_window_y) * pool_window_y
        self.ofmap_full_tilex_sz = (self.ofmap_full_tilex_sz // pool_window_x) * pool_window_x
        self.ofmap_full_tile_sz = self.ofmap_full_tilex_sz * self.ofmap_full_tiley_sz
        print("Recomputed Conv params due to fused pooling: pool_window_x=%d, pool_window_y=%d, ofmap_full_tilex_sz=%d, ofmap_full_tiley_sz=%d"
                %(pool_window_x, pool_window_y, self.ofmap_full_tilex_sz, self.ofmap_full_tiley_sz))

    # compute output tile info
    def compute_ofmap_tile_info(self, tile_id):        
        self.ofmap_tile_x_start = tile_id.w_id * self.ofmap_full_tilex_sz
        self.ofmap_tile_y_start = tile_id.h_id * self.ofmap_full_tiley_sz
        self.ofmap_cropped_tile_height = self.ofmap_full_tiley_sz
        self.ofmap_cropped_tile_width = self.ofmap_full_tilex_sz
        if ((tile_id.h_id+1) * self.ofmap_full_tiley_sz > self.E):
            self.ofmap_cropped_tile_height = self.E - self.ofmap_tile_y_start
        if ((tile_id.w_id+1) * self.ofmap_full_tilex_sz > self.F):
            self.ofmap_cropped_tile_width = self.F - self.ofmap_tile_x_start
        self.tile_size = self.ofmap_cropped_tile_height * self.ofmap_cropped_tile_width

        # number of OFMAPs for this tile 
        pe_col_start = tile_id.m_id * PEArray.NUM_COLS
        pe_col_stop = min(self.M, pe_col_start + PEArray.NUM_COLS)
        self.ofmap_count = pe_col_stop - pe_col_start

        # compute the address bounds for OFMAP tile within OFMAPs tensor
        # TODO: for Tn>1, need to have multiple bounds for each batch item
        # NCHW
        self.ofmap_tile_lower_addr = []
        self.ofmap_tile_upper_addr = []
        self.ifmap_tile_lower_addr = []
        self.ifmap_tile_upper_addr = []
        for z in range(self.Tn):
            self.ofmap_tile_lower_addr.append(self.ofmaps_file_params.ravel_nchw(
                                                tile_id.n_id * self.Tn + z, 
                                                    tile_id.m_id//2 * PEArray.NUM_ROWS,
                                                    self.ofmap_tile_y_start, 
                                                    self.ofmap_tile_x_start))
            # NCHW
            self.ofmap_tile_upper_addr.append(self.ofmaps_file_params.ravel_nchw(
                                                tile_id.n_id * self.Tn + z, 
                                                    tile_id.m_id//2 * PEArray.NUM_ROWS,
                                                    self.ofmap_tile_y_start + self.ofmap_cropped_tile_height - 1, 
                                                    self.ofmap_tile_x_start + self.ofmap_cropped_tile_width - 1))

            # compute the address bounds for IFMAP tile within IFMAPs tensor
            # NCHW
            ifmap_tile_lower_coordx = self.ofmap_tile_x_start * self.stride_x
            ifmap_tile_lower_coordy = self.ofmap_tile_y_start * self.stride_y

            if not self.src_is_psum:
                self.ifmap_tile_lower_addr.append(self.ifmaps_file_params.ravel_nchw(
                                                tile_id.n_id * self.Tn + z, 
                                                    0,
                                                    ifmap_tile_lower_coordy,
                                                    ifmap_tile_lower_coordx))

            ifmap_tile_upper_coordx = ifmap_tile_lower_coordx + self.ofmap_cropped_tile_width * self.stride_x - 1
            ifmap_tile_upper_coordy = ifmap_tile_lower_coordy + self.ofmap_cropped_tile_height * self.stride_y - 1
            if (ifmap_tile_upper_coordx > self.W-1):
                ifmap_tile_upper_coordx = self.W-1
            if (ifmap_tile_upper_coordy > self.H-1):
                ifmap_tile_upper_coordy = self.H-1
            # NCHW
            if not self.src_is_psum:
                self.ifmap_tile_upper_addr.append(self.ifmaps_file_params.ravel_nchw(
                                                tile_id.n_id * self.Tn + z, 
                                                    (self.c-1) * PEArray.NUM_ROWS,
                                                    ifmap_tile_upper_coordy,
                                                    ifmap_tile_upper_coordx))

        self.ifmap_cropped_tile_width = ifmap_tile_upper_coordx - ifmap_tile_lower_coordx + 1
        self.ifmap_cropped_tile_height = ifmap_tile_upper_coordy - ifmap_tile_lower_coordy + 1

    def compute_tile_weight_bounds (self, weights, tile_id):        
        # Address bounds of weights used for tile
        pe_col_start = tile_id.m_id * PEArray.NUM_COLS
        pe_col_stop = min(self.M, pe_col_start + PEArray.NUM_COLS)
        self.weight_tile_lower_addr = self.weights_file_params.ravel_crsm (0, 0, 0, pe_col_start)
        self.weight_tile_upper_addr = self.weights_file_params.ravel_crsm (self.weights_file_params.file_dims.C-1, self.weights_file_params.file_dims.R-1, self.weights_file_params.file_dims.S-1, pe_col_stop-1)

    # Pack the IFMAPs in columns to create a PE-Array IFMAPs input for a particular wave number
    #   ifmaps: IFMAPs in NCHW format
    #   wave_id: current wave ID, [n_id, m_id, h_id, w_id, c_id, r_id, s_id]
    #   layer_type: 'conv' or 'MaxPool'; can be extended to handle other layer types
    #   return: a 256x128 array
    def pack_wave_ifmaps(self, ifmaps, wave_id, replicate_multiple, for_softmax):
        # If we are not doing convolution (aka pooling), set out_array_dim_y to be PEArray.NUM_COLS to match pooling/activation engines dimension
        if (for_softmax):
            out_array_dim_y = PEArray.NUM_COLS
        else:            
            out_array_dim_y = PEArray.NUM_ROWS
        fmap_folding_idx = wave_id.c_id
        fmap_total_count = self.C
        out_array = np.zeros((PEArray.MAX_WAVE_SIZE, out_array_dim_y))
        # remember to extract IFMAPs starting at r_id, s_id (which should be zero for non-conv op)
        # also need to add zeros for padding
        self.ifmap_wave_lower_addr = [-1 for i in range(self.Tn)]
        self.ifmap_wave_upper_addr = [-1 for i in range(self.Tn)]
        self.ofmap_wave_lower_coordx = [0 for i in range(self.Tn)]
        self.ofmap_wave_lower_coordy = [0 for i in range(self.Tn)]
        self.ofmap_wave_upper_coordx = [0 for i in range(self.Tn)]
        self.ofmap_wave_upper_coordy = [0 for i in range(self.Tn)]
        self.psum_bank_offset = 0
        # for pooling, the "row" below actually means output columns
        pe_row_start = fmap_folding_idx * out_array_dim_y + self.stridedslice_chan_offset
        pe_row_stop = min(fmap_total_count + self.stridedslice_chan_offset, pe_row_start + out_array_dim_y)
        assert(pe_row_start < pe_row_stop)
        r_id = wave_id.r_id
        s_id = wave_id.s_id
        last_r_id = r_id
        last_s_id = s_id
        num_rows = pe_row_stop - pe_row_start
        for repl in range(replicate_multiple):
            pe_row_repl_start = num_rows * repl
            for row in range(pe_row_start, pe_row_stop):
                pe_row_offset = pe_row_repl_start + row - pe_row_start
                for z in range(self.Tn):
                    batch_id = (wave_id.n_id * self.Tn) + z
                    for x in range(self.ofmap_full_tilex_sz):
                        for y in range(self.ofmap_full_tiley_sz):
                            ifmap_tilex = (wave_id.w_id * self.ofmap_full_tilex_sz + x) * self.stride_x + self.eigenlib_offset_x + s_id - self.pad_west
                            ifmap_tiley = ((wave_id.h_id + self.unstack_h_offset)* self.ofmap_full_tiley_sz + y) * self.stride_y + self.eigenlib_offset_y + r_id - self.pad_north
                            last_r_id = r_id
                            last_s_id = s_id
                            ifmap_addr = z * self.ofmap_full_tile_sz + y * self.ofmap_full_tilex_sz + x
                            if (ifmap_tilex < 0 or ifmap_tilex >= self.W):
                                out_array[ifmap_addr, pe_row_offset] = 0
                            elif (ifmap_tiley < 0 or ifmap_tiley >= self.H):
                                out_array[ifmap_addr, pe_row_offset] = 0
                            else:
                                if (args.nname == "lm"):
                                    out_array[ifmap_addr, pe_row_offset] = self.ifmaps_file_params.elem_nchw(batch_id, row, ifmap_tiley, ifmap_tilex)
                                else:                                    
                                    out_array[ifmap_addr, pe_row_offset] = ifmaps[batch_id, row, ifmap_tiley, ifmap_tilex]
                                # Check bounds of actual pixels within the original ifmaps for the first ifmap (which should reside in first SB partition)
                                # TODO: check how N/C are arrange in memory; batching within waves may cause different atoms to be accessed by same wave
                                # TODO: for Tn>1, need to have multiple bounds for each batch item
                                if (repl == 0 and row == pe_row_start):                                
                                    # NCHW
                                    self.ifmap_wave_upper_addr[z] = self.ifmaps_file_params.ravel_nchw(batch_id, row, ifmap_tiley, ifmap_tilex)
                                    self.ofmap_wave_upper_coordx[z] = x
                                    self.ofmap_wave_upper_coordy[z] = y
                                    if (self.ifmap_wave_lower_addr[z] < 0):
                                        self.ifmap_wave_lower_addr[z] = self.ifmap_wave_upper_addr[z]
                                        self.ofmap_wave_lower_coordx[z] = x
                                        self.ofmap_wave_lower_coordy[z] = y
                                        self.psum_bank_offset = (y * self.ofmap_full_tilex_sz + x)
                            #print("x %d y %d ifmap_tilex %d ifmap_tiley %d wave_lower_coordx %d wave_upper_coordy %d wave_upper_coordx %d wave_upper_coordy %d"%(x, y, ifmap_tilex, ifmap_tiley, self.ofmap_wave_lower_coordx, self.ofmap_wave_lower_coordy, self.ofmap_wave_upper_coordx, self.ofmap_wave_upper_coordy))                                    
                            #if (args.debug > 3): print("DBG: pack_wave_ifmaps for wave %s batch_id %d x %d y %d r_id %d s_id %d ifmap_tilex %d ifmap_tiley %d wave_lower_coordx %d wave_upper_coordy %d wave_upper_coordx %d wave_upper_coordy %d"%(wave_id.show(), batch_id, x, y, r_id, s_id, ifmap_tilex, ifmap_tiley, self.ofmap_wave_lower_coordx[0], self.ofmap_wave_lower_coordy[0], self.ofmap_wave_upper_coordx[0], self.ofmap_wave_upper_coordy[0]))                                    
            s_id += 1
            if (s_id >= self.S): 
                r_id += 1
                s_id = 0
                if (r_id >= self.R): break
        self.ofmap_wave_width  = self.ofmap_wave_upper_coordx[0] - self.ofmap_wave_lower_coordx[0] + 1
        self.ofmap_wave_height = self.ofmap_wave_upper_coordy[0] - self.ofmap_wave_lower_coordy[0] + 1
        self.ofmap_wave_elems = self.ofmap_wave_width * self.ofmap_wave_height
        self.ofmap_wave_total_elems += self.ofmap_wave_elems
        return out_array

    def pack_wave_ifmaps_unfused_pooling (self, ifmaps, wave_id):
        # If we are not doing convolution (aka pooling), set out_array_dim_y to be PEArray.NUM_COLS to match pooling/activation engines dimension
        out_array_dim_y = PEArray.NUM_COLS
        fmap_folding_idx = wave_id.m_id
        fmap_total_count = self.M
        out_array = np.zeros((PEArray.MAX_WAVE_SIZE * self.stride_x * self.stride_y * 2, out_array_dim_y))
        # remember to extract IFMAPs starting at r_id, s_id (which should be zero for non-conv op)
        # also need to add zeros for padding
        self.ifmap_wave_lower_addr = [-1 for i in range(self.Tn)]
        self.ifmap_wave_upper_addr = [-1 for i in range(self.Tn)]
        self.ifmap_wave_lower_coordx = [0 for i in range(self.Tn)]
        self.ifmap_wave_lower_coordy = [0 for i in range(self.Tn)]
        self.ifmap_wave_upper_coordx = [0 for i in range(self.Tn)]
        self.ifmap_wave_upper_coordy = [0 for i in range(self.Tn)]
        self.psum_bank_offset = 0
        # for pooling, the "row" below actually means output columns
        pe_row_start = fmap_folding_idx * out_array_dim_y + self.stridedslice_chan_offset
        pe_row_stop = min(fmap_total_count + self.stridedslice_chan_offset, pe_row_start + out_array_dim_y)
        assert(pe_row_start < pe_row_stop)
        # make use of the upper 64 partitions by resetting address back to 128-channels boundary
        row_adjust = (wave_id.m_id%2)*PEArray.NUM_COLS
        self.ifmap_count = pe_row_stop - (pe_row_start - row_adjust)
        for row in range(pe_row_start, pe_row_stop):
            ifmap = ifmaps[:, row]  # NCHW
            pe_row_offset = row - pe_row_start
            for z in range(self.Tn):
                batch_id = (wave_id.n_id * self.Tn) + z
                self.ifmap_wave_lower_coordx[z] = (wave_id.w_id * self.ofmap_full_tilex_sz) * self.stride_x 
                self.ifmap_wave_lower_coordy[z] = ((wave_id.h_id + self.unstack_h_offset) * self.ofmap_full_tiley_sz) * self.stride_y
                self.ifmap_wave_upper_coordx[z] = ((wave_id.w_id+1) * self.ofmap_full_tilex_sz) * self.stride_x + (self.pool_window_x - self.stride_x) - 1
                self.ifmap_wave_upper_coordy[z] = ((wave_id.h_id + self.unstack_h_offset +1) * self.ofmap_full_tiley_sz) * self.stride_y + (self.pool_window_y - self.stride_y) - 1 
                if (self.ifmap_wave_upper_coordx[z] > self.W-1):
                    self.ifmap_wave_upper_coordx[z] = self.W-1
                if (self.ifmap_wave_upper_coordy[z] > self.H-1):
                    self.ifmap_wave_upper_coordy[z] = self.H-1
                row_temp = ifmap[batch_id,
                                 self.ifmap_wave_lower_coordy[z]:self.ifmap_wave_upper_coordy[z]+1,
                                 self.ifmap_wave_lower_coordx[z]:self.ifmap_wave_upper_coordx[z]+1].flatten()
                out_array[z * len(row_temp) : (z+1) * len(row_temp), pe_row_offset] = row_temp
                if (row == pe_row_start):                               
                    # NCHW
                    self.ifmap_wave_lower_addr[z] = self.ifmaps_file_params.ravel_nchw(
                                                    batch_id, row - row_adjust, self.ifmap_wave_lower_coordy[z], self.ifmap_wave_lower_coordx[z])
                    self.ifmap_wave_upper_addr[z] = self.ifmaps_file_params.ravel_nchw(
                                                    batch_id, row - row_adjust, self.ifmap_wave_upper_coordy[z], self.ifmap_wave_upper_coordx[z])
        #print(self.ifmap_wave_lower_coordx[0], self.ifmap_wave_lower_coordy[0], self.ifmap_wave_upper_coordx[0], self.ifmap_wave_upper_coordy[0])                    
        return out_array

    # Pack the conv weights in columns to create a PE-Array weights array for a particular wave number
    #   weights: conv weights in CRSM format
    #   wave_id: current wave ID, [n_id, m_id, h_id, w_id, c_id, r_id, s_id]
    #   return: a 128x64 array
    def pack_wave_conv_weights(self, weights, wave_id, replicate_multiple):
        out_array = np.zeros((PEArray.NUM_ROWS, PEArray.NUM_COLS))
        pe_row_start = wave_id.c_id * PEArray.NUM_ROWS
        pe_row_stop = min(self.C, pe_row_start + PEArray.NUM_ROWS)
        pe_col_start = wave_id.m_id * PEArray.NUM_COLS
        pe_col_stop = min(self.M, pe_col_start + PEArray.NUM_COLS)
        self.ifmap_count = pe_row_stop - pe_row_start
        self.ofmap_count = pe_col_stop - pe_col_start
        r_id = wave_id.r_id
        s_id = wave_id.s_id
        last_r_id = r_id
        last_s_id = s_id
        num_rows = pe_row_stop - pe_row_start
        for repl in range(replicate_multiple):
            pe_row_repl_start = num_rows * repl
            for row in range(pe_row_start, pe_row_stop):
                pe_row_offset = pe_row_repl_start + row - pe_row_start
                for col in range(pe_col_start, pe_col_stop):
                    out_array[pe_row_offset, col - pe_col_start] = weights[row, r_id, s_id, col] # CRSM
                    last_r_id = r_id
                    last_s_id = s_id
            if (args.debug > 2): print("DBG: pack_wave_conv_weights for wave %s r_id %d s_id %d"%(wave_id.show(), r_id, s_id))
            s_id += 1
            if (s_id >= self.S): 
                r_id += 1
                s_id = 0
                if (r_id >= self.R): break

        # use repl+1 here for replication multiple (for example, R=7 is broken into replicate_multiple of 4 for first wave and replicate_multiple of 3 for second wave)
        self.ifmap_count = self.ifmap_count * (repl + 1)

        self.weight_wave_lower_addr = self.weights_file_params.ravel_crsm(
                                            pe_row_start, wave_id.r_id, wave_id.s_id, pe_col_start)
        self.weight_wave_upper_addr = self.weights_file_params.ravel_crsm(
                                            pe_row_start, last_r_id, last_s_id, pe_col_stop-1)
        return out_array


##################################################################################
# Stream of waveops: consist of list of waveops that are fused (communicate through PSUM buffers)
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
            while (item_name in self.waveop_name_set):
                new_name = item_name + "__" + str(i)
                print("WARNING: waveop_name %s exists; so modifying name to %s before adding waveop to stream"%(item_name, new_name))
                item_name = new_name
                i += 1
        item['waveop_name'] = item_name
        self.waveop_name_set.add(item['waveop_name'])                
        self.append(item)
        self.waveop_count += 1
        if (item['waveop_type'] != 'SBAtomFile'):
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
        waveop['previous_waveops'] += input_list
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

##################################################################################
# FusedOpList: a list of fused-op
class FusedOpList(list):
    def __init__(self):
        pass

##################################################################################
# FusedOp: consist of list of K-Nodes that are fused (communicate through PSUM buffers)
class FusedOp(list):

    def __init__(self, out_data_type, fused_op_id):
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
        self.begin_of_first_leg = False
        self.end_of_first_leg = False
        self.ofmap_is_for_join = False
        self.residue_in_scratch = False
        # "pairup" is the region or boundary where OFMAP shrinks by 1/4 and partial-batch count doubles.
        self.partial_batch_pairup = False
        # "pre-pairup" is the region just before paired-up where OFMAP shrinks by 1/4 but partial-batch count has not doubled.
        self.partial_batch_pre_pairup = False
        self.next_batch_count = 1
        self.current_batch_count = 1

    # Add operation to list of fused operations.
    # Returns True if successful; False if cannot add (i.e. Pool cannot be fused)
    def add(self, op):
        if (args.debug > 2):
            print("DBG: adding layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
        if (op.data['layer_type'] == 'AvgPool' or op.data['layer_type'] == 'MaxPool'):
            op.populate_pooling_params()
            # If not first op, pool cannot be fused with previous op if stride != pooling window
            if (len(self) != 0
                    and (op.stride_x != op.pool_window_x 
                        or op.stride_y != op.pool_window_y
                        or op.stride_x > 1 
                        or op.stride_y > 1)):
                if (args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            elif (self.has_pool):
                if (args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            else:
                self.pool_op = op
                self.has_pool = True
        elif (op.data['layer_type'] == 'Conv' or op.data['layer_type'] == 'MatMul' or op.data['layer_type'] == 'Softmax2'):
            if (len(self) != 0):
                if (args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            elif (self.has_conv):
                if (args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            else:
                op.populate_conv_params()
                self.conv_op = op
                self.has_conv = True
        elif (op.is_join):
            self.ofmap_is_for_join = True
            if (self.has_join):
                if (args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            # cannot fuse join if there's more than one missing result
            elif op.count_missing_input_results() > 1:
                return False
            else:
                self.has_join = True
                self.join_op = op
                # set the residue selection index to the other input
                if len(self) > 1:
                    self.join_op.residue_index = 1 if op.prev[0] == self[-1] else 0
                else:
                    raise RuntimeError("Please implement unfused join, where both inputs need to be sourced from SB")
        elif (op.data['layer_type'] == 'BiasAdd'):
            if (self.has_biasadd):
                if (args.debug > 2):
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
        print("DBG: fused_ops collected: (begin_of_first_leg %d, end_of_first_leg %d, ofmap_is_for_join %d, partial_batch_pre_pairup %d, partial_batch_pairup %d, residue_in_scratch %d)"%(self.begin_of_first_leg,self.end_of_first_leg, self.ofmap_is_for_join, self.partial_batch_pre_pairup, self.partial_batch_pairup, self.residue_in_scratch))
        for i in self:
            print("    ", i.data["layer_type"],":",i.data["layer_name"], )

    def execute(self, batch_item):
        # select SB region sizing index
        sb_size_set_index = tpb.statebuffer.batcher.sb_size_set_index[(self.current_batch_count, self.partial_batch_pre_pairup)]

        # Bias region
        bias_region_start_addr = 0
        bias_region_sz = tpb.statebuffer.batcher.item_sz
        bias_file_start_addr = 0
        bias_file_sz = tpb.statebuffer.batcher.item_sz
        if self.has_biasadd:
            bias_region_start_addr = 0
            bias_region_sz = tpb.statebuffer.batcher.sb_bias_sz[sb_size_set_index]
            bias_file_start_addr = tpb.statebuffer.next_bias_file_start
            bias_file_sz = self.biasadd_op.bias_file_params.tot_partition_usage_sz
            if (bias_file_start_addr + bias_file_sz) > bias_region_sz:
                bias_file_start_addr = 0
            tpb.statebuffer.file_mapper.map_file(self.biasadd_op.bias_file_params, bias_file_start_addr, wrap_around=False, region_sz=bias_file_sz)
            # in case that file is already mapped, keep the mapped values
            if bias_file_start_addr == self.biasadd_op.bias_file_params.mapped_params.start_addr:
                tpb.statebuffer.next_bias_file_start = bias_file_start_addr + bias_file_sz
            else:                
                bias_file_start_addr = self.biasadd_op.bias_file_params.mapped_params.start_addr

        # Input/residue uses upper portion of the shared space
        if self.first_op.is_input:
            ifmaps_region_start_addr =  tpb.statebuffer.batcher.sb_bias_sz[0] \
                                      + tpb.statebuffer.batcher.sb_partialbatch_start[1] 
            ifmaps_region_sz = 56*56*tpb.statebuffer.batcher.item_sz
            # for first IFMAP, use the residue size, which is roughly equal to 3 chunks of 224x4 input tiles
            tpb.statebuffer.file_mapper.map_file(self.first_op.ifmaps_file_params, ifmaps_region_start_addr, wrap_around=True, region_sz=ifmaps_region_sz)
            # obtain the adjusted region size
            ifmaps_region_sz = self.first_op.ifmaps_file_params.mapped_params.region_sz
            # should be the same even if file was already mapped
            assert(ifmaps_region_start_addr == self.first_op.ifmaps_file_params.mapped_params.start_addr)
        else:            
            ifmaps_region_start_addr = self.first_op.ifmaps_file_params.mapped_params.start_addr
            ifmaps_region_sz  = self.first_op.ifmaps_file_params.mapped_params.region_sz
        # Individual IFMAP info
        single_ifmap_start = ifmaps_region_start_addr + (batch_item % self.current_batch_count) * self.first_op.ifmaps_file_params.batch_item_partition_usage_sz
        single_ifmap_sz = self.first_op.ifmaps_file_params.batch_item_partition_usage_sz

        # Join for partial-batch region
        ofmap_batch_count = self.current_batch_count
        # "pairup" is the region or boundary where OFMAP shrinks by 1/4 and partial-batch count doubles.
        if self.partial_batch_pairup:
            ofmap_batch_count = self.next_batch_count
            sb_size_set_index = tpb.statebuffer.batcher.sb_size_set_index[(ofmap_batch_count, False)]
        if ((self.last_op.is_fork or self.ofmap_is_for_join) != self.residue_in_scratch):
            # special case for stage after MaxPool: use scratch space for OFMAP instead of residue space
            ofmaps_region_sz = ofmap_batch_count * self.last_op.ofmaps_file_params.batch_item_partition_usage_sz
            ofmaps_region_start_addr =   tpb.statebuffer.batcher.sb_bias_sz[sb_size_set_index] \
                                       + tpb.statebuffer.batcher.sb_partialbatch_start[ofmap_batch_count]
        # Scratch (OFMAP)
        else:
            ofmaps_region_sz = tpb.statebuffer.batcher.sb_scratch_sz[sb_size_set_index]
            ofmaps_region_start_addr = tpb.statebuffer.SB_PARTITION_SZ - ofmaps_region_sz

        # If OFMAP region overlaps IFMAP region, and numober of channels > 64 or stride > 1, offset it (to lower address) by OFMAP * Tn            
        if ofmaps_region_start_addr >= ifmaps_region_start_addr \
                and ofmaps_region_start_addr < ifmaps_region_start_addr + ifmaps_region_sz:
            if (self.last_op.ofmaps_file_params.file_dims.C > 64) or (self.conv_op is not None and (self.conv_op.stride_x > 1 or self.conv_op.S > 1)):
                ofmaps_region_start_addr = ifmaps_region_start_addr - self.last_op.ofmaps_file_params.batch_item_partition_usage_sz * self.last_op.Tn                               
            # Allow modifying in place for IFMAPs which overlap the same region as OFMAPs
            self.first_op.ifmaps_file_params.mapped_params.modify_in_place = 1

        # Map the file to region and obtain adjusted region size
        tpb.statebuffer.file_mapper.map_file(self.last_op.ofmaps_file_params, ofmaps_region_start_addr, wrap_around=True, region_sz=ofmaps_region_sz)
        ofmaps_region_sz = self.last_op.ofmaps_file_params.mapped_params.region_sz
        # should be the same even if file was already mapped
        assert(ofmaps_region_start_addr == self.last_op.ofmaps_file_params.mapped_params.start_addr)

        # Individual OFMAP info
        single_ofmap_start = ofmaps_region_start_addr + (batch_item % ofmap_batch_count) * self.last_op.ofmaps_file_params.batch_item_partition_usage_sz 
        single_ofmap_sz = self.last_op.ofmaps_file_params.batch_item_partition_usage_sz

        # Weights region: remaining space after allocating for bias, residue/IFMAP, and OFMAP/scratch
        weights_region_start_addr  = 0
        weights_region_sz = tpb.statebuffer.batcher.item_sz
        weights_file_start_addr = 0
        weights_file_sz = tpb.statebuffer.batcher.item_sz 
        if self.has_conv:
            # reselect SB region sizing index based on input current_batch_count
            sb_size_set_index = tpb.statebuffer.batcher.sb_size_set_index[(self.current_batch_count, self.partial_batch_pre_pairup)]
            # right before pairup to batch count of 16, there's a jump in weights elem count, so take from partial batch space (shared space)
            weights_region_start_addr =  tpb.statebuffer.batcher.sb_bias_sz[sb_size_set_index] \
                                + tpb.statebuffer.batcher.sb_partialbatch_sz[sb_size_set_index]
            weights_region_sz = ofmaps_region_start_addr - weights_region_start_addr
            # try a different start adddress based on the last allocation                
            weights_file_start_addr = tpb.statebuffer.next_weights_file_start
            weights_file_sz = self.conv_op.weights_file_params.tot_partition_usage_sz
            if (weights_file_start_addr < weights_region_start_addr):
                weights_file_start_addr = weights_region_start_addr
            elif (weights_file_start_addr + weights_file_sz > weights_region_start_addr + weights_region_sz):
                weights_file_start_addr = weights_region_start_addr
            tpb.statebuffer.next_weights_file_start = weights_file_start_addr + weights_file_sz
            # map file to region                
            tpb.statebuffer.file_mapper.map_file(self.conv_op.weights_file_params, weights_file_start_addr, wrap_around=False, region_sz=weights_file_sz)
            # obtain the adjusted region size
            weights_region_sz = self.conv_op.weights_file_params.mapped_params.region_sz
            # also in case that file is already mapped, keep the mapped values
            weights_file_start_addr = self.conv_op.weights_file_params.mapped_params.start_addr

        # Test by reading entire IFMAP
        #num_ifmap_megachunks = ceildiv(single_ifmap_sz, ifmaps_region_sz)
        #if num_ifmap_megachunks == 0:
        #    tpb.statebuffer.file_mapper.read_file_data_region(0, self.first_op.ifmaps_file_params, batch_item, batch_item*self.first_op.ifmaps_file_params.dram_data_bytes_per_batch_item + 0, single_ifmap_sz)
        #else:
        #    # For layer 0, IFMAP is very large
        #    for i in range(num_ifmap_megachunks):
        #        megachunk_start = i*ifmaps_region_sz
        #        #megachunk_sb_start = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.first_op.ifmaps_file_params, megachunk_start)
        #        megachunk_sz = ifmaps_region_sz if (i < num_ifmap_megachunks - 1) else (single_ifmap_sz - i*ifmaps_region_sz)
        #        tpb.statebuffer.file_mapper.read_file_data_region(0, self.first_op.ifmaps_file_params, batch_item, batch_item*self.first_op.ifmaps_file_params.dram_data_bytes_per_batch_item + megachunk_start, megachunk_sz)
        # Test by writing entire OFMAP
        #tpb.statebuffer.file_mapper.write_file_data_region(0, self.last_op.ofmaps_file_params, batch_item, batch_item*self.last_op.ofmaps_file_params.dram_data_bytes_per_batch_item + 0, single_ofmap_sz)
        # Trace printout
        if (args.debug > 2 and not tpb.statebuffer.printed_map_trace_header): 
            print("SB MAP TRACE, fused op, fused op ID, batch elem, Tn, current_batch_count, next_batch_count, partial_batch_pre_pairup, partial_batch_pairup, residue_in_scratch, \
 bias file end_addr, bias region end_addr, bias region start_addr, bias file start_addr,\
 weights file end_addr, weights region end_addr, weights region start_addr, weights file start_addr,\
 ifmap single end_addr, ifmaps region end_addr, ifmaps region start_addr, ifmap single start_addr,\
 ofmap single end_addr, ofmaps region end_addr, ofmaps region start_addr, ofmap single start_addr, ifmap file, ofmap file")                    
            tpb.statebuffer.printed_map_trace_header = True
        if (args.debug > 2): print("SB MAP TRACE, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %s, %s"%(
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
        assert(tpb.statebuffer.file_mapper.check_overlap(weights_file_start_addr, weights_region_sz, ofmaps_region_start_addr, ofmaps_region_sz)==False)
        assert(tpb.statebuffer.file_mapper.check_overlap(weights_file_start_addr, weights_region_sz, ifmaps_region_start_addr, ifmaps_region_sz)==False)

        # check that regions are either exactly overlaping or not overlapping at all
        overlap_some_percent = tpb.statebuffer.file_mapper.check_overlap(single_ifmap_start, min(single_ifmap_sz, ifmaps_region_sz), single_ofmap_start, single_ofmap_sz)
        overlap_100_percent = tpb.statebuffer.file_mapper.check_overlap100(single_ifmap_start, min(single_ifmap_sz, ifmaps_region_sz), single_ofmap_start, single_ofmap_sz)
        assert(overlap_some_percent == overlap_100_percent)

        # Check conv fused op
        first_op_type = self.first_op.data['layer_type']
        if (first_op_type == "Conv" or first_op_type == "MatMul"):
            results = tpb.execute_conv_ops(batch_item)
        elif (first_op_type == "AvgPool" or first_op_type == "MaxPool"):
            results = tpb.execute_unfused_pool_op(batch_item)
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
        #elif re.search(r"Relu|Tanh|Sigmoid|Exp|Identity|Lrelu|Prelu|BiasAdd", first_op_type):
        #    first_op.src_circbuf = first_op.prev[0].dst_circbuf
        #    inputs = first_op.src_circbuf.load_data(first_op)
        #   results = tpb.execute_unfused_pool_op(inputs, result_file)
        #else:        
        #    print("ERROR: Unrecognized first operation %s"%first_op_type)
        #    exit(-1)

        # Check results against pre-computed results           
        if len(self) > 1 or (not self.first_op.is_placeholder and not self.first_op.is_nop):
        #if False:
            if 'ref_file' in self.last_op.data and os.path.isfile(last_op.data['ref_file']):
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
                    if (args.debug > 2): print("\nInput IFMAPS:\n", ifmaps)
                    if (args.debug > 1): print("\nComputed OFMAPS:\n", ofmaps)
                    if (args.debug > 1): print("\nExpected OFMAPS:\n", expected_ofmaps_extracted)
                    if (args.debug > 1): print("\nDiffed   OFMAPS:\n", diff)
                    if (not npu.allclose(ofmaps, expected_ofmaps_extracted, 1/100, 1e-5, verbose=True)):
                        print("\nERROR: layer %s batch item %d computed OFMAPS is not equal to expected OFMAPS!\n"%(self.last_op.data['layer_name'], i))
                        tpb.num_mismatches += 1
                    if self.last_op.is_output:
                        waveops = tpb.statebuffer.file_mapper.flush_file(tpb.waveop_stream.nonload_waveop_list, self.last_op.ofmaps_file_params, i)
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
        if (weights_sb_address == self.prev_weight_wave_lower_addr):
            weights_sb_address = -1
            if (args.debug > 1): print("DBG: weights has been previously loaded; reusing them instead of reloading")
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
            if args.abstract_mem:
                break_cond = chunk_id != current_chunk_id
            else:
                break_cond = not (atom_id == current_atom_id or atom_id == current_atom_id+1)
            if break_cond:
                break_at_y.append(i)
                break_addr.append(tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.conv_op.ifmaps_file_params, batch_item, address))
                current_chunk_id = chunk_id
                current_atom_id = atom_id
                if (args.debug > 3): print("DBG: breaking wave at row %d addr %d"%(i, break_addr[-1]))
        matmul_waveop = []
        start_tensor_calc = not(psum_add)

        # replication parameters
        ifmap_replication_resolution = 0
        ifmap_replication_num_rows = 0
        ifmap_replication_shift_amnt = 0
        if self.conv_op.ifmaps_file_params.replicate_multiple > 1:
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
            ifmaps_sb_address = break_addr[i]
            if i>0: weights_sb_address = -1

            waveop_name = self.conv_op.data['layer_name']+"/MatMul_"+wave_id.id_string()+"__"+str(i)

            # get dram waveops for each matmul
            dram_waveop_names = []
            if i==0:
                for j in dram_weights_waveops:
                    dram_waveop_names.append(j["waveop_name"])
            for z in range(self.conv_op.Tn):                    
                lower_file_address = self.conv_op.ifmap_wave_lower_addr[z] + break_at_y[i] * addr_step_y
                upper_file_address = min(self.conv_op.ifmap_wave_lower_addr[z] + next_break * addr_step_y - self.conv_op.item_sz, self.conv_op.ifmap_wave_upper_addr[z])
                list_of_names = tpb.statebuffer.file_mapper.get_dram_waveop_names(self.conv_op.ifmaps_file_params, batch_item + z, lower_file_address, upper_file_address)
                for name in list_of_names:
                    if name not in dram_waveop_names:
                        dram_waveop_names.append(name)
            fmap_z_step = (self.conv_op.ifmaps_file_params.batch_item_partition_usage_sz//self.conv_op.item_sz) if self.conv_op.Tn > 1 else 1

            if (args.debug > 2): print("DBG %s: MatMul wave %s subwave %d weights_sb_address %d, ifmaps_sb_address %d, fmap_y_num %d"%(self.conv_op.data['layer_name'], waveop_name, i, weights_sb_address, ifmaps_sb_address, fmap_y_num))                
            matmul_waveop.append({ 
                  'previous_waveops'        : dram_waveop_names,
                  'waveop_type'             : 'MatMul',
                  'waveop_name'             : waveop_name,
                  'layer_name'              : self.conv_op.data['layer_name'],
                  'weights_sb_address'      : weights_sb_address,
                  'ifmaps_sb_address'       : ifmaps_sb_address,
                  'in_dtype'                : in_dtype,
                  'out_dtype'               : out_dtype,
                  'wave_id_format'          : wave_id.format, # to be removed
                  'wave_id'                 : wave_id.show(), # to be removed
                  'start'                   : start_tensor_calc,    # to be removed
                  'stride_x'                : self.conv_op.stride_x, # to be removed
                  'stride_y'                : self.conv_op.stride_y, # to be removed
                  'ifmap_count'             : self.conv_op.ifmap_count, # to be removed
                  'ifmap_tile_width'        : self.conv_op.ofmap_wave_width, # to be removed 
                  'ifmap_tile_height'       : self.conv_op.ofmap_wave_height, # to be removed
                  'ofmap_count'             : self.conv_op.ofmap_count, # to be removed
                  'ofmap_tile_width'        : self.conv_op.ofmap_wave_width, # to be removed
                  'ofmap_tile_height'       : self.conv_op.ofmap_wave_height,  # to be removed
                  'batching_in_wave'        : self.conv_op.Tn, # to be removed
                  'start_tensor_calc'       : start_tensor_calc,
                  'stop_tensor_calc'        : False,
                  'fmap_x_step'             : self.conv_op.stride_x,
                  'fmap_x_num'              : self.conv_op.ofmap_wave_width,
                  'fmap_y_step'             : self.conv_op.W * self.conv_op.stride_y,
                  'fmap_y_num'              : fmap_y_num,
                  'fmap_z_step'             : fmap_z_step,
                  'fmap_z_num'              : self.conv_op.Tn,
                  'num_row_partitions'      : self.conv_op.ifmap_count,
                  'psum_bank_id'            : self.conv_op.psum_bank_dst,
                  'psum_bank_offset'        : self.conv_op.psum_bank_offset + psum_bank_additional_offset,
                  'psum_x_step'             : 1,
                  'psum_x_num'              : self.conv_op.ofmap_wave_width,
                  'psum_y_step'             : self.conv_op.ofmap_cropped_tile_width,
                  'psum_y_num'              : fmap_y_num,
                  'psum_z_step'             : self.conv_op.ofmap_full_tile_sz,
                  'psum_z_num'              : self.conv_op.Tn,
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
        if (src_is_psum):
            src_ifmap_width = self.pool_op.ifmap_cropped_tile_width
            src_ifmap_height = self.pool_op.ifmap_cropped_tile_height
            src_sb_address = 0
            if (self.pool_op.item_sz == 2):
                in_dtype = "float32"
            else:    
                in_dtype = "float32"
        else:
            src_ifmap_width = self.pool_op.W
            src_ifmap_height = self.pool_op.H
            src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.pool_op.ifmaps_file_params, partial_batch_item, self.pool_op.ifmap_wave_lower_addr[partial_batch_item])
            in_dtype = self.out_data_type
        src_psum_bank_offset = src_ifmap_width * src_ifmap_height * partial_batch_item
        psum_step_multiplier = 1   # kaena-174, tonga-310: after Inkling fix, no need for multiplier         
        waveop_name = self.pool_op.data['layer_name']+"/Pool_"+tile_id.id_string()
        pool_frequency = self.pool_op.pool_window_x * self.pool_op.pool_window_y
        pool_scale = float(1/pool_frequency)
        dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(self.pool_op.ofmaps_file_params, partial_batch_item, self.pool_op.ofmap_tile_lower_addr[partial_batch_item])
        pool_waveop = {
              'previous_waveops'        : [],   # to be added later
              'waveop_type'             : 'Pool',
              'waveop_name'             : waveop_name,
              'layer_name'              : self.pool_op.data['layer_name'],
              'tile_id_format'          : tile_id.format,
              'tile_id'                 : tile_id.show(),
              'pool_func'               : self.pool_op.data['layer_type'],
              'in_dtype'                : in_dtype,
              'out_dtype'               : self.out_data_type,
              'src_is_psum'             : src_is_psum,
              'src_psum_bank_id'        : src_psum_bank_id,
              'src_psum_bank_offset'    : src_psum_bank_offset,
              'src_sb_address'          : src_sb_address, 
              'src_start_at_mid_part'   : start_at_mid_part, 
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
              'dst_sb_address'          : dst_sb_address,
              'dst_start_at_mid_part'   : start_at_mid_part,
              'dst_x_step'              : 1,
              'dst_x_num'               : self.pool_op.ofmap_cropped_tile_width,
              'dst_y_step'              : self.pool_op.E,
              'dst_y_num'               : self.pool_op.ofmap_cropped_tile_height,
              'dst_z_step'              : 1, 
              'dst_z_num'               : 1,
            }
        return pool_waveop

    # execute PEArray matrix multiply; returns True if successful (IFMAP wave is non-zero)
    def execute_matmul_waveop(self, tpb, wave_id, inputs, weights, psum_add):
        batch_item = wave_id.n_id * self.conv_op.Tn
        pearray_packed_weights = self.conv_op.pack_wave_conv_weights(weights, wave_id, self.conv_op.ifmaps_file_params.replicate_multiple)
        pearray_packed_ifmaps = self.conv_op.pack_wave_ifmaps(
                                        inputs, 
                                        wave_id,
                                        self.conv_op.ifmaps_file_params.replicate_multiple,
                                        for_softmax=False
                                        )
        #print("\npearray_packed_ifmaps", wave_id.show(), "\n", pearray_packed_ifmaps)
        #print("\npearray_packed_weights", wave_id.show(), "\n", pearray_packed_weights)
        if (self.conv_op.ifmap_wave_lower_addr[0] < 0 or self.conv_op.ifmap_wave_upper_addr[0] < 0):
            print("WARNING layer %s: IFMAP wave (%s) has no data, so don't create waveops for this wave"%(op_list[0].data['layer_name'], wave_id.id_string()))
            return False
        else:
            tpb.pearray.wave_fp16_mm(pearray_packed_ifmaps, pearray_packed_weights, self.conv_op.psum_bank_dst, psum_add)
            # Generate waveops
            (writers, readers, dram_weights_waveops) = tpb.statebuffer.file_mapper.read_file_data_region(
                                        tpb.waveop_stream.nonload_waveop_count,
                                        tpb.waveop_stream.nonload_waveop_list,
                                        self.conv_op.weights_file_params,
                                        0,  # batch_item doesn't apply for weights
                                        self.conv_op.weight_wave_lower_addr, 
                                        self.conv_op.weight_wave_upper_addr - self.conv_op.weight_wave_lower_addr + self.conv_op.item_sz)
            for i in dram_weights_waveops: tpb.waveop_stream.append_check(i)

            dram_ifmaps_waveops = []
            prev_waveops = []
            for z in range(self.conv_op.Tn):
                # TODO: move the following into gen_matmul_waveop to handle breaking wave into two
                (writers, readers, waveops) = tpb.statebuffer.file_mapper.read_file_data_region(
                                            tpb.waveop_stream.nonload_waveop_count,
                                            tpb.waveop_stream.nonload_waveop_list,
                                            self.conv_op.ifmaps_file_params,
                                            batch_item + z,
                                            self.conv_op.ifmap_wave_lower_addr[z], 
                                            self.conv_op.ifmap_wave_upper_addr[z] - self.conv_op.ifmap_wave_lower_addr[z] + self.conv_op.item_sz)
                if args.no_inter_layer_load:
                    if (not self.conv_op.is_input and len(waveops) > 0):
                        raise RuntimeError("There are DRAM loads when option no_inter_layer_load is set")
                if (args.debug > 2): print("DBG %s: MatMul ifmaps_wave_lower_addr %d ifmap_wave_upper_addr %d"%(self.conv_op.data['layer_name'], self.conv_op.ifmap_wave_lower_addr[z], self.conv_op.ifmap_wave_upper_addr[z]))                
                dram_ifmaps_waveops += waveops
                # TODO: roll this code into read_file_data_region
                if waveops == []:
                    accessors = writers + readers
                    if accessors != []:
                        latest_accessor = max(accessors)
                        if latest_accessor >= 0:
                            prev_waveops.append(tpb.waveop_stream.nonload_waveop_list[latest_accessor]['waveop_name'])

            for i in dram_ifmaps_waveops: tpb.waveop_stream.append_check(i)
            matmul_waveop = self.gen_matmul_waveop(tpb, wave_id, psum_add, dram_weights_waveops)
            for i in range(len(matmul_waveop)):
                tpb.waveop_stream.add_linked(matmul_waveop[i], [], self.conv_op.psum_bank_dst)
                # TODO: move the following into gen_matmul_waveop to handle breaking wave into two
                matmul_waveop[i]['previous_waveops'] += prev_waveops                
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
                psum_temp = tpb.activate.act(op_list[i].data['layer_type'], psum_temp)
                psum_bank_dst = psum_bank_src
                dst_is_psum = False
                if (i != len(op_list)-1):
                    dst_is_psum = True
                    tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, psum_temp)
                tpb.gen_act_waveop_inline(None, op_list[i], self.conv_op, tile_id, 
                                          True, psum_bank_src, dst_is_psum, psum_bank_dst, [], 0)
                psum_bank_src = psum_bank_dst
            elif (layer_type == 'BiasAdd'):
                bias_chan_start = (tile_id.m_id//2) * PEArray.NUM_ROWS
                bias_chan_mid_part = (tile_id.m_id%2) == 1
                bias_chan_end = min(bias_chan_start + PEArray.NUM_ROWS, self.conv_op.M)
                bias_extracted = np.zeros(PEArray.NUM_ROWS)
                bias_extracted[0 : bias_chan_end - bias_chan_start] = bias[bias_chan_start : bias_chan_end]
                bias_addr = bias_chan_start * op_list[i].item_sz
                # TODO: fix waveop generation
                dram_bias_waveops = []
                if (tile_id.m_id%2 == 0):
                    (writers, readers, dram_bias_waveops) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                    tpb.waveop_stream.nonload_waveop_count,
                                                    tpb.waveop_stream.nonload_waveop_list,
                                                    self.biasadd_op.bias_file_params,
                                                    0,  # batch_item is not needed for bias
                                                    bias_addr,
                                                    self.biasadd_op.item_sz)
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
                    tpb.gen_act_waveop_inline(op_list[i], op_list[i+1], self.conv_op, tile_id, 
                                              True, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_addr)
                    psum_bank_src = psum_bank_dst
                    next(op_list_iter)
                else:                                    
                    if (i != len(op_list)-1):
                        dst_is_psum = True
                        tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, psum_temp)
                    tpb.gen_act_waveop_inline(op_list[i], None, self.conv_op, tile_id, 
                                              True, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_addr)
                    psum_bank_src = psum_bank_dst
            elif (self[i].is_join):
                dram_resadd_waveops = []
                prev_waveops = []
                #for z in range(op_list.conv_op.Tn):
                #    (writers, readers, waveops) = tpb.statebuffer.file_mapper.read_file_data_region(
                #                                tpb.waveop_stream.nonload_waveop_count,
                #                                tpb.waveop_stream.nonload_waveop_list,
                #                                self.last_op.ofmaps_file_params,
                #                                batch_item + z,
                #                                self.conv_op.ofmap_tile_lower_addr[z], 
                #                                self.conv_op.ofmap_tile_upper_addr[z] - self.conv_op.ofmap_tile_lower_addr[z] + self.conv_op.item_sz)
                #    if args.no_inter_layer_load:
                #        if (not self.conv_op.is_input and len(waveops) > 0):
                #            raise RuntimeError("There are DRAM loads when option no_inter_layer_load is set")
                #    if (args.debug > 2): print("DBG %s: ResAdd/Mult ofmaps_tile_lower_addr %d ofmap_tile_upper_addr %d"%(self.conv_op.data['layer_name'], self.conv_op.ofmap_tile_lower_addr[z], self.conv_op.ofmap_tile_upper_addr[z]))                
                #    dram_resadd_waveops += waveops
                #    # TODO: roll this code into read_file_data_region
                #    if waveops == []:
                #        accessors = writers + readers
                #        if accessors != []:
                #            latest_accessor = max(accessors)
                #            if latest_accessor >= 0:
                #                prev_waveops.append(tpb.waveop_stream.nonload_waveop_list[latest_accessor]['waveop_name'])

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
                tpb.gen_join_waveop_inline(op_list[i], 
                        self.conv_op, 
                        tile_id, 
                        True,
                        psum_bank_src, 
                        dst_is_psum, 
                        psum_bank_dst, 
                        dram_resadd_waveops, 
                        self.conv_op.ofmap_tile_lower_addr[0], 
                        (tile_id.m_id%2)==1)
                #tpb.waveop_stream.last_main_waveop['previous_waveops'] += prev_waveops
                # TODO: fix waveop generation
                #for z in range(op_list.conv_op.Tn):
                #    tpb.statebuffer.circbuf_residue.get_dram_waveop_names(self.conv_op.ofmap_tile_lower_addr[z], self.conv_op.ofmap_tile_upper_addr[z], tpb.waveop_stream.last_main_waveop['waveop_name'])
                #if ((tile_id.m_id%2)==1 or tile_id.m_id == tile_id.m-1):                
                #    for z in range(op_list.conv_op.Tn):
                #        tpb.statebuffer.circbuf_residue.free_data_region(self.conv_op.ofmap_tile_lower_addr[z], self.conv_op.ofmap_tile_upper_addr[z], tpb.waveop_stream.last_main_waveop)
                psum_bank_src = psum_bank_dst
            elif ((layer_type == 'AvgPool') or (layer_type == 'MaxPool')):
                tpb.activate.wait_tile_done(tile_id)
                self[i].compute_ofmap_tile_info(tile_id)
                #tilex = self.conv_op.ofmap_cropped_tile_width
                #tiley = self.conv_op.ofmap_cropped_tile_height
                tilex = self[i].ofmap_full_tilex_sz * self[i].stride_x
                tiley = self[i].ofmap_full_tiley_sz * self[i].stride_y
                #x = DBG_DUMP_PSUM_COL("PSUM before pool: ", psum_temp, 0)
                psum_temp = tpb.pool.pool(layer_type, psum_temp, self[i].stride_x, self[i].pool_window_y, self[i].Tn, tilex, tiley, self[i].ofmap_full_tilex_sz, self[i].ofmap_full_tiley_sz)
                #x = DBG_DUMP_PSUM_COL("PSUM after pool: ", psum_temp, 0)
                tpb.gen_fused_pool_waveop_inline(op_list, tile_id, psum_bank_src, (tile_id.m_id%2) == 1)
                # Don't go to back to psum for pooling
                #psum_bank_dst = 3
                #if (i != len(op_list)-1):
                #    tpb.pearray.write_psum(psum_bank_dst, 0, self[i].ofmap_full_tile_sz, psum_temp)
                #psum_bank_src = psum_bank_dst
            else:
                print ("ERROR: %s is currently not yet implemented"%layer_type)
                exit(-1)
        return psum_temp

##################################################################################
# RegExs to determine whether next node is fusable or not
next_is_fusable = {
        'Conv'     : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        'MatMul'   : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        'BiasAdd'  : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        'Add'      : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        'ResAdd'   : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        'Multiply' : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        'Relu'     : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        }

##################################################################################
# KGraph: nodes, edges, and operations
class KGraph:

    def __init__(self):
        # Node dictionary contains name -> Node pairs for quick reference
        self.node_dict = {}
        #self.first_node = []
        self.last_node = None
        self.data_type = 'float16'
        self.item_sz = 2
        self.current_node = None
        self.last_split_next_nodes = []
        self.first_leg = False
        self.seen_begin_of_first_leg = False
        self.current_file_id = 0

    # add forward edges for forward traversals        
    def add_forward_refs(self, starting_node):
        if (starting_node != None):
            #print (starting_node.data['layer_name'], len(starting_node.prev))
            if (len(starting_node.prev) > 0):
                non_const_prev_count = 0
                for i in starting_node.prev:
                    i.add_next(starting_node)
                    if not i.is_const:
                        non_const_prev_count += 1
                    self.add_forward_refs(i)
                assert(non_const_prev_count <= 2)
                starting_node.is_join = (non_const_prev_count > 1)                    

    # add a copy of layer, and change it to a new type
    def add_copy_with_new_type(self, layer, new_type, node_number):
        new_layer = copy.deepcopy(layer)
        new_layer['layer_type'] = new_type
        new_layer['layer_name'] = layer['layer_name'] + "_" + new_type
        new_layer['ref_file'] = layer['ref_file'].replace(".npy", "_" + new_type + ".npy")
        new_node = KNode(self, new_layer, self.item_sz, self.data_type, node_number)
        new_node.add_prev(self.last_node)
        self.node_dict[ new_layer['layer_name'] ] = new_node
        self.last_node = new_node

    # populate graph using layer info from JSON                    
    def populate_from_kgraph_json(self, kgraph_json):                    
        # get the lowest significant bit
        self.data_type = kgraph_json["data_type"]
        if (self.data_type == 'float16'):
            self.item_sz = 2
        elif (self.data_type == 'float32'):
            self.item_sz = 4
        elif (self.data_type == 'uint8'):
            self.item_sz = 1
        else:
            print("ERROR: cannot handle data type %s"%self.data_type)
            exit(-1)

        # process layers
        layers = kgraph_json["layers"]
        num_layers = len(layers)
        node_number = 0
        if (num_layers >= 1):
            for l in layers:
                new_node = KNode(self, l, self.item_sz, self.data_type, node_number)
                node_number += 1 
                prev_layers = l['previous_layers']
                if (len(prev_layers) > 0):
                    for i in prev_layers:
                        if i in self.node_dict:
                            #if (args.debug>0): print("Previous layer for ", new_node.data['layer_name'], " is ", i)
                            prev_node = self.node_dict[i]
                            # dissolve StridedSlice into the next operation
                            if (prev_node.data['layer_type'] == "StridedSlice"):
                                new_node.stridedslice_chan_offset = prev_node.data['channel_slice'][0]
                                print("%s: stridedslice_chan_offset %d"%(new_node.data['layer_name'], new_node.stridedslice_chan_offset))
                                assert (len(self.node_dict[i].prev) == 1)
                                new_node.add_prev(self.node_dict[i].prev[0])
                            elif (prev_node.data['layer_type'] == "Unstack"):
                                for j in prev_node.data['next_layer_order']:
                                    if j[1] == new_node.data['layer_name']:
                                        new_node.unstack_h_offset = j[0]
                                        print("%s: unstack_h_offset %d"%(new_node.data['layer_name'], j[0]))
                                        break
                                assert (len(self.node_dict[i].prev) == 1)
                                new_node.add_prev(self.node_dict[i].prev[0])
                            else:
                                new_node.add_prev(self.node_dict[i])
                        else:
                            print("ERROR: node %s isn't declared before %s"%(i, l['layer_name']))
                            exit(-1)
                else:
                    # Type "Input" node
                    if (l['layer_type'] == "Input" or l['layer_type'] == "Placeholder"):
                        new_node.is_placeholder = True
                        #self.first_node.append(new_node)
                        if (self.last_split_next_nodes == []):
                            self.last_split_next_nodes.append([])
                        self.last_split_next_nodes[0].append(new_node)
                # assume the last node is the last one processed (JSON graph is in order), at least for the last one
                self.last_node = new_node                
                self.node_dict[ l['layer_name'] ] = new_node
                # if softmax, expand to multiple subnodes
                if (l['layer_type'] == "Softmax"):
                    self.last_node.data['layer_type'] = "Exp"
                    self.add_copy_with_new_type(l, "Softmax2", node_number)
                    node_number += 1 
                    # move ref file attribute to the last operation for final comparisons
                    self.last_node.data['ref_file'] = new_node.data['ref_file']
                    new_node.data['ref_file'] = new_node.data['ref_file'].replace(".npy", "_Exp.npy")
                elif (l['layer_type'] == "Const"):
                    new_node.is_const = True
                elif (l['layer_type'] == "Reshape"):
                    new_node.is_nop = True
            if (len(self.last_split_next_nodes) > 0 and len(self.last_split_next_nodes[0]) > 0) :                    
                self.current_node = self.last_split_next_nodes[0].pop()
                if self.last_split_next_nodes[0] == []:
                    self.last_split_next_nodes.pop()
            else:
                print("ERROR: can't find any Input layer!")
                exit(-1)
        else:
            print("ERROR: there are no layers!")
            exit(-1)

        # process waveops 
        if ("waveops" in kgraph_json):
            layers = kgraph_json["waveops"]
            num_layers = len(layers)
            if (num_layers >= 1):
                for l in layers:
                    new_node = KNode(self, l, self.item_sz, self.data_type, node_number)
                    node_number += 1 
                    prev_layers = l['previous_waveops']
                    if (len(prev_layers) > 0):
                        for i in prev_layers:
                            if i in self.node_dict:
                                #if (args.debug > 0): print("Previous waveop for ", new_node.data['waveop_name'], " is ", i)
                                new_node.add_prev(self.node_dict[i])
                            else:
                                print("ERROR: node %s isn't declared before %s"%(i, l['waveop_name']))
                                exit(-1)
                    #else:
                        # the node with "Placeholder" type is input
                        #self.first_node.append(new_node)
                    # assume the last node is the last one processed (JSON graph is in order), at least for the last one
                    self.last_node = new_node                
                    self.node_dict[ l['waveop_name'] ] = new_node
                #self.current_node = self.first_node[0]
            else:
                print("ERROR: there are no layers!")
                exit(-1)

    # get next fused op            
    def get_next_fused_op(self, fused_ops):
        next_nodes = fused_ops[-1].next
        last_node = fused_ops[-1]
        last_node_type = fused_ops[-1].data['layer_type']
        # if there's only one next node, check if it is fusable and add
        if (len(next_nodes) == 1):
            if last_node_type in next_is_fusable:
                if next_nodes[0].count_missing_input_results() <= 1:
                    regex = next_is_fusable[last_node_type]
                    if re.search(regex, next_nodes[0].data['layer_type']):               
                        if fused_ops.add(next_nodes[0]):
                            fused_ops = self.get_next_fused_op(fused_ops)
                elif next_nodes[0].is_join:
                    fused_ops.ofmap_is_for_join = True
        return fused_ops                    

    # starting from current node position, collect as many operations as possible            
    def get_fused_ops(self, fused_op_id):
        fused_ops = FusedOp(self.data_type, fused_op_id)
        if (self.current_node == None):
            print("ERROR: found zero operations to fuse")
            exit(-1)
        # when we see ResAdd/Multiply, backtrack to the last split and follow the next leg in list
        if (self.current_node.is_join and self.current_node.count_missing_input_results() > 0):
            if (args.debug > 0): print("DBG: found join (ResAdd, Multiply, etc), back-track to last split and follow next leg")
            if self.last_split_next_nodes != [] and self.last_split_next_nodes[-1] != []:
                self.current_node = self.last_split_next_nodes[-1].pop()
                if self.last_split_next_nodes[-1] == []: 
                    self.last_split_next_nodes.pop()
            else:
                print("ERROR: back-track from a join %s, but can't find fork!"%(self.current_node.data['layer_name']))
                exit(-1)
            self.first_leg = False
        fused_ops.add(self.current_node)
        if (self.first_leg and not self.seen_begin_of_first_leg):
            fused_ops.begin_of_first_leg = True
            self.seen_begin_of_first_leg = True
        for i in self.current_node.next:
            print(i.data['layer_type'], ":", i.data['layer_name'])
        fused_ops = self.get_next_fused_op(fused_ops)
        # if there are multiple next nodes
        next_nodes = [i for i in fused_ops[-1].next]
        last_node_type = fused_ops[-1].data['layer_type']
        if (len(next_nodes) == 1):
            self.current_node = next_nodes[0]   
        elif (len(next_nodes) > 1):
            fused_ops[-1].is_fork = True
            # Delete the leg that goes to ResAdd directly first, if it exists.
            # At the first fusedop, begin_of_first_leg=1, and the IFMAPs will be saved to residue and used by ResAdd
            for i in range(len(next_nodes)):
                if (next_nodes[i].is_join):
                    resadd_node = next_nodes[i]
                    del next_nodes[i]
                    #next_nodes.insert(0, resadd_node)
            # pick the first leg as current_node                        
            self.current_node = next_nodes.pop()
            self.first_leg = True
            self.seen_begin_of_first_leg = False
            # save the remaining legs in a list
            if (next_nodes != []):
                self.last_split_next_nodes.append(next_nodes)
        else:
            if self.last_split_next_nodes != [] and self.last_split_next_nodes[-1] != []:
                self.current_node = self.last_split_next_nodes[-1].pop()
                if self.last_split_next_nodes[-1] == []: 
                    self.last_split_next_nodes.pop()
            else:                
                self.current_node = None
        # if the last node is Conv or MatMul, add an identity pool op
        if (last_node_type == "Conv" or last_node_type == "MatMul"):
            fused_ops.add(self.gen_id_pool_op(fused_ops[-1]))
        # set the first op source is not PSUM, and last op dest is not PSUM
        fused_ops.first_op = fused_ops[0]
        fused_ops.first_op.src_is_psum = False
        fused_ops.last_op = fused_ops[-1]
        fused_ops.last_op.dst_is_psum = False
        # If ResAdd capture ofmaps_file_params from the other branch.
        # Also, if it is followed by BiasAdd or Activate, use the same ofmaps_file_params for last op
        if fused_ops.has_join:
            if fused_ops.last_op != fused_ops.join_op:
                assert(fused_ops.join_op.ofmaps_file_params.file_dims.shape_tuple == fused_ops.last_op.ofmaps_file_params.file_dims.shape_tuple)
                fused_ops.join_op.ofmaps_file_params = fused_ops.last_op.ofmaps_file_params
                fused_ops.join_op.ofmaps_file_params.writers_of_shared_fmap.append(fused_ops.join_op)
            residue_op = fused_ops.join_op.prev[fused_ops.join_op.residue_index]                
            assert(residue_op.ofmaps_file_params.file_dims.shape_tuple == fused_ops.join_op.ofmaps_file_params.file_dims.shape_tuple)
            # must change readers of shared FMAP before changing writers, because one of the writers is residue_op itself
            for i in residue_op.ofmaps_file_params.readers_of_shared_fmap:
                fused_ops.join_op.ofmaps_file_params.readers_of_shared_fmap.append(i)
                i.ifmaps_file_params = fused_ops.join_op.ofmaps_file_params
            for i in residue_op.ofmaps_file_params.writers_of_shared_fmap:
                fused_ops.join_op.ofmaps_file_params.writers_of_shared_fmap.append(i)
                i.ofmaps_file_params = fused_ops.join_op.ofmaps_file_params
            residue_op.ofmaps_file_params = fused_ops.join_op.ofmaps_file_params
        # preload files
        if fused_ops.first_op.ifmaps_file_params is not None:
            fused_ops.first_op.ifmaps_file_params.load_file()
        if fused_ops.last_op.ofmaps_file_params is not None:
            fused_ops.last_op.ofmaps_file_params.zero_file()
            fused_ops.last_op.ofmaps_file_params.writers_of_shared_fmap.append(fused_ops.last_op)
        # transfer replicate_multiple from weights_file_params to ifmaps_file_params
        if fused_ops.has_conv:
            fused_ops.conv_op.ifmaps_file_params.replicate_multiple = fused_ops.conv_op.weights_file_params.replicate_multiple
            fused_ops.conv_op.ifmaps_file_params.weights_S_dim = fused_ops.conv_op.weights_file_params.file_dims.S
            fused_ops.conv_op.ifmaps_file_params.stride_x = fused_ops.conv_op.stride_x
            fused_ops.conv_op.ifmaps_file_params.stride_y = fused_ops.conv_op.stride_y
            print("copied ifmaps_file_params.replicate_multiple = weights_file_params.replicate_multiple %d"%(fused_ops.conv_op.weights_file_params.replicate_multiple))
        # mark fusedops to be at end of first leg if the following op is ResAdd
        if (self.first_leg 
                and self.current_node != None 
                and (self.current_node.data['layer_type'] == "ResAdd" or self.current_node.data['layer_type'] == "Multiply")):
            fused_ops.end_of_first_leg = True
        if (args.debug > 0):
            fused_ops.show()
        return fused_ops                   

    def gen_id_pool_op(self, last_op):
        id_pool_layer_data = {
          "kernel_shape"    : [ 1, 1, 1, 1 ],
          "layer_name"      : last_op.data['layer_name'],
          "layer_type"      : "MaxPool",
          "ofmap_format"    : last_op.data['ofmap_format'],
          "ofmap_shape"     : last_op.data['ofmap_shape'],
          "padding"         : [ [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ] ],
          "previous_layers" : [ last_op.data['layer_name'] ],
          "stride"          : [ 1, 1, 1, 1 ],
          "ref_file"        : last_op.data['ref_file']
        }
        # WARNING: node_number is not unique when there's ID pool
        id_pool_op = KNode(self, id_pool_layer_data, self.item_sz, self.data_type, last_op.node_number + 1)
        id_pool_op.is_fork = last_op.is_fork
        id_pool_op.prev.append(last_op)
        for next_op in last_op.next:
            for j in range(len(next_op.prev)):
                if next_op.prev[j] == last_op:
                    del next_op.prev[j]
                    next_op.prev.append(id_pool_op)
        return id_pool_op

    def walk_ended(self):
        return (self.current_node == None and self.last_split_next_nodes == [])

##################################################################################
# The TPB scheduler has access to:
#   PEArray 
#   Pool 
#   BiasAddAct 
class TPBSched:
    def __init__(self, batcher):
        self.pearray = PEArray()
        self.pool = Pool()
        self.activate = BiasAddAct()
        self.statebuffer = StateBuffer(batcher)
        self.waveop_stream = WaveopStream()
        self.num_mismatches = 0

    # generate activation instruction and add it to instruction stream
    def gen_recip_waveop_inline(self, op, psum_bank_src, dst_is_psum, psum_bank_dst):
        layer_name = op.data["layer_name"]
        in_dtype = "float32"
        out_dtype = "float32"
        waveop_name = layer_name+"/Reciprocal"
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : 'Reciprocal',
              'waveop_name'             : waveop_name,
              'layer_name'              : layer_name,
              'in_dtype'                : in_dtype,
              'out_dtype'               : out_dtype,
              'src_psum_bank_id'        : psum_bank_src,
              'src_x_step'              : 1,
              'src_x_num'               : 1,
              'src_y_step'              : 1,
              'src_y_num'               : 1,
              'src_z_step'              : 1,
              'src_z_num'               : 1,
              'dst_is_psum'             : dst_is_psum, 
              'dst_psum_bank_id'        : psum_bank_dst,
              'dst_sb_address'          : 0, # Destination is PSUM, so no need for this
              'dst_start_at_mid_part'   : False,
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
            src_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_address(op.ifmaps_file_params, 0, op.ifmap_wave_lower_addr[0])
        if (dst_is_psum):
            print("ERROR: for scale/add waveop, cannot handle destination PSUM")
            exit(-1)
        dst_x_num = op.ofmap_full_tilex_sz
        dst_y_step = op.E
        dst_y_num = op.ofmap_full_tiley_sz
        dst_z_step = (op.ofmaps_file_params.batch_item_partition_usage_sz//op.item_sz) if op.Tn > 1 else 1
        dst_z_num = op.Tn  # Need CNHW data format
        num_partitions = op.ofmap_count
        dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, 0, op.ofmap_tile_lower_addr[0])
        waveop_name = layer_name+"/ScaleAdd_"+tile_id.id_string()            
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : 'ScaleAdd',
              'waveop_name'             : waveop_name,
              'layer_name'              : layer_name,
              'tile_id_format'          : tile_id.format,
              'tile_id'                 : tile_id.show(),
              'in_dtype'                : in_dtype,
              'out_dtype'               : out_dtype,
              'src_is_psum'             : src_is_psum,
              'src_psum_bank_id'        : psum_bank_src,
              'src_psum_bank_offset'    : 0,
              'src_sb_address'          : src_sb_address, 
              'src_x_step'              : 1,
              'src_x_num'               : dst_x_num,
              'src_y_step'              : dst_y_step,
              'src_y_num'               : dst_y_num,
              'src_z_step'              : dst_z_step,
              'src_z_num'               : dst_z_num,
              'src_start_at_mid_part'   : tile_id.m_id%2 == 1,
              'dst_is_psum'             : dst_is_psum,
              'dst_psum_bank_id'        : psum_bank_dst,
              'dst_psum_bank_offset'    : 0,
              'dst_sb_address'          : dst_sb_address,
              'dst_x_step'              : 1,
              'dst_x_num'               : dst_x_num,
              'dst_y_step'              : dst_y_step,
              'dst_y_num'               : dst_y_num,
              'dst_z_step'              : dst_z_step,
              'dst_z_num'               : dst_z_num,
              'dst_start_at_mid_part'   : tile_id.m_id%2 == 1,
              'num_partitions'          : num_partitions,
              'scale'                   : scale_val,
              'add'                     : add_val,
            }
        self.waveop_stream.add_linked(instr, dram_waveops, psum_bank_src if src_is_psum else -1)

    # generate activation instruction and add it to instruction stream
    def gen_act_waveop_inline(self, biasadd_op, act_op, conv_op, tile_id, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_start):
        layer_name = ""
        bias_add_en = False
        bias_sb_address = 0
        # TODO: update in_dtype when src_is_psum is added
        in_dtype = "float32"
        out_dtype = "float32"
        act_or_biasadd_op = None
        if (biasadd_op != None):
            act_or_biasadd_op = biasadd_op
            bias_add_en = True
            bias_sb_address = self.statebuffer.file_mapper.get_sb_addr_from_file_addr(biasadd_op.bias_file_params, 0, bias_start)
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
        dst_x_num = 1
        dst_y_step = 1
        dst_y_num = 1
        dst_z_num = 1
        dst_z_step = 1
        num_partitions = PEArray.NUM_COLS
        if (conv_op != None):
            if (dst_is_psum):
                dst_x_num = conv_op.ofmap_cropped_tile_width
                dst_y_step = conv_op.ofmap_cropped_tile_width
                dst_y_num = conv_op.ofmap_cropped_tile_height
                dst_z_step = dst_y_step * dst_y_num 
                dst_z_num = conv_op.Tn
            else:                
                dst_x_num = conv_op.ofmap_cropped_tile_width
                dst_y_step = conv_op.F
                dst_y_num = conv_op.ofmap_cropped_tile_height
                dst_z_step = (conv_op.ofmaps_file_params.batch_item_partition_usage_sz//conv_op.item_sz) if conv_op.Tn > 1 else 1
                dst_z_num = conv_op.Tn
            num_partitions = conv_op.ofmap_count
        elif (act_or_biasadd_op !=  None):
            # unfused
            dst_x_num = act_or_biasadd_op.F
            dst_y_step = act_or_biasadd_op.F
            dst_y_num = act_or_biasadd_op.E
            dst_z_step = dst_y_step * dst_y_num # Need CNHW data format
            dst_z_num = act_or_biasadd_op.Tn  # Need CNHW data format
            num_partitions = act_or_biasadd_op.ofmap_count
        src_sb_address = 0
        if not src_is_psum:
            src_sb_address = self.statebuffer.file_mapper.get_sb_addr_from_file_addr(act_or_biasadd_op.ifmaps_file_params, batch_item, act_or_biasadd_op.ifmap_wave_lower_addr[0])
        dst_sb_address = 0
        if not dst_is_psum:
            if (conv_op != None):
                dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(act_or_biasadd_op.ofmaps_file_params, batch_item, conv_op.ofmap_tile_lower_addr[0])
            else:                
                dst_sb_address = tpb.statebuffer.file_mapper.get_sb_addr_from_file_addr(act_or_biasadd_op.ofmaps_file_params, batch_item, act_or_biasadd_op.ofmap_tile_lower_addr[0])
        waveop_name = layer_name+"/Activation_"+tile_id.id_string()            
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : 'Activation',
              'waveop_name'             : waveop_name,
              'layer_name'              : layer_name,
              'tile_id_format'          : tile_id.format,
              'tile_id'                 : tile_id.show(),
              'activation_func'         : act_type,
              'in_dtype'                : in_dtype,
              'bias_dtype'              : act_or_biasadd_op.ifmaps_file_params.data_type, 
              'out_dtype'               : out_dtype,
              'src_is_psum'             : src_is_psum,
              'src_sb_address'          : src_sb_address,
              'src_psum_bank_id'        : psum_bank_src,
              'src_start_at_mid_part'   : tile_id.m_id%2 == 1,
              'src_x_step'              : 1,
              'src_x_num'               : dst_x_num,
              'src_y_step'              : dst_y_step,
              'src_y_num'               : dst_y_num * dst_z_num,
              'dst_is_psum'             : dst_is_psum,
              'dst_psum_bank_id'        : psum_bank_dst,
              'dst_sb_address'          : dst_sb_address,
              'dst_start_at_mid_part'   : tile_id.m_id%2 == 1,
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
        self.waveop_stream.add_linked(instr, dram_bias_waveops, psum_bank_src if src_is_psum else -1)

    # generate ResAdd instruction and add it to instruction stream
    def gen_join_waveop_inline(self, op, conv_op, tile_id, src_is_psum, psum_bank_src, dst_is_psum, psum_bank_dst, dram_resadd_waveops, data_start, start_at_mid_part):
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
        dst_x_num = 1
        dst_y_step = 1
        dst_y_num = 1
        dst_z_num = 1
        dst_z_step = 1
        num_partitions = PEArray.NUM_COLS
        if (conv_op != None):
            if (dst_is_psum):
                dst_x_num = conv_op.ofmap_cropped_tile_width
                dst_y_step = conv_op.ofmap_cropped_tile_width
                dst_y_num = conv_op.ofmap_cropped_tile_height
                dst_z_step = dst_y_step * dst_y_num 
                dst_z_num = conv_op.Tn
            else:                
                dst_x_num = conv_op.ofmap_full_tilex_sz
                dst_y_step = conv_op.E
                dst_y_num = conv_op.ofmap_cropped_tile_height
                dst_z_step = (conv_op.ofmaps_file_params.batch_item_partition_usage_sz//conv_op.item_sz) if conv_op.Tn > 1 else 1
                dst_z_num = conv_op.Tn
            num_partitions = conv_op.ofmap_count
        else:
            # unfused
            dst_x_num = op.E
            dst_y_step = op.E
            dst_y_num = op.F
            dst_z_step = dst_y_step * dst_y_num # Need CNHW data format
            dst_z_num = op.Tn  # Need CNHW data format
            num_partitions = op.ofmap_count
        src_b_sb_address = 0
        if not src_is_psum:
            src_b_sb_address = self.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ifmaps_file_params, 0, data_start)
        src_a_sb_address = self.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, 0, data_start)
        dst_sb_address = 0
        if not dst_is_psum:
            if (conv_op != None):
                dst_sb_address = self.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, 0, conv_op.ofmap_tile_lower_addr[0])
            else:                
                dst_sb_address = self.statebuffer.file_mapper.get_sb_addr_from_file_addr(op.ofmaps_file_params, 0, op.ofmap_tile_lower_addr[0])
        waveop_name = op.data['layer_name']+"/"+op.data['layer_type']+"_"+tile_id.id_string()
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : "ResAdd", #op.data['layer_type'],
              'waveop_name'             : waveop_name,
              'multiply'                : op.data['layer_type'] == "Multiply",    # Hack to use ResAdd in old ISA to run Multiply 
              'layer_name'              : op.data['layer_name'],
              'tile_id_format'          : tile_id.format,
              'tile_id'                 : tile_id.show(),
              'in_a_dtype'              : in_a_dtype,
              'in_b_dtype'              : in_b_dtype,
              'out_dtype'               : out_dtype,
              'src_a_is_psum'           : False,
              'src_a_psum_bank_id'      : 0,
              'src_a_psum_bank_offset'  : 0,
              'src_a_sb_address'        : src_a_sb_address,
              'src_a_start_at_mid_part' : start_at_mid_part,
              'src_a_x_step'            : 1,
              'src_a_x_num'             : dst_x_num,
              'src_a_y_step'            : dst_y_step,
              'src_a_y_num'             : dst_y_num,
              'src_a_z_step'            : dst_z_step,
              'src_a_z_num'             : dst_z_num,
              'src_b_is_psum'           : src_is_psum,
              'src_b_psum_bank_id'      : psum_bank_src,
              'src_b_psum_bank_offset'  : 0,
              'src_b_sb_address'        : src_b_sb_address,
              'src_b_start_at_mid_part' : start_at_mid_part,
              'src_b_x_step'            : 1,
              'src_b_x_num'             : dst_x_num,
              'src_b_y_step'            : dst_y_step,
              'src_b_y_num'             : dst_y_num,
              'src_b_z_step'            : dst_z_step,
              'src_b_z_num'             : dst_z_num,
              'dst_is_psum'             : dst_is_psum,
              'dst_psum_bank_id'        : psum_bank_dst,
              'dst_psum_bank_offset'    : 0,
              'dst_sb_address'          : dst_sb_address,
              'dst_start_at_mid_part'   : start_at_mid_part,
              'dst_x_step'              : 1,
              'dst_x_num'               : dst_x_num,
              'dst_y_step'              : dst_y_step,
              'dst_y_num'               : dst_y_num,
              'dst_z_step'              : dst_z_step,
              'dst_z_num'               : dst_z_num,
              'num_partitions'          : num_partitions,
            }
        self.waveop_stream.add_linked(instr, dram_resadd_waveops, psum_bank_src if src_is_psum else -1)

    def gen_fused_pool_waveop_inline (self, fused_ops, tile_id, psum_bank_src, start_at_mid_part):
        for z in range(fused_ops.pool_op.Tn):
            pool_waveop = fused_ops.gen_pool_waveop(self, tile_id, True, psum_bank_src, start_at_mid_part, z)
            self.waveop_stream.add_linked(pool_waveop, [], psum_bank_src)

    def gen_unfused_pool_waveop_inline (self, fused_ops, tile_id, dram_waveops, start_at_mid_part):
        for z in range(fused_ops.pool_op.Tn):
            pool_waveop = fused_ops.gen_pool_waveop(self, tile_id, False, 0, start_at_mid_part, z)
            self.waveop_stream.add_linked(pool_waveop, dram_waveops if z==0 else [], -1)

    # collect stats
    def collect_stats(self, layer_name):        
        if (args.debug > 1):
            # collect stats
            stats = Stats()
            for i in [self.statebuffer.circbuf_weights, self.statebuffer.circbuf_ifmaps, self.statebuffer.circbuf_scratch, self.statebuffer.circbuf_residue, self.statebuffer.circbuf_bias]:
                i.populate_stats(stats)
            self.calculate_compute_to_load_ratio(layer_name, stats)
            stats_per_layer.append(stats)
            # reset stats
            self.pearray.num_wave_fp16_mm = 0
            self.pearray.total_pearray_latency_cycles = 0
            self.pearray.total_pearray_wave_elems = 0
            self.pearray.num_of_ops_executed = 0
            for i in [self.statebuffer.circbuf_weights, self.statebuffer.circbuf_ifmaps, self.statebuffer.circbuf_scratch, self.statebuffer.circbuf_residue, self.statebuffer.circbuf_bias]:
                i.DRAM_elem_read = 0
                i.DRAM_elem_written = 0
                i.DRAM_atoms_read = 0
                i.DRAM_atoms_read_short = 0
                i.DRAM_atoms_written = 0
                i.circbuf_stats = CircbufStats()

    # print out statistics
    def calculate_compute_to_load_ratio(self, layer_name, stats):
        if (args.debug > 1):
            stats.layer_name            = layer_name
            stats.num_waves             = self.pearray.num_wave_fp16_mm
            #stats.num_waves_x_max_pe_ops = self.pearray.num_wave_fp16_mm * self.pearray.NUM_ROWS * self.pearray.NUM_COLS * self.pearray.MAX_WAVE_SIZE
            stats.num_of_weight_reads   = self.statebuffer.circbuf_weights.DRAM_elem_read
            stats.num_of_reads_elems          = self.statebuffer.circbuf_weights.DRAM_elem_read + self.statebuffer.circbuf_ifmaps.DRAM_elem_read + self.statebuffer.circbuf_scratch.DRAM_elem_read
            stats.num_of_writes_elems         = self.statebuffer.circbuf_scratch.DRAM_elem_written
            #stats.total_dram_transfer_cycles = stats.num_of_reads_elems*self.statebuffer.circbuf_weights.item_sz / 10 # assuming 10GB/s BW available per TPB
            stats.total_pearray_latency_cycles  = tpb.pearray.total_pearray_latency_cycles
            stats.total_pearray_wave_elems    = tpb.pearray.total_pearray_wave_elems
            stats.minibatch_multiplier   = self.statebuffer.circbuf_residue.minibatch_multiplier
            stats.batching_in_wave       = self.pearray.batching_in_wave 
            stats.num_of_ops_executed    = self.pearray.num_of_ops_executed
            #if (stats.num_waves_x_max_pe_ops != 0):
            #    stats.wave_op_efficiency    = self.pearray.num_of_ops_executed / stats.num_waves_x_max_pe_ops
            #else:
            #    stats.wave_op_efficiency    = 0
            if (self.statebuffer.circbuf_weights.dram_data != []):
                stats.total_weight_elems   = self.statebuffer.circbuf_weights.dram_data.size
                stats.total_weight_ifmaps_elems         = self.statebuffer.circbuf_weights.dram_data.size \
                                            + (self.statebuffer.circbuf_ifmaps.dram_data.size if self.statebuffer.circbuf_ifmaps.DRAM_elem_read > 0 else 0)
                stats.ideal_compute_to_load_ratio = 2 * self.pearray.num_of_ops_executed / self.statebuffer.circbuf_weights.dram_data.size
                stats.actual_to_min_read_ratio = stats.num_of_reads_elems / stats.total_weight_ifmaps_elems
            else:
                stats.total_weight_elems  = 0
                stats.total_weight_ifmaps_elems        = 0
                stats.ideal_compute_to_load_ratio = 0.0
                stats.actual_to_min_read_ratio = 0
            #print("num_waves_x_max_pe_ops: ",stats.num_waves_x_max_pe_ops, "num_of_ops_executed: ", self.pearray.num_of_ops_executed, "wave_op_efficiency", stats.wave_op_efficiency)
            print("num_of_reads_elems: ",stats.num_of_reads_elems,"total_weight_ifmaps_elems: ",stats.total_weight_ifmaps_elems, "actual_to_min_read_ratio: ",stats.actual_to_min_read_ratio,\
                  "weights_read:",self.statebuffer.circbuf_weights.DRAM_elem_read,"ifmaps_read: ",self.statebuffer.circbuf_ifmaps.DRAM_elem_read)
            print("min weight read:",self.statebuffer.circbuf_weights.dram_data.size, "min ifmap read: ",self.statebuffer.circbuf_ifmaps.dram_data.size)
            print("ideal_compute_to_load_ratio: ",stats.ideal_compute_to_load_ratio)
            print("number_of_waves: ",self.pearray.num_wave_fp16_mm)
            #print("STATS summary %s: %d %d %d %d %d %d %d %d %f %f %f %d %f"%(layer_name, self.pearray.num_wave_fp16_mm, num_of_wave_ops, self.pearray.num_of_ops_executed, wave_op_efficiency, num_of_weight_reads, num_of_reads_elems, num_of_writes_elems, num_of_weights_elem, total_weight_ifmaps_elems, actual_to_min_read_ratio, ideal_compute_to_load_ratio, tpb.pearray.total_pearray_latency_cycles, total_dram_transfer_cycles))

    # Execute softmax (second part, which includes Sum, Reciprocate, Scale)
    def execute_softmax2(self, inputs, result_file):
        # create and save ones as weights file, then load them back
        ones_shape = [op_list[0].C, 1, 1, 1]
        ones_tensor = np.ones(ones_shape, dtype=op_list[0].data_type)
        ones_file = op_list[0].data['ref_file'].replace(".npy", "-ones.npy")
        if (not args.inference):
            np.save(ones_file, ones_tensor)
        weights = []
        # TODO: needs better way to load ones into weights region
        #if (op_list.has_conv):
        #    op_list[0].data['kernel_file'] = ones_file
        #    op_list[0].data['kernel_format'] = "CRSM"
        #    op_list[0].data['kernel_shape'] = ones_shape
        #    weights = self.statebuffer.file_mapper.load_data(op_list.conv_op)

        # reallocate statebuffer resources
        #self.statebuffer.reallocate_capacities()

        # initial psum bank is 0
        op_list.conv_op.set_psum_bank(tpb.pearray.last_psum_bank_used)
        # start tensor computation by clearing psum bank
        psum_add = False                               

        # use conv to sum the exponential results
        # wave loop ordering scheme: nmhwcRS
        for n_id in range(op_list.conv_op.n):
            for m_id in range(op_list.conv_op.m):
                for h_id in range(op_list.conv_op.h):
                    for w_id in range(op_list.conv_op.w):
                        tile_id = TileID(n_id, m_id, h_id, w_id, op_list.conv_op.n, op_list.conv_op.m, op_list.conv_op.h, op_list.conv_op.w)
                        # compute ofmap tile information (tile startx, starty, height, width)
                        op_list.conv_op.compute_ofmap_tile_info(tile_id)
                        op_list.conv_op.compute_tile_weight_bounds(weights, tile_id)
                        # loops for constructing a tile
                        for c_id in range(op_list.conv_op.c):
                            for r_id in range(op_list.conv_op.R):
                                for s_id in range(op_list.conv_op.S):
                                    wave_id = WaveID(n_id, m_id, h_id, w_id, c_id, r_id, s_id)
                                    if (args.debug > 2): print (wave_id.show())
                                    # execute PEArray matrix multiply, and add to PSUM after first wave
                                    if (op_list.execute_matmul_waveop(self, wave_id, inputs, weights, psum_add)):
                                        psum_add = True
                        # tile is done                                   
                        self.waveop_stream.last_main_waveop['stop_tensor_calc'] = True
                        self.pearray.trig_tile_done(tile_id)
                        # extract PSUM data
                        psum_bank_src = op_list.conv_op.get_psum_bank()
                        psum_temp = self.pearray.extract_psum(psum_bank_src, 0, op_list.conv_op.ofmap_full_tile_sz * op_list.conv_op.Tn)
                        # go through the remaining operations
                        psum_temp = self.pool.reciprocate(psum_temp, op_list.conv_op.M)
                        psum_bank_dst = psum_bank_src
                        tpb.pearray.write_psum(psum_bank_dst, 0, op_list.conv_op.ofmap_full_tile_sz * op_list.conv_op.Tn, psum_temp)
                        tpb.gen_recip_waveop_inline(op_list.conv_op, psum_bank_src, True, psum_bank_dst)
                        psum_bank_src = psum_bank_dst
                        # loops for final scaling
                        for c_id in range(ceildiv(op_list.conv_op.C, PEArray.NUM_COLS)):
                            wave_id = WaveID(n_id, m_id, h_id, w_id, c_id, 0, 0)
                            pearray_packed_ifmaps = op_list.conv_op.pack_wave_ifmaps(inputs, wave_id, 1, for_softmax=True)
                            scale_val = self.pearray.extract_psum(psum_bank_src, 0, 1)[0,0]
                            psum_temp = self.pool.scale(pearray_packed_ifmaps, scale_val)
                            # if operation is the last one, dump current result into a portion of final result
                            # use c_id instead of m_id because we collapsed M to 1 to do the summation
                            output_params_op = op_list.conv_op
                            dram_output_waveops = []                            
                            for z in range(op_list.conv_op.Tn):
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
                                #dram_output_waveops += self.statebuffer.circbuf_scratch.write_data_region(
                                #                            tile_id, 
                                #                            output_params_op.ofmap_tile_lower_addr[z], 
                                #                            output_params_op.ofmap_tile_upper_addr[z], 
                                #                            output_params_op.ifmap_count,   # Do we have to use IFMAP count here?
                                #                            self.waveop_stream.last_main_waveop)
                       # The scale_add destination need to be adjusted after the above writes to data region
                        if (self.waveop_stream.last_main_waveop['waveop_type'] == "ScaleAdd"):
                            sb_addr = self.statebuffer.file_mapper.get_sb_addr_from_file_addr(op_list.conv_op.ofmaps_file_params, 0, output_params_op.ofmap_tile_lower_addr[0])
                            self.waveop_stream.last_main_waveop['dst_sb_address'] = sb_addr
                        self.waveop_stream.add_outputs(dram_output_waveops)
                        if args.abstract_mem:
                            if len(dram_output_waveops) > 0:
                                self.waveop_stream.last_main_waveop = None

                        # Advance to new bank (ping-pong between 0 and 1) for PEArray, while the old bank is being processed by other engines
                        op_list.conv_op.set_psum_bank((op_list.conv_op.get_psum_bank()+1)%4)
                        tpb.pearray.last_psum_bank_used = op_list.conv_op.get_psum_bank()
                        psum_add = False

        return result

    # Execute an unfused pooling operator
    def execute_unfused_pool_op(self, batch_item):
        inputs = op_list.first_op.ifmaps_file_params.dram_data

        # load bias values
        bias = []
        if (op_list.has_biasadd):
            bias_temp = op_list.biasadd_op.bias_file_params.dram_data
            bias = bias_temp.flatten()

        # save result to create a scratch space (in DRAM), then use circular buffer load to populate params
        result = op_list.last_op.ofmaps_file_params.dram_data

        # for ResAdd/Multiply, retrieve the saved result file for one of the completed legs if it's not already loaded
        #if op_list.has_join:
        #    if (self.statebuffer.circbuf_residue.dram_data_in_file == None):
        #        self.statebuffer.circbuf_residue.load_data(op_list.join_op)
        #    elif (op_list.join_op.prev[op_list.join_op.residue_index].result_file != self.statebuffer.circbuf_residue.dram_data_in_file):
        #        self.statebuffer.circbuf_residue.reset_keep_consumers()
        #        self.statebuffer.circbuf_residue.load_data(op_list.join_op)

        # wave loop ordering scheme: nmhw
        pool_op = op_list[0]
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
                            psum_temp = self.pool.pool(pool_op.data['layer_type'], psum_fake_extract, pool_op.stride_x, pool_op.pool_window_y, pool_op.Tn, input_tilex, input_tiley, output_tilex, output_tiley)
                        elif (pool_op.data['layer_type'] == "Multiply" or pool_op.data['layer_type'] == "ResAdd"):
                            if ("mul_scalar" in pool_op.data):
                                assert (pool_op.data['layer_type'] == "Multiply")
                                psum_temp = self.pool.scale(psum_fake_extract, pool_op.data['mul_scalar'])
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
                            psum_temp = self.activate.biasadd(psum_fake_extract, bias_extracted)
                        elif re.search(r"Relu|Tanh|Sigmoid|Exp|Identity|Lrelu|Prelu", pool_op.data['layer_type']):
                            psum_temp = self.activate.act(pool_op.data['layer_type'], psum_fake_extract)
                        else:
                            print("ERROR: cannot execute %s in execute_unfused_pool_op"%pool_op.data['layer_type'])
                            exit(-1)

                        # TODO: fix waveop generation
                        dram_ifmaps_waveops = []
                        prev_waveops = []
                        for z in range(pool_op.Tn):
                           if (tile_id.m_id%2 == 0):
                                fmap_count = pool_op.ifmap_count
                                if (tile_id.m_id+1 != tile_id.m):
                                    fmap_count = 2*pool_op.ifmap_count
                                (writers, readers, waveops) = tpb.statebuffer.file_mapper.read_file_data_region(
                                                            tpb.waveop_stream.nonload_waveop_count,
                                                            tpb.waveop_stream.nonload_waveop_list,
                                                            pool_op.ifmaps_file_params,
                                                            batch_item + z,
                                                            pool_op.ifmap_wave_lower_addr[z], 
                                                            pool_op.ifmap_wave_upper_addr[z] - pool_op.ifmap_wave_lower_addr[z] + pool_op.item_sz)
                                dram_ifmaps_waveops += waveops
                                # TODO: roll this code into read_file_data_region
                                if waveops == []:
                                    accessors = writers + readers
                                    if accessors != []:
                                        latest_accessor = max(accessors)
                                        if latest_accessor >= 0:
                                            prev_waveops.append(tpb.waveop_stream.nonload_waveop_list[latest_accessor]['waveop_name'])

                        start_at_mid_part = tile_id.m_id%2 == 1
                        if (pool_op.data['layer_type'] == "AvgPool" or pool_op.data['layer_type'] == "MaxPool"):
                            self.gen_unfused_pool_waveop_inline(op_list, tile_id, dram_ifmaps_waveops, start_at_mid_part)
                       # elif (pool_op.data['layer_type'] == "Multiply" or pool_op.data['layer_type'] == "ResAdd"):
                       #     if ("mul_scalar" in pool_op.data):
                       #         self.gen_scaleadd_waveop_inline(pool_op, tile_id, False, 0, False, 0, dram_ifmaps_waveops, pool_op.data['mul_scalar'], 0.0)
                       #     else:
                       #         self.gen_join_waveop_inline(pool_op, None, tile_id, False, 0, False, 0, dram_ifmaps_waveops+dram_resadd_waveops, pool_op.ofmap_tile_lower_addr[0], start_at_mid_part)
                       # elif (pool_op.data['layer_type'] == "BiasAdd"): 
                       #     self.gen_act_waveop_inline(pool_op, None, None, tile_id, False, 0, False, 0, dram_ifmaps_waveops + dram_bias_waveops, bias_addr)
                       #     #tpb.statebuffer.circbuf_bias.free_data_region(bias_addr, bias_addr, self.waveop_stream.last_main_waveop)
                       # else:                            
                       #     self.gen_act_waveop_inline(None, pool_op, None, tile_id, False, 0, False, 0, dram_ifmaps_waveops, 0)

                        tpb.waveop_stream.last_main_waveop['previous_waveops'] += prev_waveops

                        dram_output_waveops = []                            
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
                            (writers, readers, waveops) = self.statebuffer.file_mapper.write_file_data_region(
                                                        self.waveop_stream.nonload_waveop_count - 1,    # adjust since pool waveop already generated
                                                        self.waveop_stream.nonload_waveop_list,
                                                        pool_op.ofmaps_file_params,
                                                        batch_item + z,
                                                        pool_op.ofmap_tile_lower_addr[z], 
                                                        pool_op.ofmap_tile_upper_addr[z] - pool_op.ofmap_tile_lower_addr[z] + pool_op.item_sz, 
                                                        start_at_mid_part)
                            assert(len(waveops) == 0)                            
                            # TODO: roll this code into write_file_data_region
                            accessors = writers + readers
                            prev_waveops = []
                            if accessors != []:
                                latest_accessor = max(accessors)
                                if latest_accessor >= 0:
                                    prev_waveops.append(tpb.waveop_stream.nonload_waveop_list[latest_accessor]['waveop_name'])
                                    tpb.waveop_stream.last_main_waveop['previous_waveops'] += prev_waveops
                            if (args.debug > 3): print("TRACE execute_unfused_pool_op %s: tile %s done, ifmap_tile_lower_addr %d ifmap_tile_upper_addr %d psum_bank %d, ofmap_tile_lower_addr %d ofmap_tile_upper_addr %dx"%(pool_op.data["layer_name"], tile_id.id_string(), pool_op.ifmap_wave_lower_addr[z], pool_op.ifmap_wave_upper_addr[z], -1, pool_op.ofmap_tile_lower_addr[z], pool_op.ofmap_tile_upper_addr[z]))
                        #if args.abstract_mem:
                        #    if len(dram_output_waveops) > 0:
                        #        self.waveop_stream.last_main_waveop = None
        return result

    # Execute conv and other operations in list: for each op, load parameters and perform op with input
    def execute_conv_ops(self, batch_item):
        inputs = op_list.first_op.ifmaps_file_params.dram_data
        weights = op_list.conv_op.weights_file_params.dram_data

        # load bias values
        bias = []
        if (op_list.has_biasadd):
            bias_temp = op_list.biasadd_op.bias_file_params.dram_data
            bias = bias_temp.flatten()

        # save result to create a scratch space (in DRAM), then use circular buffer load to populate params
        result = op_list.last_op.ofmaps_file_params.dram_data

        # for ResAdd/Multiply, retrieve the saved result file for one of the completed legs if it's not already loaded
        #if op_list.has_join:
        #    if (self.statebuffer.circbuf_residue.dram_data_in_file == None):
        #        self.statebuffer.circbuf_residue.load_data(op_list.join_op)
        #    elif (op_list.join_op.prev[op_list.join_op.residue_index].result_file != self.statebuffer.circbuf_residue.dram_data_in_file):
        #        self.statebuffer.circbuf_residue.reset_keep_consumers()
        #        self.statebuffer.circbuf_residue.load_data(op_list.join_op)

        # initial psum bank is 0
        op_list.conv_op.set_psum_bank(tpb.pearray.last_psum_bank_used)
        # start tensor computation by clearing psum bank
        psum_add = False                               

        # wave loop ordering scheme: nmhwcRS
        n_id = batch_item // op_list.conv_op.Tn
        if True:
            for m_id in range(op_list.conv_op.m):
                for h_id in range(op_list.conv_op.h):
                    for w_id in range(op_list.conv_op.w):
                        tile_id = TileID(n_id, m_id, h_id, w_id, op_list.conv_op.n, op_list.conv_op.m, op_list.conv_op.h, op_list.conv_op.w)
                        # compute ofmap tile information (tile startx, starty, height, width)
                        op_list.conv_op.compute_ofmap_tile_info(tile_id)
                        op_list.conv_op.compute_tile_weight_bounds(weights, tile_id)
                        # loops for constructing a tile
                        for c_id in range(op_list.conv_op.c):
                            r_id = 0
                            s_id = 0
                            while r_id < op_list.conv_op.weights_file_params.file_dims.R:
                                while s_id < op_list.conv_op.weights_file_params.file_dims.S:
                                    wave_id = WaveID(n_id, m_id, h_id, w_id, c_id, r_id, s_id)
                                    if (args.debug > 2): print (wave_id.show())
                                    # execute PEArray matrix multiply, and add to PSUM after first wave
                                    if (op_list.execute_matmul_waveop(self, wave_id, inputs, weights, psum_add)):
                                        psum_add = True
                                    s_id += op_list.conv_op.ifmaps_file_params.replicate_multiple
                                r_id += s_id//op_list.conv_op.S
                                s_id = s_id%op_list.conv_op.S
                        # tile is done                                   
                        # TODO: fix waveop generation
                        #self.waveop_stream.last_main_waveop['stop_tensor_calc'] = True
                        self.pearray.trig_tile_done(tile_id)
                        # extract PSUM data
                        psum_bank_src = op_list.conv_op.get_psum_bank()
                        psum_temp = self.pearray.extract_psum(psum_bank_src, 0, op_list.conv_op.ofmap_full_tile_sz * op_list.conv_op.Tn)
                        #x = DBG_DUMP_PSUM_COL("PSUM after PEArray: ", psum_temp, 0)
                        # go through the remaining operations
                        psum_temp = op_list.execute_tile_ops(tpb, wave_id, tile_id, psum_bank_src, bias, psum_temp)
                        #x = DBG_DUMP_PSUM_COL("PSUM after PEArray: ", psum_temp, 0)
                        # if operation is the last one, dump current result into a portion of final result
                        output_params_op = op_list.conv_op
                        if (op_list.has_pool):
                            output_params_op = op_list.pool_op
                        dram_output_waveops = []                            
                        for z in range(op_list.conv_op.Tn):
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
                            (writers, readers, waveops) = self.statebuffer.file_mapper.write_file_data_region(
                                                        self.waveop_stream.nonload_waveop_count - 1,    # adjust since pool waveop already generated
                                                        self.waveop_stream.nonload_waveop_list,
                                                        op_list.last_op.ofmaps_file_params,
                                                        batch_item + z,
                                                        output_params_op.ofmap_tile_lower_addr[z], 
                                                        output_params_op.ofmap_tile_upper_addr[z] - output_params_op.ofmap_tile_lower_addr[z] + output_params_op.item_sz, 
                                                        start_at_mid_part)
                            assert(len(waveops) == 0)                            
                            # TODO: roll this code into write_file_data_region
                            accessors = writers + readers
                            prev_waveops = []
                            if accessors != []:
                                latest_accessor = max(accessors)
                                if latest_accessor >= 0:
                                    prev_waveops.append(tpb.waveop_stream.nonload_waveop_list[latest_accessor]['waveop_name'])
                                    self.waveop_stream.last_psum_waveop[psum_bank_src]['previous_waveops'] += prev_waveops
                        #if (args.debug > 3): print("TRACE execute_conv_ops %s: tile %s done, input region type %s start %d ifmap_tile_lower_addr %d ifmap_tile_upper_addr %d psum_bank %d, output region type %s start %d ofmap_tile_lower_addr %d ofmap_tile_upper_addr %dx"%(op_list[-1].data["layer_name"], tile_id.id_string(), self.statebuffer.circbuf_ifmaps.circbuf_type, self.statebuffer.circbuf_ifmaps.start, op_list.conv_op.ifmap_tile_lower_addr[0], op_list.conv_op.ifmap_tile_upper_addr[0], psum_bank_src, self.statebuffer.circbuf_scratch.circbuf_type, self.statebuffer.circbuf_scratch.start, output_params_op.ofmap_tile_lower_addr[0], output_params_op.ofmap_tile_upper_addr[0]))
                        #if args.abstract_mem:
                        #    if len(dram_output_waveops) > 0:
                        #        self.waveop_stream.last_psum_waveop[psum_bank_src] = None
                        # Advance to new bank (ping-pong between 0 and 1) for PEArray, while the old bank is being processed by other engines
                        op_list.conv_op.set_psum_bank((op_list.conv_op.get_psum_bank()+1)%4)
                        tpb.pearray.last_psum_bank_used = op_list.conv_op.get_psum_bank()
                        psum_add = False

        return result                   

def print_stats_headers(stats, prefix):    
    for item in inspect.getmembers(stats):
        if (not re.search(r"^__", item[0])): 
            if (re.search(r"^circbuf", item[0])): 
                for j in item[1]:
                    print_stats_headers(item[1][j], j+"_")
            else:                            
                print(prefix+item[0], end=" ")

def print_stats_items(stats):    
    for item in inspect.getmembers(stats):
        if (not re.search(r"^__", item[0])): 
            if (re.search(r"^circbuf", item[0])): 
                for j in item[1]:
                    print_stats_items(item[1][j])
            else:                            
                print(item[1], end=" ")

# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kgraph", help="K-graph Json file to read")
    parser.add_argument("--wavegraph", help="Wave-graph Json file to write")
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
    args = parser.parse_args()

    print("Middle Sched v2: Running in %s mode"%(args.nname))

    if (args.debug > 5): np.set_printoptions(threshold=np.nan)

    stats_per_layer = []

    # loading Kgraph
    try:
        print("\nLoading K-graph %s"%args.kgraph)
        kgraph_json = json.load(open(args.kgraph))
    except Exception as e:
        print(e)
        exit(-1)

    # create graph from JSON file        
    kgraph = KGraph()
    kgraph.populate_from_kgraph_json(kgraph_json)

    # add forward references
    kgraph.add_forward_refs(kgraph.last_node)

    # batching machine
    batcher = BatchMachine(16, kgraph.data_type)

    # go through all layers and add the fusable operations
    tpb = TPBSched(batcher)
    #tpb.statebuffer.circbuf_bias.data_type = kgraph.data_type
    #tpb.statebuffer.circbuf_bias.item_sz = kgraph.item_sz
    result_file = None

    # obtain full list of fused ops in first pass
    fused_ops_list = FusedOpList()
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
        #result_file = last_op.data['ref_file'].replace(".npy", "-midout.npy")
        #result_file = last_op.ofmaps_file_params.file_name
        #print("Output file for layer %s is %s"%(last_op.data['layer_name'], result_file))

        # Check the first op of fused op
        first_op = op_list[0]
        first_op_type = first_op.data['layer_type'] 
        # Dissolve Input of Placeholder types
        if first_op.is_placeholder:
            first_op.result_file = first_op.data['ref_file']
            first_op.populate_ofmaps_file_params()
            prev_join_batch_item_partition_usage_sz = op_list.last_op.ofmaps_file_params.batch_item_partition_usage_sz
            # Mark the first node after Input/Placeholder as input node
            for i in first_op.next: i.is_input = True
        # Dissolve Reshape
        elif first_op.is_nop:
            for j in first_op.prev:
                if j.result_file is not None:
                    first_op.result_file = j.result_file
                    first_op.populate_ofmaps_file_params()
                    first_op.ofmaps_file_params = j.ofmaps_file_params
                    break
        else:                
            if len(fused_ops_list) > 0:
                op_list.prev = fused_ops_list[-1]
                op_list.current_batch_count = op_list.prev.next_batch_count
            fused_ops_list.append(op_list)
            op_list.partial_batch_pairup = False
            op_list.last_op.result_file = last_op.ofmaps_file_params.file_name
            print("Output file for layer %s is %s"%(op_list.last_op.data['layer_name'], op_list.last_op.result_file))
            # Check for convenient location to pair-up batch items for processing together
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
                        backtrack_op_list = fused_ops_list[-1]
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
                        backtrack_op_list = fused_ops_list[-1]
                        while (backtrack_op_list != prev_join_op_list):
                            # here we just check that the image (FMAP) sizes are the same without considering number of channels (C) 
                            if (backtrack_op_list.last_op.ofmaps_file_params.fmap_data_len == op_list.last_op.ofmaps_file_params.fmap_data_len):
                                backtrack_op_list.residue_in_scratch = True
                            backtrack_op_list = backtrack_op_list.prev
                prev_join_op_list = fused_ops_list[-1]
                prev_join_batch_item_partition_usage_sz = op_list.last_op.ofmaps_file_params.batch_item_partition_usage_sz

        #print("Fused op #%d, fmap data len %d"%(fused_op_count, op_list.last_op.ofmaps_file_params.fmap_data_len))                

        # Mark the last node as output node
        if last_op.next == []:
            last_op.is_output = True

        # TODO: put back golden_inputs option for debugging
        #if ((args.golden_inputs or args.inference) # TODO: why is args.inference here???
        #if (args.golden_inputs):
        #    # if using golden inputs, save the ref_file instead of result_file
        #    last_op.result_file = last_op.data['ref_file']
        #else:            
        #    last_op.result_file = result_file

        # print circular buffer stats
        #tpb.statebuffer.print_stats()
        #tpb.collect_stats(last_op.data['layer_name'])

        # increment count of fused ops (for ID purpose)
        fused_op_count += 1

    # Execute fused ops
    batch_count = fused_ops_list[0].first_op.ofmaps_file_params.file_dims.N
    b = 0
    Tn = 1
    while b < batch_count:
        i = 0
        while i < len(fused_ops_list):
            op_list = fused_ops_list[i]
            Tn = op_list.first_op.Tn
            capped_current_batch_count = min(batch_count, op_list.current_batch_count)
            capped_next_batch_count = min(batch_count, op_list.next_batch_count)
            for j in range(capped_current_batch_count-1, -1, -Tn):
                if (args.debug > 2): print("TRACE: executing fused op %s, batch elem %d to %d, partial_batch_pre_pairup %d, partial_batch_pairup %d, has_join %d, has_pool %d"%(op_list.last_op.data['layer_name'], b - j, b - j + Tn - 1, op_list.partial_batch_pre_pairup, op_list.partial_batch_pairup, op_list.has_join, op_list.has_pool))
                op_list.execute(b - j)
            # kaena-409: the marker must be qualified with the condition that the fused-op contains a join or fork, 
            # because the marker is set for both branches before the join 
            # (the fork condition also must be considered for the first MaxPool, since we double-up there too).
            if op_list.partial_batch_pairup and (op_list.has_join or op_list.has_pool):
                if (b % capped_next_batch_count) == (capped_next_batch_count - 1):
                    if (args.debug > 2): print("TRACE: batch element %d is last of the next partial-batch group (count %d), continuing to next pairup location"%(b, capped_next_batch_count))
                    i += 1                    
                else:
                    #f (b%2) == 0:
                    i = 0
                    b += Tn
                    if (args.debug > 2): print("TRACE: go back to beginning for batch element %d"%(b))
            else:                    
                i += 1                    
        b += Tn

    # write out wavegraph           
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
        wavegraph = KGraph()
        wavegraph.populate_from_kgraph_json(wavegraph_json)

        # check for SBAtomFile nodes with no input
        if (args.debug > 2):
            print("DBG: check for all SBAtomFile nodes with no input")
            for i in wavegraph.node_dict:
                entry = wavegraph.node_dict[i]
                if 'waveop_type' in entry.data:
                    if entry.data['waveop_type'] == "SBAtomFile":
                        if entry.data['previous_waveops'] == []:
                            print(entry.data['waveop_name'])

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

    # stats printing
    if (args.debug > 3):
        #print("STATS summary headings: num_wave num_waves_x_max_pe_ops num_of_ops_executed num_of_weight_reads num_of_reads_elems num_of_writes_elems total_weight_elems total_weight_ifmaps_elems actual_to_min_read_ratio ideal_compute_to_load_ratio wave_op_efficiency total_pearray_latency_cycles total_dram_transfer_cycles")
        printed_header = False
        print("STATS:", end=" ")
        for i in range(len(stats_per_layer)):
            if (not printed_header):
                print_stats_headers(stats_per_layer[i], "")
                printed_header = True
                print("")                        
            print("STATS:", end=" ")
            print_stats_items(stats_per_layer[i])
            print("")                        

    # check for comparison errors
    if (tpb.num_mismatches > 0):
        print("\nFAILED (num mismatches %d)"%tpb.num_mismatches)
    else:        
        print("\nPASSED")
