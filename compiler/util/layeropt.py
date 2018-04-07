import json
import os
import math
import re
import numpy as np
import copy
import argparse
import inspect
from layeropt_utils import CircbufPtrs
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

    def __init__(self):
        self.psum_buf = np.zeros((self.PSUM_NUM_BANKS, self.MAX_WAVE_SIZE, self.NUM_COLS), dtype=np.float32)
        self.Tn = 0

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
    SB_ATOM_SZ = 1024 # can be down to 256B for maximum DMA efficiency
    #SB_ATOM_SZ = 2048 # can be down to 256B for maximum DMA efficiency
    SB_PARTITION_SZ = 96*SB_ATOM_SZ # 96KB per partition
    SB_NUM_1K_ATOMS = SB_PARTITION_SZ//SB_ATOM_SZ
    SB_NUM_64B_MORSELS = SB_PARTITION_SZ // 64

    def __init__(self):
        self.circbuf_caps = { "ifmaps"  : 27*self.SB_ATOM_SZ, 
                              "weights" : (self.SB_NUM_1K_ATOMS-27-27-27-1)*self.SB_ATOM_SZ, 
                              "residue" : 27*self.SB_ATOM_SZ, 
                              "bias"    : 1*self.SB_ATOM_SZ, 
                              "scratch" : 27*self.SB_ATOM_SZ}
        #self.data = np.zeros((self.SB_NUM_PARTITIONS, self.SB_PARTITION_SZ))
        # initial sizes and start address, will be readjusted after loading data
        self.circbuf_ifmaps  = CircularBuffer(self, "ifmaps",  self.circbuf_caps["ifmaps"],  self.SB_ATOM_SZ, 0)
        self.circbuf_weights = CircularBuffer(self, "weights", self.circbuf_caps["weights"], self.SB_ATOM_SZ, self.circbuf_ifmaps.total_size)
        self.circbuf_residue = CircularBuffer(self, "residue", self.circbuf_caps["residue"], self.SB_ATOM_SZ, self.circbuf_weights.start + self.circbuf_weights.total_size)
        self.circbuf_bias    = CircularBuffer(self, "bias",    self.circbuf_caps["bias"],    self.SB_ATOM_SZ, self.circbuf_residue.start + self.circbuf_residue.total_size)
        self.circbuf_scratch = CircularBuffer(self, "scratch", self.circbuf_caps["scratch"], self.SB_ATOM_SZ, self.circbuf_bias.start + self.circbuf_bias.total_size)
        # dictionary of saved result files, needed if the files are to be reread
        self.saved_result_files = {}
        self.chunk2saved_map = {}   # holds all the saved atoms
        self.outfile2atomsz_map = {} # holds the atom size for each output file # TODO: move to global area when we do swapping of regions
        self.consumer_of_64byte_morsel = ["" for i in range(self.SB_NUM_64B_MORSELS)]

    def keep_scratch_and_reset(self, begin_of_first_leg = False, end_of_first_leg = False):        
        # - At the completion of first fused op in the first leg of fork, move IFMAPs to Residue buffer, and scratch to IFMAPs
        # - When last operation in fusedop is not ResAdd, but next operation is ResAdd, then back-track, but first move Residue to IFMAPs, and scratch to Residue
        # - If both is true, just move scratch to Residue (IFMAPs is kept as input to next leg)
        # - When last operation is ResAdd, read data from Residue buffer to sum with PSUM data
        self.circbuf_scratch.circbuf_ptrs.reset_ptrs_clear_endsink_mode()
        if (begin_of_first_leg and end_of_first_leg):
            if (self.circbuf_residue.atom_sz != self.circbuf_scratch.atom_sz):
                self.circbuf_scratch.minibatch_multiplier *= 2
            start = self.circbuf_scratch.start
            atom_sz = self.circbuf_scratch.atom_sz
            self.circbuf_residue = self.circbuf_scratch
            self.circbuf_residue.circbuf_type = "residue"
            self.circbuf_residue.dram_data_in_file = self.circbuf_residue.dram_data_out_file
            self.circbuf_residue.dram_data_out_file = None
            self.circbuf_scratch = CircularBuffer(self, "scratch", self.circbuf_caps["scratch"], atom_sz, start)
            # keep IFMAPs
        elif (begin_of_first_leg):
            start = self.circbuf_ifmaps.start
            atom_sz = self.circbuf_ifmaps.atom_sz
            if (self.circbuf_residue.atom_sz != self.circbuf_ifmaps.atom_sz):
                self.circbuf_ifmaps.minibatch_multiplier *= 2
            self.circbuf_residue = self.circbuf_ifmaps
            self.circbuf_residue.circbuf_type = "residue"
            self.circbuf_ifmaps = self.circbuf_scratch
            self.circbuf_ifmaps.circbuf_type = "ifmaps"
            self.circbuf_ifmaps.dram_data_in_file = self.circbuf_ifmaps.dram_data_out_file
            self.circbuf_ifmaps.dram_data_out_file = None
            self.circbuf_scratch = CircularBuffer(self, "scratch", self.circbuf_caps["scratch"], atom_sz, start)
        elif (end_of_first_leg):
            start = self.circbuf_residue.start
            atom_sz = self.circbuf_residue.atom_sz
            self.circbuf_ifmaps = self.circbuf_residue
            self.circbuf_ifmaps.circbuf_type = "ifmaps"
            if (self.circbuf_residue.atom_sz != self.circbuf_scratch.atom_sz):
                self.circbuf_scratch.minibatch_multiplier *= 2
            self.circbuf_residue = self.circbuf_scratch
            self.circbuf_residue.circbuf_type = "residue"
            self.circbuf_residue.dram_data_in_file = self.circbuf_residue.dram_data_out_file
            self.circbuf_residue.dram_data_out_file = None
            self.circbuf_scratch = CircularBuffer(self, "scratch", self.circbuf_caps["scratch"], atom_sz, start)
        else:
            start = self.circbuf_ifmaps.start
            atom_sz = self.circbuf_ifmaps.atom_sz
            self.circbuf_ifmaps = self.circbuf_scratch
            self.circbuf_ifmaps.circbuf_type = "ifmaps"
            self.circbuf_ifmaps.dram_data_in_file = self.circbuf_ifmaps.dram_data_out_file
            self.circbuf_ifmaps.dram_data_out_file = None
            self.circbuf_scratch = CircularBuffer(self, "scratch", self.circbuf_caps["scratch"], atom_sz, start)
        self.circbuf_weights.reset()    # TODO: allow weights to be reused for depth-first batching
        self.circbuf_bias.reset()

    def print_stats(self):        
        self.circbuf_ifmaps.print_stats()
        self.circbuf_weights.print_stats()
        self.circbuf_residue.print_stats()
        self.circbuf_bias.print_stats()
        self.circbuf_scratch.print_stats()

    def reset_all(self):        
        self.circbuf_ifmaps.reset()
        self.circbuf_weights.reset()
        self.circbuf_residue.reset()
        self.circbuf_bias.reset()
        self.circbuf_scratch.reset()


##################################################################################
class CircularBuffer:
    def __init__(self, parent, circbuf_type, total_size, atom_sz, start):
        self.parent = parent
        self.atom_sz = atom_sz
        self.capacity = total_size // atom_sz
        self.total_size = self.capacity*self.atom_sz
        self.start = start
        self.circbuf_type = circbuf_type
        #self.head_pointer = 0
        #self.tail_pointer = 0
        self.reset()
        self.DRAM_atoms_read = 0
        self.DRAM_atoms_read_short = 0
        self.DRAM_elem_read = 0
        self.DRAM_elem_written = 0
        self.DRAM_atoms_written = 0
        self.minibatch_multiplier = 1
        self.last_biasweight_waveop = ""

    def reset(self):
        #self.head_pointer = self.tail_pointer
        #self.head_pointer = 0
        #self.tail_pointer = 0
        self.current_atom_id = 0
        self.atom_data_sz = self.atom_sz
        self.need_spare_atoms = 1
        self.need_skip_atoms = False
        self.num_kickout_atoms = 0
        #self.num_kickout_atoms = 4 if self.circbuf_type == "scratch" else 0
        #self.num_kickout_atoms = self.capacity
        self.circbuf_ptrs = CircbufPtrs(self.capacity, self.num_kickout_atoms)
        self.num_allocated_atoms = 0
        self.num_evicted_atoms = 0
        self.max_count = 0
        self.allocated = [False for x in range(self.capacity)]
        self.skipped = [False for x in range(self.capacity)]
        self.consumer_of_freed_atom = [None for x in range(self.capacity)]
        self.dram_data_in_file = None
        self.dram_data_out_file = None
        self.dram_data = np.empty([1,1,1,1])
        self.dram_data_len = 0
        self.ifmap_data_len = 0
        self.ofmap_data_len = 0
        self.tracked_lower_addr = -1
        self.tracked_lower_addr_chunked = 0
        self.layer_name = ""
        self.layer_name_for_save = ""
        self.layer_type = "Output"
        self.layer_format = ""
        self.layer_shape = []
        self.chunk2atom_map = {}
        self.chunk2spare_map = {}
        self.chunk2skip_map = {}
        self.item_sz = 2
        self.data_type = 'float16'
        self.circbuf_stats = CircbufStats()

    def get_chunk_addr(self, addr):
        return addr // self.atom_data_sz

    def get_atom_offset(self, addr):
        return addr % self.atom_data_sz

    def get_sb_address(self, addr):
        addr_chunked = self.get_chunk_addr(addr) 
        if (addr_chunked in self.chunk2atom_map):
            sb_address = self.start + self.chunk2atom_map[addr_chunked]*self.atom_sz + self.get_atom_offset(addr)
            assert (sb_address < StateBuffer.SB_PARTITION_SZ)
            return sb_address
        else:
            print("WARNING %s: addr/atom_data_sz %d (addr %d) not found in chunk2atom_map of %s (returning -1):"%(self.circbuf_type, addr_chunked, addr, self.layer_name))
            for i in self.chunk2atom_map.keys():
                print("     %s: %d"%(i, self.chunk2atom_map[i]))
            return -1                

    def load_data(self, op, file_name = None):
        fmap_full_tiley_sz = 0
        fmap_full_tilex_sz = 0
        filter_x = 1
        stride_x = 1
        self.layer_name = op.data['layer_name']
        self.layer_type = op.data['layer_type']
        if (self.circbuf_type == "weights"):                
            self.dram_data_in_file = op.data['kernel_file']
            self.layer_format = op.data['kernel_format']
            self.layer_shape = op.data['kernel_shape']
        elif (self.circbuf_type == "ifmaps"):                
            self.layer_type = "Input" #op.data['layer_type']
            if (op.data['layer_type'] == "Conv" or op.data['layer_type'] == "MatMul"):
                fmap_full_tiley_sz = op_list.conv_op.ofmap_full_tiley_sz * op_list.conv_op.stride_y
                fmap_full_tilex_sz = op_list.conv_op.ofmap_full_tilex_sz * op_list.conv_op.stride_x
                filter_x = op_list.conv_op.S
                stride_x = op_list.conv_op.stride_x
            elif (op.data['layer_type'] == "AvgPool" or op.data['layer_type'] == "MaxPool"):
                fmap_full_tiley_sz = op_list.pool_op.ofmap_full_tiley_sz * op_list.pool_op.stride_y
                fmap_full_tilex_sz = op_list.pool_op.ofmap_full_tilex_sz * op_list.pool_op.stride_x
                filter_x = op_list.pool_op.pool_window_x
                stride_x = op_list.pool_op.stride_x
            else:
                fmap_full_tiley_sz = op.ofmap_full_tiley_sz
                fmap_full_tilex_sz = op.ofmap_full_tilex_sz
                filter_x = 1
                stride_x = 1
            for j in op.prev:
                if j.data['layer_name'] in self.parent.saved_result_files:
                    self.dram_data_in_file = self.parent.saved_result_files[j.data['layer_name']]
                    self.layer_name = j.data['layer_name']
                    self.layer_format = j.data['ofmap_format']
                    self.layer_shape = j.data['ofmap_shape']
                    break
        elif (self.circbuf_type == "bias"):                
            for j in op.prev:
                if (j.data['layer_type'] == "Const"): # assert sizes can be flattened
                    self.dram_data_in_file = j.data['ref_file']
                    self.layer_name = j.data['layer_name']
                    self.layer_format = j.data['ofmap_format']
                    self.layer_shape = j.data['ofmap_shape']
                    break
        elif (self.circbuf_type == "scratch"):
            self.layer_type = "Scratch" #op.data['layer_type']
            if (op.next == []):
                self.layer_type = "Output"
            # Ensure atom data size is multiple of tile Y for pooling
            if (op_list.has_pool):
                fmap_full_tiley_sz = op_list.pool_op.ofmap_full_tiley_sz
                data_type = op_list.pool_op.data_type
            else:    
                fmap_full_tiley_sz = op.ofmap_full_tiley_sz
                data_type = op.data_type
            if (op.data['layer_type'] == "ResAdd" and file_name == None):
                assert(self.dram_data_in_file != None)  # make sure that scratch has been initialized with output data
                for j in op.prev:
                    if j.data['layer_name'] in self.parent.saved_result_files:
                        self.dram_data_in_file = self.parent.saved_result_files[j.data['layer_name']]
                        self.layer_name = j.data['layer_name']
                        self.layer_format = j.data['ofmap_format']
                        self.layer_shape = j.data['ofmap_shape']
                        break
            else:
                assert(file_name != None)
                self.dram_data_in_file = file_name
                self.layer_format = op.data['ofmap_format']
                self.layer_shape = op.data['ofmap_shape']
                empty_tensor = np.zeros(self.layer_shape, dtype=data_type)
                np.save(self.dram_data_in_file, empty_tensor)
        assert (self.dram_data_in_file != None)
        self.load_file(self.dram_data_in_file, fmap_full_tiley_sz, fmap_full_tilex_sz, filter_x, stride_x)
        if (file_name != None):
            self.dram_data_out_file = file_name
            self.layer_name_for_save = op.data['layer_name']
        return self.dram_data

    def load_file(self, file, fmap_full_tiley_sz = 0, fmap_full_tilex_sz = 0, filter_sz=1, stride_sz=1):
        self.dram_data_in_file = file
        self.dram_data = np.load(self.dram_data_in_file)
        self.item_sz = self.dram_data.dtype.itemsize   
        self.data_type = self.dram_data.dtype.name
        self.dram_data_len = self.dram_data.size * self.item_sz
        self.replicate_multiple = 1
        # Determine the actual amount of data per atom
        # TODO: come up with a better formula for atom_data_sz to take care of all cases
        # Constraints for atom_data_sz for Conv IFMAPs/OFMAPs: 
        #   * less than or equal to 1KB
        #   * IFMAP: multiple of H*W, else multiple of W (for CNHW)
        #   * IFMAP: one H*W, else multiple of W (for NCHW)
        #   * OFMAP: multiple of E*F, else multiple of OFMAP tile size
        #   * Filter: multiple of R*S*M, else multiple of S*M, else multiple of M
        #   * different FMAPs in one batch will be in different atoms (for now)
        #   * different FMAPs folds will be in different atoms (for now)
        # TODO: refactor the following function since it is used in multiple places
        if (self.circbuf_type == "weights"):            
            if (self.layer_format == "CRSM"):
                C, R, S, M = self.dram_data.shape
            elif (self.layer_format == "CM"):
                C, M = self.dram_data.shape
                R, S = 1, 1
                self.dram_data = self.dram_data.reshape((C, R, S, M))
            else:
                print("ERROR: wrong weights format %s"%self.layer_format)
                exit(-1)
            assert(C * R * S * M * self.item_sz == self.dram_data_len)                
            self.total_filter_size = R * S
            #if (C < PEArray.NUM_ROWS and (R > 1 or S > 1)):
            #   self.replicate_multiple = min(PEArray.NUM_ROWS//C, self.total_filter_size)
            # Here ifmap is RSM
            self.ifmap_data_len = self.dram_data_len//C
            m_data_len = M * self.item_sz
            sm_data_len = S * m_data_len
            # Folding multiple: if too high (for FP32, it is 16), there's alot of reads (more than allocated) just to perform the first matmul
            # Just scale down by half to fit
            folding_multiple = (C//PEArray.NUM_ROWS) * (M//PEArray.NUM_COLS)
            atom_sz_for_computation = StateBuffer.SB_ATOM_SZ
            if (folding_multiple > 16):
                atom_sz_for_computation = StateBuffer.SB_ATOM_SZ // 2
            # If RSM is less than atom, use that as atom size                
            if (self.ifmap_data_len <= atom_sz_for_computation):
                self.atom_data_sz = self.ifmap_data_len
            # Else find the largest   
            elif (sm_data_len <= atom_sz_for_computation):
                multiple = atom_sz_for_computation // sm_data_len
                self.atom_data_sz = sm_data_len * min(R, multiple)
            elif (m_data_len <= atom_sz_for_computation):
                multiple = atom_sz_for_computation // m_data_len
                self.atom_data_sz = m_data_len * min(S, multiple)
            else:
                self.atom_data_sz = atom_sz_for_computation
        else:
            if (self.layer_format == 'NCHW'):
                N, C, H, W = self.dram_data.shape
            elif (self.layer_format == 'NC'):
                N, C = self.dram_data.shape
                H, W = 1, 1
                self.dram_data = self.dram_data.reshape((N, C, H, W))
            elif (self.layer_format == 'C'):
                assert (self.circbuf_type == "bias")
                C = self.dram_data.shape[0]
                N, H, W = 1, 1, 1
                self.dram_data = self.dram_data.reshape((N, C, H, W))
            elif (self.layer_format == 'CNHW'):    
                C, N, H, W = self.dram_data.shape
            else:
                print("ERROR in load_file: Unrecognized layer %s type %s format %s"%(self.layer_name, self.layer_type, self.layer_format))
                exit(-1)
            assert(N * C * H * W * self.item_sz == self.dram_data_len)                
            self.ifmap_data_len = self.dram_data_len//(N*C)
            # layer_shape is the ofmap_shape, in the format of N, M, E, F
            if (self.layer_format == 'NCHW' or self.layer_format == 'CNHW'):
                self.ofmap_data_len = self.layer_shape[2]*self.layer_shape[3]*self.item_sz
            elif (self.layer_format == 'NC' or self.layer_format == 'C'):
                self.ofmap_data_len = self.item_sz
            ifmap_width_data_len = W * self.item_sz
            # reset atom_sz if loading from previously saved file
            if (self.dram_data_in_file in tpb.statebuffer.outfile2atomsz_map):
                self.atom_sz = tpb.statebuffer.outfile2atomsz_map[self.dram_data_in_file]
            # make atom size multiple of IFMAP if IFMAP is smaller than default atom size (CNHW)
            #if (self.ifmap_data_len <= self.atom_sz):
            #    multiple = self.atom_sz // self.ifmap_data_len
            #    self.atom_data_sz = self.ifmap_data_len * multiple
            # For NCHW, just use ifmap size as atom size (see rule above: "different FMAPs folds will be in different atoms")
            if (self.ifmap_data_len <= self.atom_sz):
                self.atom_data_sz = self.ifmap_data_len
            # Cannot handle crossing atom gaps for case where number of IFMAPs is larger than PEArray rows, and filter size > 1
            elif (self.layer_type == 'Input' and C > 128 and filter_sz > 1):
                print("ERROR %s: cannot yet handle case where number of IFMAPs > 128, and filter size is > 1"%(self.circbuf_type))
                exit(-1)
            # To prevent crossing atom gaps for FMAPs, make it contiguous
            elif (self.layer_type == 'Input' 
                    and (C <= 128 or stride_sz > 1 or filter_sz > 1)):  # make contiguous if not folding, or folding but stride > 1, or filter size > 1
                self.atom_data_sz = self.atom_sz
                # need spare atoms for large images
                self.need_spare_atoms = max(1, ceildiv(fmap_full_tiley_sz * fmap_full_tilex_sz * self.item_sz, self.atom_sz))
                if self.circbuf_type == "scratch" and self.num_kickout_atoms > 0:
                        self.num_kickout_atoms = self.need_spare_atoms + self.num_kickout_atoms
                print("Reserve %d as spare atoms"%self.need_spare_atoms)
                if (C > 128):
                    self.need_skip_atoms = True
                    print("Reserving last atom for each wave load as skip atom")
            # make atom size multiple of width data length if it is smaller than default atom size
            elif (ifmap_width_data_len <= self.atom_sz):
                if (stride_sz > 1 or filter_sz > 1):
                    self.need_spare_atoms = max(1, ceildiv(fmap_full_tiley_sz * fmap_full_tilex_sz * self.item_sz, self.atom_sz))
                    if self.circbuf_type == "scratch" and self.num_kickout_atoms > 0:
                        self.num_kickout_atoms = self.need_spare_atoms + self.num_kickout_atoms
                multiple = self.atom_sz // ifmap_width_data_len
                multiple = min(H, multiple)
                if (fmap_full_tiley_sz != 0):
                    if (fmap_full_tiley_sz < multiple):
                        multiple = (multiple//fmap_full_tiley_sz)*fmap_full_tiley_sz
                self.atom_data_sz = ifmap_width_data_len * min(H, multiple)
            else:
                self.atom_data_sz = self.atom_sz
        # make atom_sz same as atom_data_sz
        self.capacity = self.total_size//self.atom_data_sz
        self.allocated = [False for x in range(self.capacity)]
        self.skipped = [False for x in range(self.capacity)]
        self.consumer_of_freed_atom = [None for x in range(self.capacity)]
        self.circbuf_ptrs.reset_full(self.capacity, self.num_kickout_atoms)
        self.atom_sz = self.atom_data_sz
        print("%s: Loaded %s for layer %s, first data is %f, data size is %d bytes, atom size %d bytes, atom data size %d bytes, replicate multiple %d"%(self.circbuf_type, self.dram_data_in_file, self.layer_name, self.dram_data[0,0,0,0], self.item_sz, self.atom_sz, self.atom_data_sz, self.replicate_multiple)) 
        return self.dram_data

    def recompute_ifmaps_params(self, op):
        self.layer_type = "Input" #op.data['layer_type']
        if (op.data['layer_type'] == "Conv" or op.data['layer_type'] == "MatMul"):
            fmap_full_tiley_sz = op_list.conv_op.ofmap_full_tiley_sz * op_list.conv_op.stride_y
            fmap_full_tilex_sz = op_list.conv_op.ofmap_full_tilex_sz * op_list.conv_op.stride_x
            filter_x = op_list.conv_op.S
            stride_x = op_list.conv_op.stride_x
        elif (op.data['layer_type'] == "AvgPool" or op.data['layer_type'] == "MaxPool"):
            fmap_full_tiley_sz = op_list.pool_op.ofmap_full_tiley_sz * op_list.pool_op.stride_y
            fmap_full_tilex_sz = op_list.pool_op.ofmap_full_tilex_sz * op_list.pool_op.stride_x
            filter_x = op_list.pool_op.pool_window_x
            stride_x = op_list.pool_op.stride_x
        else:
            fmap_full_tiley_sz = op.ofmap_full_tiley_sz
            fmap_full_tilex_sz = op.ofmap_full_tilex_sz
            filter_x = 1
            stride_x = 1
        if (stride_x > 1 or filter_x > 1):
            self.need_spare_atoms = max(1, ceildiv(fmap_full_tiley_sz * fmap_full_tilex_sz * self.item_sz, self.atom_sz))
            if self.circbuf_type == "scratch" and self.num_kickout_atoms > 0:
                self.num_kickout_atoms = self.need_spare_atoms + self.num_kickout_atoms


    def gen_dram_read_waveop(self, wave_id, atom_id, chunk_id, fmap_count, ifmaps_replicate=False):
        offset_in_file = chunk_id*self.atom_data_sz
        length = self.atom_data_sz
        # for scratch buffer (output), if we have to load, then use the m_id (OFMAP fold) instead of c_id (IFMAP fold)
        if (self.circbuf_type == "scratch" or self.circbuf_type == "residue"):              
            fmap_fold_idx = wave_id.m_id//2
            fmap_data_len = self.ofmap_data_len
            start_at_mid_part = (wave_id.m_id%2) == 1
        else:            
            fmap_fold_idx = wave_id.c_id
            fmap_data_len = self.ifmap_data_len
            start_at_mid_part = False 
        adjust_for_folding = fmap_fold_idx * fmap_data_len * PEArray.NUM_ROWS
        offset_in_fold = offset_in_file - adjust_for_folding
        # if address is larger than IFMAP size (H*W) for the case that IFMAP size is larger than Atom Data Size,
        # then try to get the modulo; but if the modulo is 0, then keep length = Atom Data Size
        if ((offset_in_fold + length) > fmap_data_len and fmap_data_len > self.atom_data_sz):
            length = fmap_data_len % self.atom_data_sz
        if (length == 0): length = self.atom_data_sz
        assert (length > 0)           
        # collect stats
        if (args.debug > 1):
            self.DRAM_elem_read += length * fmap_count / self.item_sz
            self.DRAM_atoms_read += 1
            self.circbuf_stats.sb_all_channels_memcpys_in += fmap_count
            if (length < self.atom_data_sz):
                self.DRAM_atoms_read_short += 1
        #print("gen_dram_read_waveop - DRAM_elem_read: ", self.DRAM_elem_read, "length: ", length, "fmap_count: ",fmap_count)
        #print("fmap_data_len",fmap_data_len, "atom_data_sz",self.atom_data_sz)
        #print("chunk_id", chunk_id, "offset", offset)
        if (args.golden_inputs):            
            simout_file = self.dram_data_in_file.replace("-midout.", ".")
        else:            
            simout_file = self.dram_data_in_file.replace("-midout.", "-simout.")
        waveop_name = self.layer_name+"/SBAtomFile_%s_%d_%s"%(self.circbuf_type, atom_id, wave_id.id_string())           
        sb_addr = self.start + atom_id*self.atom_sz
        # add dependency if the chunk belongs to a saved atom
        previous_waveops = []
        if (self.circbuf_type == "weights"): # TODO: if space is reused for other regions, need to apply to other regions
            for i in range(sb_addr//64, (sb_addr + length)//64):
                if tpb.statebuffer.consumer_of_64byte_morsel[i] != "":
                    previous_waveops.append(tpb.statebuffer.consumer_of_64byte_morsel[i])
                    tpb.statebuffer.consumer_of_64byte_morsel[i] = ""
                    break
        # string bias reads together (TODO: include weights?)
        if (self.circbuf_type == "bias"):  # or self.circbuf_type == "weights"):
            if (self.last_biasweight_waveop != "" and len(previous_waveops) == 0):
                previous_waveops.append(self.last_biasweight_waveop)
            self.last_biasweight_waveop = waveop_name                
        chunk_name = "%s_%d"%(simout_file, chunk_id)
        if (chunk_name in tpb.statebuffer.chunk2saved_map):
            previous_waveops.append(tpb.statebuffer.chunk2saved_map[chunk_name])
        if (args.debug > 2): print("LOAD FROM DRAM: region %s layer %s file %s start %d length %d fmap_count %d fmap_data_len %d"%(self.circbuf_type, self.layer_name, simout_file, offset_in_file, length, fmap_count, fmap_data_len))
        return {
              'previous_waveops' : previous_waveops,
              'waveop_type'      : "SBAtomFile",
              'waveop_name'      : waveop_name,
              'layer_name'       : self.layer_name,
              'sb_address'       : sb_addr,
              'data_type'        : self.data_type,
              'contain_weights'  : self.circbuf_type == "weights",
              'ref_file'         : simout_file,
              'ref_file_format'  : self.layer_format,
              'ref_file_shape'   : self.layer_shape,
              'offset_in_file'   : offset_in_file,
              'length'           : length,
              'ifmaps_replicate' : ifmaps_replicate,
              'ifmaps_fold_idx'  : fmap_fold_idx,
              'start_at_mid_part' : start_at_mid_part,
              'batch_fold_idx'   : wave_id.n_id,
              'ifmap_count'      : fmap_count,  # if this is larger than C, replicate fmap_count/C times
              'partition_step_bytes': fmap_data_len,
            }

    def gen_dram_save_waveop(self, tile_id, atom_id, chunk_id, ofmap_count):
        offset_in_file = chunk_id*self.atom_data_sz
        length = self.atom_data_sz
        adjust_for_folding = (tile_id.m_id//2) * self.ofmap_data_len * PEArray.NUM_ROWS
        offset_in_fold = offset_in_file - adjust_for_folding
        # if address is larger than IFMAP size (H*W) for the case that IFMAP size is larger than Atom Data Size,
        # then try to get the modulo; but if the modulo is 0, then keep length = Atom Data Size
        if ((offset_in_fold + length) > self.ofmap_data_len and self.ofmap_data_len > self.atom_data_sz):
            length = self.ofmap_data_len % self.atom_data_sz
            if (length == 0): length = self.atom_data_sz
        assert (length > 0)            
        # collect stats
        if (args.debug > 1):
            self.DRAM_elem_written += length * ofmap_count / self.item_sz
            self.DRAM_atoms_written += 1
            self.circbuf_stats.sb_all_channels_memcpys_out += ofmap_count*((tile_id.m_id%2)+1)
        # if this is last chunk in OFMAP, mark it as last
        last_atom_of_file = (tile_id.m_id+1 == tile_id.m) and (ceildiv(offset_in_fold+length, self.atom_data_sz) == ceildiv(self.ofmap_data_len, self.atom_data_sz))
        #print("m_id %d m %d offset_in_fold %d length %d ofmap_data_len %d last %d"%(tile_id.m_id, tile_id.m, offset_in_fold, length, self.ofmap_data_len, last_atom_of_file))
        # use "simout" tag for Back-end/Inkling result file
        assert(self.dram_data_out_file != None)
        simout_file = self.dram_data_out_file.replace("-midout.", "-simout.")
        waveop_name = self.layer_name_for_save + "/SBAtomSave_%s_%d_%s"%(self.circbuf_type, atom_id, tile_id.id_string())
        self.parent.chunk2saved_map["%s_%d"%(simout_file, chunk_id)] = waveop_name
        self.parent.outfile2atomsz_map[self.dram_data_out_file] = self.atom_sz
        fmap_count = ofmap_count*((tile_id.m_id%2)+1)
        if (args.debug > 2): print("SAVE TO DRAM: region %s layer %s file %s start %d length %d fmap_count %d fmap_data_len %d"%(self.circbuf_type, self.layer_name, simout_file, offset_in_file, length, fmap_count, self.ofmap_data_len))
        return {
              'previous_waveops' : [],
              'waveop_type'      : "SBAtomSave",
              'waveop_name'      : waveop_name,
              'layer_name'       : self.layer_name,
              'sb_address'       : self.start + atom_id*self.atom_sz,
              'data_type'        : self.data_type,
              'ref_file'         : simout_file,
              'ref_file_format'  : self.layer_format,
              'ref_file_shape'   : self.layer_shape,
              'offset_in_file'   : offset_in_file,
              'length'           : length,
              'start_at_mid_part' : False, #(tile_id.m_id%2) == 1,
              'ofmaps_fold_idx'  : tile_id.m_id,
              'batch_fold_idx'   : tile_id.n_id,
              'ofmap_count'      : fmap_count,
              'partition_step_bytes': self.ofmap_data_len,
              'last'             : last_atom_of_file,
            }

    def is_a_spare_atom(self, atom_id):
        return (self.need_spare_atoms > 0
                and atom_id >= self.capacity - self.need_spare_atoms
                and atom_id < self.capacity)

    def is_a_skip_atom(self, atom_id):
        return (self.need_skip_atoms and self.skipped[atom_id])
    
    def map_chunk_to_nonspare_atom(self, atom_id, chunk_id):
        consumer_of_evicted_atom = None
        for k in self.chunk2atom_map.keys():
            if (self.chunk2atom_map[k] == atom_id):
                if (args.debug > 2): print("%s: evicting %s at nonspare atom_id %d, replacing with nonspare chunk %d"%(self.circbuf_type, k, atom_id, chunk_id))
                self.num_evicted_atoms += 1
                del self.chunk2atom_map[k]
                consumer_of_evicted_atom = self.consumer_of_freed_atom[atom_id]
                self.consumer_of_freed_atom[atom_id] = None
                break
        for k in self.chunk2skip_map.keys():
            if (self.chunk2skip_map[k] == atom_id):
                if (args.debug > 2): print("%s: evicting %s at skip atom_id %d, replacing with nonspare chunk %d"%(self.circbuf_type, k, atom_id, chunk_id))
                self.num_evicted_atoms += 1
                del self.chunk2skip_map[k]
                consumer_of_evicted_atom = self.consumer_of_freed_atom[atom_id]
                self.consumer_of_freed_atom[atom_id] = None
                break
        self.chunk2atom_map[chunk_id] = atom_id
        return consumer_of_evicted_atom

    def map_chunk_to_spare_atom(self, atom_id, chunk_id):
        consumer_of_evicted_atom = None
        for k in self.chunk2spare_map.keys():
            if (self.chunk2spare_map[k] == atom_id):
                if (args.debug > 2): print("%s: evicting %s at spare atom_id %d, replacing with spare chunk %d"%(self.circbuf_type, k, atom_id, chunk_id))
                self.num_evicted_atoms += 1
                del self.chunk2spare_map[k]
                consumer_of_evicted_atom = self.consumer_of_freed_atom[atom_id]
                self.consumer_of_freed_atom[atom_id] = None
                break
        self.chunk2spare_map[chunk_id] = atom_id
        return consumer_of_evicted_atom

    def map_chunk_to_skip_atom(self, atom_id, chunk_id):
        consumer_of_evicted_atom = None
        for k in self.chunk2skip_map.keys():
            if (self.chunk2skip_map[k] == atom_id):
                if (args.debug > 2): print("%s: evicting %s at skip atom_id %d, replacing with skip chunk %d"%(self.circbuf_type, k, atom_id, chunk_id))
                self.num_evicted_atoms += 1
                del self.chunk2skip_map[k]
                consumer_of_evicted_atom = self.consumer_of_freed_atom[atom_id]
                self.consumer_of_freed_atom[atom_id] = None
                break
        self.chunk2skip_map[chunk_id] = atom_id
        return consumer_of_evicted_atom

    def read_data_region(self, wave_id, lower_addr, upper_addr, ifmap_count, ifmaps_replicate=False, start_at_mid_part=False):
        if (args.debug > 2): print("%s: read byte range %d to %d"%(self.circbuf_type, lower_addr, upper_addr))
        dram_waveops = []
        lower_addr_chunked = self.get_chunk_addr(lower_addr)
        upper_addr_chunked = self.get_chunk_addr(upper_addr)
        atom_id = -1
        if (self.atom_data_sz < self.atom_sz and lower_addr_chunked != upper_addr_chunked):
            print("ERROR %s: data region %d to %d (chunk %d to %d) is crossing gappy atom boundary!"%(self.circbuf_type, lower_addr, upper_addr, lower_addr_chunked, upper_addr_chunked));
            #exit(-1)
        # allocate the first chunk
        # check whether the starting chunk is near end of buffer; skip to beginning if it is within spare region
        if (lower_addr_chunked not in self.chunk2atom_map):
            while (self.is_a_spare_atom(self.circbuf_ptrs.get(CircbufPtrs.TAIL)) or self.is_a_skip_atom(self.circbuf_ptrs.get(CircbufPtrs.TAIL))):
                self.circbuf_ptrs.advance(CircbufPtrs.TAIL)
                if (args.debug > 2): self.circbuf_ptrs.print(self.circbuf_type)
            atom_id = self.allocate_atom()
            dram_waveops.append(self.gen_dram_read_waveop(wave_id, atom_id, lower_addr_chunked, ifmap_count, ifmaps_replicate))
            prev_consumer = self.map_chunk_to_nonspare_atom(atom_id, lower_addr_chunked)
            if (args.debug > 2): print("%s: loading chunk %d into atom %d"%(self.circbuf_type, lower_addr_chunked, atom_id))
            if (prev_consumer != None):
                dram_waveops[-1]["previous_waveops"].append(prev_consumer)
        else:
            atom_id = self.chunk2atom_map[lower_addr_chunked]
            if (args.debug > 2): print("%s: chunk %d is already mapped to atom %d"%(self.circbuf_type, lower_addr_chunked, atom_id));
        # allocate the remaining chunks
        # check whether chunk is already in the spares
        starting_spares = False
        for i in range(lower_addr_chunked+1, upper_addr_chunked+1):
            if (self.need_skip_atoms 
                    and i in self.chunk2skip_map
                    and self.chunk2skip_map[i] == atom_id+1
                    and i == upper_addr_chunked
                    ):
                atom_id = self.chunk2skip_map[i]
                if (args.debug > 2): print("%s: reusing atom_id %d as skip for chunk %d (range %d-%d)"%(self.circbuf_type, atom_id, i, lower_addr, upper_addr))
            elif i in self.chunk2atom_map:
                atom_id = self.chunk2atom_map[i]
                if (args.debug > 2): print("%s: chunk %d is already mapped to atom %d"%(self.circbuf_type, i, atom_id));
            else:
                if (self.is_a_spare_atom(self.circbuf_ptrs.get(CircbufPtrs.TAIL))):
                    starting_spares = True
                    if (i not in self.chunk2spare_map):
                        atom_id = self.allocate_atom()
                        dram_waveops.append(self.gen_dram_read_waveop(wave_id, atom_id, i, ifmap_count, ifmaps_replicate))
                        if (args.debug > 2): print("%s: loading chunk %d into atom %d"%(self.circbuf_type, i, atom_id))
                        prev_consumer = self.map_chunk_to_spare_atom(atom_id, i)
                        if (prev_consumer != None):
                            dram_waveops[-1]["previous_waveops"].append(prev_consumer)
                        self.allocated[atom_id] = False
                        self.num_allocated_atoms -= 1
                        if (args.debug > 2): print("%s: keeping atom_id %d as spare for chunk %d (range %d-%d)"%(self.circbuf_type, atom_id, i, lower_addr, upper_addr))
                    else:                        
                        if (args.debug > 2): print("%s: reusing atom_id %d as spare for chunk %d (range %d-%d)"%(self.circbuf_type, self.chunk2spare_map[i], i, lower_addr, upper_addr))
                else:
                    assert(starting_spares == False)    # indicate fragmented space (bad!)
                    # if multiple atoms used, and need skip atoms, then keep the last atom as skip-atom
                    if (self.need_skip_atoms and i == upper_addr_chunked):
                        atom_id = self.allocate_atom()
                        dram_waveops.append(self.gen_dram_read_waveop(wave_id, atom_id, i, ifmap_count, ifmaps_replicate))
                        if (args.debug > 2): print("%s: loading chunk %d into atom %d"%(self.circbuf_type, i, atom_id))
                        prev_consumer = self.map_chunk_to_skip_atom(atom_id, i)
                        if (prev_consumer != None):
                            dram_waveops[-1]["previous_waveops"].append(prev_consumer)
                        if (args.debug > 2): print("%s: keeping last atom_id %d as skip for chunk %d (range %d-%d)"%(self.circbuf_type, atom_id, i, lower_addr, upper_addr))
                        self.allocated[atom_id] = False
                        self.num_allocated_atoms -= 1
                        self.skipped[atom_id] = True
                    else:                        
                        atom_id = self.allocate_atom()
                        dram_waveops.append(self.gen_dram_read_waveop(wave_id, atom_id, i, ifmap_count, ifmaps_replicate))
                        if (args.debug > 2): print("%s: loading chunk %d into atom %d"%(self.circbuf_type, i, atom_id))
                        prev_consumer = self.map_chunk_to_nonspare_atom(atom_id, i)
                        if (prev_consumer != None):
                            dram_waveops[-1]["previous_waveops"].append(prev_consumer)
                        if (self.skipped[atom_id]):
                            self.skipped[atom_id] = False
        return dram_waveops
    
    # hit_end_addr is used in write_data_region; so it should use ofmap_data_len
    def hit_end_addr(self, tile_id, upper_addr):
        upper_addr_chunked = self.get_chunk_addr(upper_addr)
        # if upper addr is larger than IFMAP size, then it is in a different channel or batch item,
        # so use the modulo to check the end address
        upper_addr_mod = upper_addr % self.ofmap_data_len
        if ((tile_id.m_id%2)==1 or tile_id.m_id == tile_id.m-1):                
        #if (1):
            if ((upper_addr_mod == (self.ofmap_data_len - self.item_sz)) 
                    or (upper_addr == (upper_addr_chunked+1)*self.atom_data_sz - self.item_sz)):
                return True
        return False

    def is_in_kickout_range(self, atom_id):
        if (self.circbuf_type == "scratch" and self.num_kickout_atoms > 0):
            return (atom_id >= self.capacity-self.num_kickout_atoms and atom_id < self.capacity)
        else:
            return False
        #return True

    def write_data_region(self, tile_id, lower_addr, upper_addr, ofmap_count):
        if (args.debug > 2): print("%s: write byte range %d to %d"%(self.circbuf_type, lower_addr, upper_addr))
        if (self.tracked_lower_addr == -1): 
            self.tracked_lower_addr = lower_addr
            self.tracked_lower_addr_chunked = lower_addr // self.atom_data_sz
        if (args.debug > 2): print("%s: written range is now %d to %d"%(self.circbuf_type, self.tracked_lower_addr, upper_addr))
        dram_waveops = []
        lower_addr_chunked = self.get_chunk_addr(lower_addr)
        upper_addr_chunked = self.get_chunk_addr(upper_addr)
        if (self.atom_data_sz < self.atom_sz and lower_addr_chunked != upper_addr_chunked):
            print("ERROR %s: data region %d to %d (chunk %d to %d) is crossing gappy atom boundary!"%(self.circbuf_type, lower_addr, upper_addr, lower_addr_chunked, upper_addr_chunked));
        for i in range(lower_addr_chunked, upper_addr_chunked+1):
            if i not in self.chunk2atom_map:
                atom_id = self.allocate_atom()
                self.chunk2atom_map[i] = atom_id
        # assuming that we always write to the last piece of atom last, when 
        # there's a write to last piece of atom, trigger to dump to DRAM and deallocate atom
        # TODO: optimize by keep some atoms between layers
        # TODO: how to handle multiple chunks???
        if self.hit_end_addr(tile_id, upper_addr):
            if (args.debug > 2): print("%s: freeing range %d to %d"%(self.circbuf_type, self.tracked_lower_addr, upper_addr))
            for i in range(self.tracked_lower_addr_chunked, upper_addr_chunked+1):
                atom_id = self.chunk2atom_map[i]
                self.free_atom(atom_id)
                self.tracked_lower_addr = -1
                if (self.is_in_kickout_range(atom_id) 
                        or self.layer_type == "Output"
                        or args.save_layer_output
                        ):
                    dram_waveops.append(self.gen_dram_save_waveop(tile_id, atom_id, i, ofmap_count))
                    if (self.is_in_kickout_range(atom_id)):
                        del self.chunk2atom_map[i]
                        self.num_evicted_atoms += 1
        return dram_waveops

    def free_data_region(self, lower_addr, upper_addr, waveop):
        if (args.debug > 2): print("%s: free byte range %d to %d"%(self.circbuf_type, lower_addr, upper_addr))
        lower_addr_chunked = self.get_chunk_addr(lower_addr)
        upper_addr_chunked = self.get_chunk_addr(upper_addr)
        for i in range(lower_addr_chunked, upper_addr_chunked+1):
            if i in self.chunk2atom_map:
                self.free_atom(self.chunk2atom_map[i])
                self.consumer_of_freed_atom[self.chunk2atom_map[i]] = waveop["waveop_name"] 
                if (args.debug > 2): print("%s: freeing atom_id %d for chunk %d (lower_addr %d, upper_addr %d), but keep around for any subsequent read"%(self.circbuf_type, self.chunk2atom_map[i], i, lower_addr, upper_addr))
                # keep data around just in case, but allow pointers to wrap around
                #del self.chunk2atom_map[i]

    def allocate_atom(self):
        if (self.num_allocated_atoms == self.capacity):
            print ("ERROR %s: no more space during allocate_atom for layer %s!"%(self.circbuf_type, self.layer_name))
            self.print_stats()
            exit(-1)
            return -1
        self.current_atom_id = self.circbuf_ptrs.get(CircbufPtrs.TAIL)
        if (self.allocated[self.current_atom_id]):
            print("ERROR %s: allocating a still allocated (non-free) atom at atom_id %d"%(self.circbuf_type, self.current_atom_id))
            self.print_stats()
            exit(-1)
        self.allocated[self.current_atom_id] = True
        if (args.debug > 2): print ("%s: Added atom_id %d for layer %s"%(self.circbuf_type, self.current_atom_id, self.layer_name))
        self.circbuf_ptrs.advance(CircbufPtrs.TAIL)
        if (args.debug > 2): self.circbuf_ptrs.print(self.circbuf_type)
        self.num_allocated_atoms += 1
        if (self.num_allocated_atoms > self.max_count):
            self.max_count = self.num_allocated_atoms
        return self.current_atom_id            

    def print_allocated(self):
        if (args.debug > 2):
            print("start %d head pointer %d, tail pointer %d"%(self.start, self.circbuf_ptrs.get[CircbufPtrs.HEAD], self.circbuf_ptrs.get[CircbufPtrs.TAIL]))
            print("%s: "%self.circbuf_type)
            for i in self.allocated:
                print("%d"%i, end="")
            print("")

    def free_atom(self, atom_id):   
        if (self.allocated[atom_id]):
            self.allocated[atom_id] = False
            self.num_allocated_atoms -= 1
            if (args.debug > 2): print ("%s: Freed atom_id %d for layer %s"%(self.circbuf_type, atom_id, self.layer_name))
        #else:
        #    print ("ERROR %s: cannot free atom ID %d since it is unallocated for layer %s!"%(self.circbuf_type, atom_id, self.layer_name))
        #    return -1
        # garbage collection: advance head pointer until it sees allocated atom
        while (not self.allocated[self.circbuf_ptrs.get(CircbufPtrs.HEAD)]):
            if self.circbuf_ptrs.get(CircbufPtrs.HEAD) == self.circbuf_ptrs.get(CircbufPtrs.TAIL):
                break
            self.circbuf_ptrs.advance(CircbufPtrs.HEAD)
            if (args.debug > 2): self.circbuf_ptrs.print(self.circbuf_type)

    def populate_stats(self, stats):
        self.circbuf_stats.sb_tot_1partition_usage      = (len(self.chunk2atom_map) + len(self.chunk2skip_map) + len(self.chunk2skip_map) + self.num_evicted_atoms)*self.atom_sz
        self.circbuf_stats.sb_overhead_1partition_usage = (len(self.chunk2skip_map) + len(self.chunk2skip_map))*self.atom_sz
        self.circbuf_stats.sb_tot_1partition_eviction   = self.num_evicted_atoms * self.atom_sz
        self.circbuf_stats.sb_atom_sz                   = self.atom_sz
        self.circbuf_stats.sb_tot_1partition_preallocated = self.total_size
        stats.circbuf[self.circbuf_type]                = copy.copy(self.circbuf_stats)

    def print_stats(self):
        #print("STATS circular buffer type %s layer %s: input file %s output file %s"%(self.circbuf_type, self.layer_name, self.dram_data_in_file, self.dram_data_out_file))
        if (args.debug > 1):
            print("STATS circular buffer type %s layer %s: item_sz %d capacity %d kickout %d atom_size %d num_allocated_atoms %d max_count %d num_evicted_atoms %d len(chunk2atom_map) %d len(chunk2spare_map) %d len(chunk2skip_map) %d total_size %d dram_data_len %d ifmap_data_len %d ofmap_data_len %d DRAM_elem_written %d DRAM_atoms_written %d DRAM_elem_read %d DRAM_atoms_read %d DRAM_atoms_read_short %d"%(self.circbuf_type, self.layer_name, self.item_sz, self.capacity, self.num_kickout_atoms, self.atom_sz, self.num_allocated_atoms, self.max_count, self.num_evicted_atoms, len(self.chunk2atom_map), len(self.chunk2spare_map), len(self.chunk2skip_map), self.total_size, self.dram_data_len, self.ifmap_data_len, self.ofmap_data_len, self.DRAM_elem_written, self.DRAM_atoms_written, self.DRAM_elem_read, self.DRAM_atoms_read, self.DRAM_atoms_read_short))
            print("start %d head pointer %d, tail pointer %d"%(self.start, self.circbuf_ptrs.get(CircbufPtrs.HEAD), self.circbuf_ptrs.get(CircbufPtrs.TAIL)))
            #print("allocated: ", end="")
            #for i in self.allocated: print(1 if i else 0, end=" ")
            print("")

##################################################################################
# Neural network node, containing data read from JSON
class KNode:
    def __init__(self, data, item_sz, data_type):
        self.prev = []
        self.next = []
        self.data = data
        self.psum_bank_dst = 0
        self.item_sz = item_sz
        self.data_type = data_type
        self.ofmap_wave_total_elems = 0
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

    # set/get dest PSUM bank
    def set_psum_bank(self, dest):
        self.psum_bank_dst = dest
    def get_psum_bank(self):
        return self.psum_bank_dst

    # populate common parameters for Conv and Pool
    def populate_common_params(self, adjust_for_pool):
        # get input shape from previous layer's data
        for i in self.prev:
            if i.data['layer_type'] != "Const":
                input_layer = i.data
                break
        self.ifmap_shape = input_layer['ofmap_shape']
        self.internal_ifmap_shape = input_layer['ofmap_shape']
        if (input_layer['ofmap_format'] == 'NCHW'):
            self.N, self.C, self.H, self.W = input_layer['ofmap_shape']
        elif (input_layer['ofmap_format'] == 'NC'):            
            self.N, self.C = input_layer['ofmap_shape']
            self.H, self.W = 1, 1
            self.internal_ifmap_shape = [self.N, self.C, self.H, self.W]
        elif (input_layer['ofmap_format'] == 'CNHW'):            
            self.C, self.N, self.H, self.W = input_layer['ofmap_shape']
        else:
            print("ERROR in populate_common_params: Unrecognized previous layer %s format %s"%(input_layer['layer_name'], input_layer['ofmap_format']))
            exit(-1)
        # get output shape from current layer's data
        layer_info = self.data
        self.ofmap_shape = layer_info['ofmap_shape']
        self.internal_ofmap_shape = layer_info['ofmap_shape']
        if (layer_info['ofmap_format'] == 'NCHW'):
            self.N, self.M, self.E, self.F = layer_info['ofmap_shape']
        elif (layer_info['ofmap_format'] == 'NC'):            
            self.N, self.M = layer_info['ofmap_shape']
            self.E, self.F = 1, 1
            self.internal_ofmap_shape = [self.N, self.M, self.E, self.F]
        elif (layer_info['ofmap_format'] == 'CNHW'):            
            self.M, self.N, self.E, self.F = layer_info['ofmap_shape']
        else:
            print("ERROR in populate_common_params: Unrecognized current layer %s format %s"%(layer_info['layer_name'], layer_info['ofmap_format']))
            exit(-1)
        if (layer_info['layer_type'] == 'Softmax2'): self.M = 1
        if ('padding' in layer_info):            
            self.pad_north, self.pad_south = layer_info['padding'][2]
            self.pad_west, self.pad_east = layer_info['padding'][3]
        else:
            self.pad_north, self.pad_south = 0, 0
            self.pad_west, self.pad_east = 0, 0
        if ('stride' in layer_info):            
            self.stride_y = layer_info['stride'][2]
            self.stride_x = layer_info['stride'][3]
        else:
            self.stride_y = 1
            self.stride_x = 1
        # IFMAP and OFMAP total areas
        self.HW = self.H * self.W
        self.EF = self.E * self.F
        # compute batch folding and batching within wave, Tn cannot be greater than batch size N
        self.Tn = PEArray.MAX_WAVE_SIZE // self.EF
        if (self.Tn < 1):
            self.Tn = 1
        elif (self.Tn > self.N):
            self.Tn = self.N
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
            self.R, self.S = 1, 1 
        elif (layer_info['kernel_format'] == 'CRSM'):
            self.C, self.R, self.S, self.M = layer_info['kernel_shape']
        elif (layer_info['kernel_format'] == 'CM'):
            self.C, self.M = layer_info['kernel_shape']
            self.R, self.S = 1, 1
        else:
            print("ERROR: wrong weights format %s"%layer_info['kernel_format'])
            exit(-1)
        print("Conv params for layer %s: R=%d, S=%d"%(self.data['layer_name'], self.R, self.S))

    # Compute pooling params
    def populate_pooling_params(self):
        # are the dimensions from layer info correct?
        layer_info = self.data
        self.pool_window_y = layer_info['kernel_shape'][2]
        self.pool_window_x = layer_info['kernel_shape'][3]
        self.stride_y = layer_info['stride'][2]
        self.stride_x = layer_info['stride'][3]
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
            self.ofmap_tile_lower_addr.append(int(np.ravel_multi_index(
                                                (tile_id.n_id * self.Tn + z, 
                                                    tile_id.m_id//2 * PEArray.NUM_ROWS,
                                                    self.ofmap_tile_y_start, 
                                                    self.ofmap_tile_x_start),
                                            dims=self.internal_ofmap_shape) * self.item_sz))
            # NCHW
            self.ofmap_tile_upper_addr.append(int(np.ravel_multi_index(
                                                (tile_id.n_id * self.Tn + z, 
                                                    tile_id.m_id//2 * PEArray.NUM_ROWS,
                                                    self.ofmap_tile_y_start + self.ofmap_cropped_tile_height - 1, 
                                                    self.ofmap_tile_x_start + self.ofmap_cropped_tile_width - 1),
                                            dims=self.internal_ofmap_shape) * self.item_sz))

            # compute the address bounds for IFMAP tile within IFMAPs tensor
            # TODO: for Tn>1, need to have multiple bounds for each batch item
            # NCHW
            ifmap_tile_lower_coordx = self.ofmap_tile_x_start * self.stride_x
            ifmap_tile_lower_coordy = self.ofmap_tile_y_start * self.stride_y
            self.ifmap_tile_lower_addr.append(int(np.ravel_multi_index(
                                                (tile_id.n_id * self.Tn + z, 
                                                    0,
                                                    ifmap_tile_lower_coordy,
                                                    ifmap_tile_lower_coordx),
                                            dims=self.internal_ifmap_shape) * self.item_sz))

            ifmap_tile_upper_coordx = ifmap_tile_lower_coordx + self.ofmap_cropped_tile_width * self.stride_x - 1
            ifmap_tile_upper_coordy = ifmap_tile_lower_coordy + self.ofmap_cropped_tile_height * self.stride_y - 1
            if (ifmap_tile_upper_coordx > self.W-1):
                ifmap_tile_upper_coordx = self.W-1
            if (ifmap_tile_upper_coordy > self.H-1):
                ifmap_tile_upper_coordy = self.H-1
            # NCHW
            self.ifmap_tile_upper_addr.append(int(np.ravel_multi_index(
                                                (tile_id.n_id * self.Tn + z, 
                                                    (self.c-1) * PEArray.NUM_ROWS,
                                                    ifmap_tile_upper_coordy,
                                                    ifmap_tile_upper_coordx),
                                            dims=self.internal_ifmap_shape) * self.item_sz))

        self.ifmap_cropped_tile_width = ifmap_tile_upper_coordx - ifmap_tile_lower_coordx + 1
        self.ifmap_cropped_tile_height = ifmap_tile_upper_coordy - ifmap_tile_lower_coordy + 1

    def compute_tile_weight_bounds (self, weights, tile_id):        
        # Address bounds of weights used for tile
        pe_col_start = tile_id.m_id * PEArray.NUM_COLS
        pe_col_stop = min(self.M, pe_col_start + PEArray.NUM_COLS)
        self.weight_tile_lower_addr = int(np.ravel_multi_index(
                                            (0, 0, 0, pe_col_start), # CRSM
                                            dims=weights.shape) 
                                            * weights.dtype.itemsize)
        self.weight_tile_upper_addr = int(np.ravel_multi_index(
                                            (self.C-1, self.R-1, self.S-1, pe_col_stop-1), # CRSM
                                            dims=weights.shape) 
                                            * weights.dtype.itemsize)

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
        pe_row_start = fmap_folding_idx * out_array_dim_y
        pe_row_stop = min(fmap_total_count, pe_row_start + out_array_dim_y)
        assert(pe_row_start < pe_row_stop)
        r_id = wave_id.r_id
        s_id = wave_id.s_id
        last_r_id = r_id
        last_s_id = s_id
        for repl in range(replicate_multiple):
            for row in range(pe_row_start, pe_row_stop):
                #out_array[:,row] = self.pack_wave_ifmap(ifmaps[:, wave_id.c_id * out_array_dim_y + row], wave_id)
                ifmap = ifmaps[:, row]  # NCHW
                pe_row_offset = row - pe_row_start
                for z in range(self.Tn):
                    batch_id = (wave_id.n_id * self.Tn) + z
                    for x in range(self.ofmap_full_tilex_sz):
                        for y in range(self.ofmap_full_tiley_sz):
                            ifmap_tilex = (wave_id.w_id * self.ofmap_full_tilex_sz + x) * self.stride_x + s_id - self.pad_west
                            ifmap_tiley = (wave_id.h_id * self.ofmap_full_tiley_sz + y) * self.stride_y + r_id - self.pad_north
                            last_r_id = r_id
                            last_s_id = s_id
                            ifmap_addr = z * self.ofmap_full_tile_sz + y * self.ofmap_full_tilex_sz + x
                            if (ifmap_tilex < 0 or ifmap_tilex >= self.W):
                                out_array[ifmap_addr, pe_row_offset] = 0
                            elif (ifmap_tiley < 0 or ifmap_tiley >= self.H):
                                out_array[ifmap_addr, pe_row_offset] = 0
                            else:
                                out_array[ifmap_addr, pe_row_offset] = ifmap[batch_id, ifmap_tiley, ifmap_tilex]
                                # Check bounds of actual pixels within the original ifmaps for the first ifmap (which should reside in first SB partition)
                                # TODO: check how N/C are arrange in memory; batching within waves may cause different atoms to be accessed by same wave
                                # TODO: for Tn>1, need to have multiple bounds for each batch item
                                if (row == pe_row_start):                                
                                    # NCHW
                                    self.ifmap_wave_upper_addr[z] = int(np.ravel_multi_index((batch_id, row, ifmap_tiley, ifmap_tilex),
                                                                        dims=ifmaps.shape) * ifmaps.dtype.itemsize)
                                    self.ofmap_wave_upper_coordx[z] = x
                                    self.ofmap_wave_upper_coordy[z] = y
                                    if (self.ifmap_wave_lower_addr[z] < 0):
                                        self.ifmap_wave_lower_addr[z] = self.ifmap_wave_upper_addr[z]
                                        self.ofmap_wave_lower_coordx[z] = x
                                        self.ofmap_wave_lower_coordy[z] = y
                                        self.psum_bank_offset = (y * self.ofmap_full_tilex_sz + x)
                            #print("x %d y %d ifmap_tilex %d ifmap_tiley %d wave_lower_coordx %d wave_upper_coordy %d wave_upper_coordx %d wave_upper_coordy %d"%(x, y, ifmap_tilex, ifmap_tiley, self.ofmap_wave_lower_coordx, self.ofmap_wave_lower_coordy, self.ofmap_wave_upper_coordx, self.ofmap_wave_upper_coordy))                                    
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
        pe_row_start = fmap_folding_idx * out_array_dim_y
        pe_row_stop = min(fmap_total_count, pe_row_start + out_array_dim_y)
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
                self.ifmap_wave_lower_coordy[z] = (wave_id.h_id * self.ofmap_full_tiley_sz) * self.stride_y
                self.ifmap_wave_upper_coordx[z] = ((wave_id.w_id+1) * self.ofmap_full_tilex_sz) * self.stride_x + (self.pool_window_x - self.stride_x) - 1
                self.ifmap_wave_upper_coordy[z] = ((wave_id.h_id+1) * self.ofmap_full_tiley_sz) * self.stride_y + (self.pool_window_y - self.stride_y) - 1 
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
                    self.ifmap_wave_lower_addr[z] = int(np.ravel_multi_index((batch_id, row - row_adjust, self.ifmap_wave_lower_coordy[z], self.ifmap_wave_lower_coordx[z]),
                                                        dims=ifmaps.shape) * ifmaps.dtype.itemsize)
                    self.ifmap_wave_upper_addr[z] = int(np.ravel_multi_index((batch_id, row - row_adjust, self.ifmap_wave_upper_coordy[z], self.ifmap_wave_upper_coordx[z]),
                                                        dims=ifmaps.shape) * ifmaps.dtype.itemsize)
        #print(self.ifmap_wave_lower_coordx, self.ifmap_wave_lower_coordy, self.ifmap_wave_upper_coordx, self.ifmap_wave_upper_coordy)                    
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
        self.ifmap_count = self.ifmap_count * replicate_multiple           
        self.ofmap_count = pe_col_stop - pe_col_start
        r_id = wave_id.r_id
        s_id = wave_id.s_id
        last_r_id = r_id
        last_s_id = s_id
        for repl in range(replicate_multiple):
            for row in range(pe_row_start, pe_row_stop):
                for col in range(pe_col_start, pe_col_stop):
                    out_array[row - pe_row_start, col - pe_col_start] = weights[row, r_id, s_id, col] # CRSM
                    last_r_id = r_id
                    last_s_id = s_id
            s_id += 1
            if (s_id >= self.S): 
                r_id += 1
                s_id = 0
                if (r_id >= self.R): break

        self.weight_wave_lower_addr = int(np.ravel_multi_index(
                                            (pe_row_start, wave_id.r_id, wave_id.s_id, pe_col_start), # CRSM
                                            dims=weights.shape) 
                                            * weights.dtype.itemsize)
        self.weight_wave_upper_addr = int(np.ravel_multi_index(
                                            (pe_row_start, last_r_id, last_s_id, pe_col_stop-1), # CRSM
                                            dims=weights.shape) 
                                            * weights.dtype.itemsize)
        return out_array


##################################################################################
# Stream of waveops: consist of list of waveops that are fused (communicate through PSUM buffers)
class WaveopStream(list):

    def __init__(self):
        self.last_main_waveop = None
        self.waveop_name_set = set()

    def append_check(self, item):
        item_name = item['waveop_name']
        i = 0
        while (item_name in self.waveop_name_set):
            new_name = item_name + "__" + str(i)
            print("WARNING: waveop_name %s exists; so modifying name to %s before adding waveop to stream"%(item_name, new_name))
            item_name = new_name
            i += 1
        item['waveop_name'] = item_name
        self.waveop_name_set.add(item['waveop_name'])                
        self.append(item)

    def add_linked(self, waveop, side_waveops):
        input_list = []
        for i in side_waveops:
            self.append_check(i)
            input_list.append(i['waveop_name'])
        if (self.last_main_waveop != None):
            input_list.append(self.last_main_waveop['waveop_name'])
        waveop['previous_waveops'] = input_list
        self.append_check(waveop)
        self.last_main_waveop = waveop

    def add_outputs(self, waveops):
        for i in waveops:
            i['previous_waveops'].append(self.last_main_waveop['waveop_name'])
            self.append_check(i)

    def add_group(self, waveops):
        if (len(waveops)>0):
            if (self.last_main_waveop != None):
                waveops[0]['previous_waveops'].append(self.last_main_waveop)
            self = self + waveops
            self.last_main_waveop = waveops.last_main_waveop

##################################################################################
# FusedOp: consist of list of K-Nodes that are fused (communicate through PSUM buffers)
class FusedOp(list):

    def __init__(self, out_data_type):
        # only accept max one of each type in fused op
        self.has_pool = False
        self.has_resadd = False
        self.has_conv = False
        self.has_biasadd = False
        self.pool_op = None
        self.resadd_op = None
        self.conv_op = None
        self.biasadd_op = None
        self.out_data_type = out_data_type 
        self.prev_weight_wave_lower_addr = -1
        self.begin_of_first_leg = False
        self.end_of_first_leg = False

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
        elif (op.data['layer_type'] == 'ResAdd'):
            if (self.has_resadd):
                if (args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            else:
                self.resadd_op = op
                self.has_resadd = True
        elif (op.data['layer_type'] == 'BiasAdd'):
            if (self.has_biasadd):
                if (args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            else:
                self.biasadd_op = op
                self.has_biasadd = True
        if (len(op.prev) > 0):
            op.populate_common_params(adjust_for_pool=self.has_pool)
        # recompute Conv params due to constrained Pooling tile dimensions
        # (only if it is not identity pool, where window/stride are both 1)
        if (self.has_pool and op.pool_window_y > 1 and self.has_conv):
            self.conv_op.recompute_conv_params(op.pool_window_x,op.pool_window_y)
        self.append(op)
        return True            

    def show(self):
        print("DBG: fused_ops collected: (begin_of_first_leg %d, end_of_first_leg %d)"%(self.begin_of_first_leg,self.end_of_first_leg))
        for i in self:
            print("    ", i.data["layer_type"],":",i.data["layer_name"], )

    # generate MatMul waveop and add it to waveop stream
    def gen_matmul_waveop(self, tpb, wave_id, psum_add):
        if (self.conv_op.item_sz == 2):
            in_dtype = "float16"
            out_dtype = "float32"
        elif (self.conv_op.item_sz == 4):
            in_dtype = "float32"
            out_dtype = "float32"
        else:            
            print("ERROR: item_sz %d not yet supported"%self.conv_op.item_sz)
        waveop_name = self.conv_op.data['layer_name']+"/MatMul_"+wave_id.id_string()           
        # find the weights offset within atom; -1 means don't load new weights
        weights_sb_address = tpb.statebuffer.circbuf_weights.get_sb_address(self.conv_op.weight_wave_lower_addr)
        if (weights_sb_address == self.prev_weight_wave_lower_addr):
            weights_sb_address = -1
            if (args.debug > 1): print("DBG: weights has been previously loaded; reusing them instead of reloading")
        else:            
            self.prev_weight_wave_lower_addr = weights_sb_address
        matmul_waveop = {
              'previous_waveops'        : [],   # to be added later
              'waveop_type'             : 'MatMul',
              'waveop_name'             : waveop_name,
              'layer_name'              : self.conv_op.data['layer_name'],
              'weights_sb_address'      : weights_sb_address,
              'ifmaps_sb_address'       : tpb.statebuffer.circbuf_ifmaps.get_sb_address(self.conv_op.ifmap_wave_lower_addr[0]),
              'in_dtype'                : in_dtype,
              'out_dtype'               : out_dtype,
              'wave_id_format'          : wave_id.format, # to be removed
              'wave_id'                 : wave_id.show(), # to be removed
              'start'                   : not(psum_add),    # to be removed
              'stride_x'                : self.conv_op.stride_x, # to be removed
              'stride_y'                : self.conv_op.stride_y, # to be removed
              'ifmap_count'             : self.conv_op.ifmap_count, # to be removed
              'ifmap_tile_width'        : self.conv_op.ofmap_wave_width, # to be removed 
              'ifmap_tile_height'       : self.conv_op.ofmap_wave_height, # to be removed
              'ofmap_count'             : self.conv_op.ofmap_count, # to be removed
              'ofmap_tile_width'        : self.conv_op.ofmap_wave_width, # to be removed
              'ofmap_tile_height'       : self.conv_op.ofmap_wave_height,  # to be removed
              'batching_in_wave'        : self.conv_op.Tn, # to be removed
              'start_tensor_calc'       : not(psum_add),
              'stop_tensor_calc'        : False,
              'fmap_x_step'             : self.conv_op.stride_x,
              'fmap_x_num'              : self.conv_op.ofmap_wave_width,
              'fmap_y_step'             : self.conv_op.W * self.conv_op.stride_y,
              'fmap_y_num'              : self.conv_op.ofmap_wave_height,
              'fmap_z_step'             : tpb.statebuffer.circbuf_ifmaps.atom_sz,
              'fmap_z_num'              : self.conv_op.Tn,
              'num_row_partitions'      : self.conv_op.ifmap_count,
              'psum_bank_id'            : self.conv_op.psum_bank_dst,
              'psum_bank_offset'        : self.conv_op.psum_bank_offset,
              'psum_x_step'             : 1,
              'psum_x_num'              : self.conv_op.ofmap_wave_width,
              'psum_y_step'             : self.conv_op.ofmap_cropped_tile_width,
              'psum_y_num'              : self.conv_op.ofmap_wave_height * self.conv_op.Tn,
              'num_column_partitions'   : self.conv_op.ofmap_count,
            }
        return matmul_waveop

    # generate Pool waveop and add it to waveop stream
    # TODO: currently, always go to SB after Pooling
    def gen_pool_waveop(self, tpb, tile_id, src_is_psum, src_psum_bank_id, start_at_mid_part):
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
            src_sb_address = tpb.statebuffer.circbuf_ifmaps.get_sb_address(self.pool_op.ifmap_wave_lower_addr[0])
            in_dtype = self.out_data_type
        psum_step_multiplier = 1   # kaena-174, tonga-310: after Inkling fix, no need for multiplier         
        waveop_name = self.pool_op.data['layer_name']+"/Pool_"+tile_id.id_string()
        pool_frequency = self.pool_op.pool_window_x * self.pool_op.pool_window_y
        pool_scale = float(1/pool_frequency)
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
              'src_psum_bank_offset'    : 0,
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
              'dst_sb_address'          : 0, # Need to adjust this after allocating atoms
              'dst_start_at_mid_part'   : start_at_mid_part,
              'dst_x_step'              : 1,
              'dst_x_num'               : self.pool_op.ofmap_cropped_tile_width,
              'dst_y_step'              : self.pool_op.E,
              'dst_y_num'               : self.pool_op.ofmap_cropped_tile_height,
              'dst_z_step'              : self.pool_op.E * self.pool_op.F,  # Need CNHW data format
              'dst_z_num'               : self.pool_op.Tn,  # Need CNHW data format
            }
        return pool_waveop

    # execute PEArray matrix multiply; returns True if successful (IFMAP wave is non-zero)
    def execute_matmul_waveop(self, tpb, wave_id, inputs, weights, psum_add):
        pearray_packed_weights = self.conv_op.pack_wave_conv_weights(weights, wave_id, tpb.statebuffer.circbuf_weights.replicate_multiple)
        pearray_packed_ifmaps = self.conv_op.pack_wave_ifmaps(inputs, wave_id, tpb.statebuffer.circbuf_weights.replicate_multiple, for_softmax=False)
        #print("\npearray_packed_ifmaps", wave_id.show(), "\n", pearray_packed_ifmaps)
        #print("\npearray_packed_weights", wave_id.show(), "\n", pearray_packed_weights)
        if (self.conv_op.ifmap_wave_lower_addr[0] < 0 or self.conv_op.ifmap_wave_upper_addr[0] < 0):
            print("WARNING layer %s: IFMAP wave (%s) has no data, so don't create waveops for this wave"%(op_list[0].data['layer_name'], wave_id.id_string()))
            return False
        else:
            dram_weights_waveops = tpb.statebuffer.circbuf_weights.read_data_region(
                                        wave_id, 
                                        self.conv_op.weight_wave_lower_addr, 
                                        self.conv_op.weight_wave_upper_addr,
                                        self.conv_op.ifmap_count,
                                        ifmaps_replicate=False)
            dram_ifmaps_waveops = []
            for i in range(self.conv_op.Tn):
                dram_ifmaps_waveops += tpb.statebuffer.circbuf_ifmaps.read_data_region(
                                            wave_id, 
                                            self.conv_op.ifmap_wave_lower_addr[i], 
                                            self.conv_op.ifmap_wave_upper_addr[i],
                                            self.conv_op.ifmap_count,
                                            self.conv_op.ifmap_count > self.conv_op.C)
            tpb.pearray.wave_fp16_mm(pearray_packed_ifmaps, pearray_packed_weights, self.conv_op.psum_bank_dst, psum_add)
            tpb.pearray.batching_in_wave = self.conv_op.Tn
            matmul_waveop = self.gen_matmul_waveop(tpb, wave_id, psum_add)
            tpb.waveop_stream.add_linked(matmul_waveop, dram_weights_waveops + dram_ifmaps_waveops)
            # mark this matmul as consumer of the 64B weights morsel
            matmul_waveop_name = matmul_waveop["waveop_name"]
            for i in dram_weights_waveops: # + dram_ifmaps_waveops:
                sb_addr = i["sb_address"]
                sb_length = i["length"]
                for j in range(sb_addr//64, (sb_addr+sb_length)//64):
                    tpb.statebuffer.consumer_of_64byte_morsel[j] = matmul_waveop_name
            # collect statistics
            if (args.debug > 1):
                tpb.pearray.total_pearray_wave_elems += self.conv_op.ofmap_wave_elems
                if (matmul_waveop["weights_sb_address"] < 0):
                    tpb.pearray.total_pearray_latency_cycles += self.conv_op.ofmap_wave_elems
                else:    
                    tpb.pearray.total_pearray_latency_cycles += max(self.conv_op.ofmap_count, self.conv_op.ofmap_wave_elems)
                tpb.pearray.num_of_ops_executed += self.conv_op.ofmap_count * self.conv_op.ofmap_wave_elems * self.conv_op.Tn * self.conv_op.ifmap_count
            return True
        
    # execute remaining fused ops
    def execute_tile_waveops (self, tpb, wave_id, tile_id, psum_bank_src, bias, psum_temp):
        op_list_iter = iter(range(1, len(self)))
        op_list = self
        for i in op_list_iter:
            layer_type = self[i].data['layer_type'] 
            if (re.search(r"Relu|Tanh|Sigmoid|Exp|Identity|Lrelu|Prelu", layer_type)):
                tpb.activate.wait_tile_done(tile_id)
                psum_temp = tpb.activate.act(op_list[i].data['layer_type'], psum_temp)
                psum_bank_dst = psum_bank_src
                dst_is_psum = False
                if (i != len(op_list)-1):
                    dst_is_psum = True
                    tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, psum_temp)
                tpb.gen_act_waveop_inline(None, op_list[i], self.conv_op, tile_id, 
                                          psum_bank_src, dst_is_psum, psum_bank_dst, [], 0)
                psum_bank_src = psum_bank_dst
            elif (layer_type == 'BiasAdd'):
                tpb.activate.wait_tile_done(tile_id)
                bias_chan_start = (tile_id.m_id//2) * PEArray.NUM_ROWS
                bias_chan_mid_part = (tile_id.m_id%2) == 1
                bias_chan_end = min(bias_chan_start + PEArray.NUM_ROWS, self.conv_op.M)
                bias_extracted = np.zeros(PEArray.NUM_ROWS)
                bias_extracted[0 : bias_chan_end - bias_chan_start] = bias[bias_chan_start : bias_chan_end]
                bias_addr = bias_chan_start * op_list[i].item_sz
                # TODO: use start_at_mid_part to effectively use all 128 partitions
                dram_bias_waveops = []
                if (tile_id.m_id%2 == 0):
                    dram_bias_waveops = tpb.statebuffer.circbuf_bias.read_data_region(wave_id, bias_addr, bias_addr, bias_chan_end - bias_chan_start)
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
                                              psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_addr)
                    tpb.statebuffer.circbuf_bias.free_data_region(bias_addr, bias_addr, tpb.waveop_stream.last_main_waveop)
                    psum_bank_src = psum_bank_dst
                    next(op_list_iter)
                else:                                    
                    if (i != len(op_list)-1):
                        dst_is_psum = True
                        tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, psum_temp)
                    tpb.gen_act_waveop_inline(op_list[i], None, self.conv_op, tile_id, 
                                              psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_addr)
                    tpb.statebuffer.circbuf_bias.free_data_region(bias_addr, bias_addr, tpb.waveop_stream.last_main_waveop)
                    psum_bank_src = psum_bank_dst
            elif (layer_type == 'ResAdd'):
                tpb.pool.wait_tile_done(tile_id)
                dram_resadd_waveops = []
                for z in range(op_list.conv_op.Tn):
                    dram_resadd_waveops += tpb.statebuffer.circbuf_residue.read_data_region(
                                                    wave_id,
                                                    self.conv_op.ofmap_tile_lower_addr[z], 
                                                    self.conv_op.ofmap_tile_upper_addr[z], 
                                                    self.conv_op.ofmap_count,
                                                    ifmaps_replicate = False,
                                                    start_at_mid_part = (wave_id.m_id%2)==1
                                                    )
                residue_ifmaps = np.zeros((self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, PEArray.NUM_COLS), dtype=np.float32)
                for z in range(op_list.conv_op.Tn):
                    for j in range(PEArray.NUM_COLS):
                        M_idx = tile_id.m_id * PEArray.NUM_COLS + j
                        if (M_idx >= self.conv_op.M):
                            break
                        else:
                            # NCHW
                            residue_tile_ifmap = np.zeros((self.conv_op.ofmap_full_tiley_sz, self.conv_op.ofmap_full_tilex_sz), dtype=np.float32)
                            residue_tile_ifmap[0:self.conv_op.ofmap_cropped_tile_height, 0:self.conv_op.ofmap_cropped_tile_width] = tpb.statebuffer.circbuf_residue.dram_data[
                                    tile_id.n_id * op_list.conv_op.Tn + z, 
                                    M_idx, 
                                    self.conv_op.ofmap_tile_y_start : self.conv_op.ofmap_tile_y_start + self.conv_op.ofmap_cropped_tile_height, 
                                    self.conv_op.ofmap_tile_x_start : self.conv_op.ofmap_tile_x_start + self.conv_op.ofmap_cropped_tile_width]
                            residue_ifmaps[z * self.conv_op.ofmap_full_tile_sz : (z+1) * self.conv_op.ofmap_full_tile_sz,j] = residue_tile_ifmap.flatten()
                #x1 = DBG_DUMP_PSUM_COL("PSUM col0 before ResAdd (FP32): ", psum_temp, 0)
                #x2 = DBG_DUMP_PSUM_COL("Residue col0 before ResAdd (FP32): ", residue_ifmaps, 0)
                psum_temp = tpb.pool.resadd(psum_temp, residue_ifmaps)
                #y1 = DBG_DUMP_PSUM_COL("PSUM col0 after RessAdd (FP32): ", psum_temp, 0)
                psum_bank_dst = psum_bank_src
                dst_is_psum = False
                if (i != len(op_list)-1):
                    dst_is_psum = True
                    tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz * self.conv_op.Tn, psum_temp)
                tpb.gen_resadd_waveop_inline(op_list[i], 
                        self.conv_op, 
                        tile_id, 
                        psum_bank_src, 
                        dst_is_psum, 
                        psum_bank_dst, 
                        dram_resadd_waveops, 
                        self.conv_op.ofmap_tile_lower_addr[0], 
                        (tile_id.m_id%2)==1)
                if ((tile_id.m_id%2)==1 or tile_id.m_id == tile_id.m-1):                
                    for i in range(op_list.conv_op.Tn):
                        tpb.statebuffer.circbuf_residue.free_data_region(self.conv_op.ofmap_tile_lower_addr[i], self.conv_op.ofmap_tile_upper_addr[i], tpb.waveop_stream.last_main_waveop)
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
        'Conv'   : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|ResAdd",
        'MatMul' : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|ResAdd",
        'BiasAdd': "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|ResAdd",
        'Add'    : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|ResAdd",
        'ResAdd' : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|ResAdd",
        'Relu'   : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|ResAdd",
        }

##################################################################################
# KGraph: nodes, edges, and operations
class KGraph:

    def __init__(self):
        # Node dictionary contains name -> Node pairs for quick reference
        self.node_dict = {}
        self.first_node = None
        self.last_node = None
        self.data_type = 'float16'
        self.item_sz = 2
        self.current_node = None
        self.last_split_next_nodes = []
        self.first_leg = False
        self.seen_begin_of_first_leg = False 

    # add forward edges for forward traversals        
    def add_forward_refs(self, starting_node):
        if (starting_node != None):
            #print (starting_node.data['layer_name'], len(starting_node.prev))
            if (len(starting_node.prev) > 0):
                for i in starting_node.prev:
                    i.add_next(starting_node)
                    self.add_forward_refs(i)

    # add a copy of layer, and change it to a new type
    def add_copy_with_new_type(self, layer, new_type):
        new_layer = copy.deepcopy(layer)
        new_layer['layer_type'] = new_type
        new_layer['layer_name'] = layer['layer_name'] + "_" + new_type
        new_layer['ref_file'] = layer['ref_file'].replace(".npy", "_" + new_type + ".npy")
        new_node = KNode(new_layer, self.item_sz, self.data_type)
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
        if (num_layers >= 1):
            for l in layers:
                new_node = KNode(l, self.item_sz, self.data_type)
                prev_layers = l['previous_layers']
                if (len(prev_layers) > 0):
                    for i in prev_layers:
                        if i in self.node_dict:
                            #if (args.debug>0): print("Previous layer for ", new_node.data['layer_name'], " is ", i)
                            new_node.add_prev(self.node_dict[i])
                        else:
                            print("ERROR: node %s isn't declared before %s"%(i, l['layer_name']))
                            exit(-1)
                else:
                    # Type "Input" node
                    if (l['layer_type'] == "Input" and self.first_node == None):
                        self.first_node = new_node
                # assume the last node is the last one processed (JSON graph is in order), at least for the last one
                self.last_node = new_node                
                self.node_dict[ l['layer_name'] ] = new_node
                # if softmax, expand to multiple subnodes
                if (l['layer_type'] == "Softmax"):
                    self.last_node.data['layer_type'] = "Exp"
                    self.add_copy_with_new_type(l, "Softmax2")
                    # move ref file attribute to the last operation for final comparisons
                    self.last_node.data['ref_file'] = new_node.data['ref_file']
                    new_node.data['ref_file'] = new_node.data['ref_file'].replace(".npy", "_Exp.npy")
            self.current_node = self.first_node
        else:
            print("ERROR: there are no layers!")
            exit(-1)

        # process waveops 
        if ("waveops" in kgraph_json):
            layers = kgraph_json["waveops"]
            num_layers = len(layers)
            if (num_layers >= 1):
                for l in layers:
                    new_node = KNode(l, self.item_sz, self.data_type)
                    prev_layers = l['previous_waveops']
                    if (len(prev_layers) > 0):
                        for i in prev_layers:
                            if i in self.node_dict:
                                #if (args.debug > 0): print("Previous waveop for ", new_node.data['waveop_name'], " is ", i)
                                new_node.add_prev(self.node_dict[i])
                            else:
                                print("ERROR: node %s isn't declared before %s"%(i, l['waveop_name']))
                                exit(-1)
                    else:
                        # the node with "Placeholder" type is input
                        self.first_node = new_node
                    # assume the last node is the last one processed (JSON graph is in order), at least for the last one
                    self.last_node = new_node                
                    self.node_dict[ l['waveop_name'] ] = new_node
                self.current_node = self.first_node
            else:
                print("ERROR: there are no layers!")
                exit(-1)

    # get next fused op            
    def get_next_fused_op(self, fused_ops):
        next_nodes = fused_ops[-1].next
        last_node_type = fused_ops[-1].data['layer_type']
        # if there's only one next node, check if it is fusable and add
        if (len(next_nodes) == 1):
            if (last_node_type in next_is_fusable
                    and not (next_nodes[0].data['layer_type'] == "ResAdd" and self.last_split_next_nodes != [])):
                regex = next_is_fusable[last_node_type]
                if (re.search(regex, next_nodes[0].data['layer_type'])):               
                    # TODO: don't fuse if pool size != stride size
                    if (fused_ops.add(next_nodes[0])):
                        fused_ops = self.get_next_fused_op(fused_ops)
        return fused_ops                    

    # starting from current node position, collect as many operations as possible            
    def get_fused_ops(self):
        fused_ops = FusedOp(self.data_type)
        if (self.current_node == None):
            print("ERROR: found zero operations to fuse")
            exit(-1)
        # when we see ResAdd, backtrack to the last split and follow the next leg in list
        if (self.current_node.data['layer_type'] == "ResAdd" and self.last_split_next_nodes != []):
            if (args.debug > 0): print("DBG: found ResAdd, back-track to last split and follow next leg")
            self.current_node = self.last_split_next_nodes[0] 
            self.last_split_next_nodes = self.last_split_next_nodes[1:]
            self.first_leg = False
        fused_ops.add(self.current_node)
        if (self.first_leg and not self.seen_begin_of_first_leg):
            fused_ops.begin_of_first_leg = True
            self.seen_begin_of_first_leg = True
        for i in self.current_node.next:
            print(i.data['layer_type'], ":", i.data['layer_name'])
        fused_ops = self.get_next_fused_op(fused_ops)
        # if there are multiple next nodes
        next_nodes = fused_ops[-1].next
        last_node_type = fused_ops[-1].data['layer_type']
        if (len(next_nodes) == 1):
            self.current_node = next_nodes[0]   
        elif (len(next_nodes) > 1):
            # Delete the leg that goes to ResAdd directly first, if it exists.
            # At the first fusedop, begin_of_first_leg=1, and the IFMAPs will be saved to residue and used by ResAdd
            if (len(next_nodes) > 2):
                print("ERROR: can only handle fork to 2 branches (ResNet50)")
                exit(-1)
            for i in range(len(next_nodes)):
                if (next_nodes[i].data['layer_type'] == "ResAdd"):
                    resadd_node = next_nodes[i]
                    del next_nodes[i]
                    #next_nodes.insert(0, resadd_node)
            # pick the first leg as current_node                        
            self.current_node = next_nodes[0]
            self.first_leg = True
            self.seen_begin_of_first_leg = False
            # save the remaining legs in a list
            self.last_split_next_nodes = next_nodes[1:]
        else:
            self.current_node = None
            self.last_split_next_nodes = []
        # if the last node is Conv or MatMul, add an identity pool op
        if (last_node_type == "Conv" or last_node_type == "MatMul"):
            fused_ops.add(self.gen_id_pool_op(fused_ops[-1]))
        # mark fusedops to be at end of first leg if the following op is ResAdd
        if (self.first_leg 
                and self.current_node != None 
                and self.current_node.data['layer_type'] == "ResAdd"):
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
        id_pool_op = KNode(id_pool_layer_data, self.item_sz, self.data_type)
        id_pool_op.prev.append(last_op)
        return id_pool_op

    def walk_ended(self):
        return self.current_node == None

##################################################################################
# The TPB scheduler has access to:
#   PEArray 
#   Pool 
#   BiasAddAct 
class TPBSched:
    def __init__(self):
        self.pearray = PEArray()
        self.pool = Pool()
        self.activate = BiasAddAct()
        self.statebuffer = StateBuffer()
        self.waveop_stream = WaveopStream()

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
              'dst_sb_address'          : 0, # Need to adjust this after allocating atoms
              'dst_start_at_mid_part'   : False,
              'dst_x_step'              : 1,
              'dst_x_num'               : 1,
              'dst_y_step'              : 1,
              'dst_y_num'               : 1,
              'dst_z_step'              : 1,
              'dst_z_num'               : 1,
              'num_partitions'          : 1
            }
        self.waveop_stream.add_linked(instr, [])

    # generate scaleadd instruction and add it to instruction stream
    def gen_scaleadd_waveop_inline(self, op, tile_id, src_is_psum, psum_bank_src, src_sb_address, dst_is_psum, psum_bank_dst, scale_val, add_val):
        layer_name = op.data["layer_name"]
        # TODO: update in_dtype when src_is_psum is added
        in_dtype = "float32"
        out_dtype = "float32"
        # TODO: refactor to some class to determine in_dtype and out_dtype
        if (op.item_sz == 2 and not src_is_psum):
            in_dtype = "float16"
        elif (op.item_sz == 1 and not src_is_psum):
            print("ERROR: item_sz %d not yet supported"%op.item_sz)
        if (op.item_sz == 2 and not dst_is_psum):
            out_dtype = "float16"
        elif (op.item_sz == 1 and not dst_is_psum):
            print("ERROR: item_sz %d not yet supported"%op.item_sz)
        dst_x_num = 1
        dst_y_step = 1
        dst_y_num = 1
        dst_z_num = 1
        dst_z_step = 1
        num_partitions = PEArray.NUM_COLS
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
              'dst_is_psum'             : dst_is_psum,
              'dst_psum_bank_id'        : psum_bank_dst,
              'dst_psum_bank_offset'    : 0,
              'dst_sb_address'          : 0, # Need to adjust this after allocating atoms
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
        self.waveop_stream.add_linked(instr, [])

    # generate activation instruction and add it to instruction stream
    def gen_act_waveop_inline(self, biasadd_op, act_op, conv_op, tile_id, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_start):
        layer_name = ""
        bias_add_en = False
        bias_sb_address = 0
        # TODO: update in_dtype when src_is_psum is added
        in_dtype = "float32"
        out_dtype = "float32"
        if (biasadd_op != None):
            bias_add_en = True
            bias_sb_address = self.statebuffer.circbuf_bias.get_sb_address(bias_start)
            layer_name = biasadd_op.data['layer_name']
            if (biasadd_op.item_sz == 2 and not dst_is_psum):
                out_dtype = "float16"
            elif (biasadd_op.item_sz == 1 and not dst_is_psum):
                print("ERROR: item_sz %d not yet supported"%biasadd_op.item_sz)
        act_type = "Identity"    
        if (act_op != None):
            act_type = act_op.data['layer_type']
            layer_name = act_op.data['layer_name']
            # TODO: refactor to some class to determine in_dtype and out_dtype
            if (act_op.item_sz == 2 and not dst_is_psum):
                out_dtype = "float16"
            elif (act_op.item_sz == 1 and not dst_is_psum):
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
                dst_z_step = dst_y_step * dst_y_num # Need CNHW data format
                dst_z_num = conv_op.Tn  # Need CNHW data format
            else:                
                dst_x_num = conv_op.ofmap_full_tilex_sz
                dst_y_step = conv_op.E
                dst_y_num = conv_op.ofmap_full_tiley_sz
                dst_z_step = dst_y_step * dst_y_num # Need CNHW data format
                dst_z_num = conv_op.Tn  # Need CNHW data format
            num_partitions = conv_op.ofmap_count
        else:
            print("ERROR: expecting a convolution/matmul before activation at %s!"%act_op.data['layer_name'])
            exit -1
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
              'bias_dtype'              : tpb.statebuffer.circbuf_bias.data_type, 
              'out_dtype'               : out_dtype,
              'src_psum_bank_id'        : psum_bank_src,
              'src_x_step'              : 1,
              'src_x_num'               : dst_x_num,
              'src_y_step'              : dst_y_step,
              'src_y_num'               : dst_y_num * dst_z_num,
              'dst_is_psum'             : dst_is_psum,
              'dst_psum_bank_id'        : psum_bank_dst,
              'dst_sb_address'          : 0, # Need to adjust this after allocating atoms
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
        self.waveop_stream.add_linked(instr, dram_bias_waveops)

    # generate ResAdd instruction and add it to instruction stream
    def gen_resadd_waveop_inline(self, op, conv_op, tile_id, psum_bank_src, dst_is_psum, psum_bank_dst, dram_resadd_waveops, data_start, start_at_mid_part):
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
                dst_z_step = dst_y_step * dst_y_num # Need CNHW data format if HW is large
                dst_z_num = conv_op.Tn  # Need CNHW data format if HW is large
            else:                
                dst_x_num = conv_op.ofmap_full_tilex_sz
                dst_y_step = conv_op.E
                dst_y_num = conv_op.ofmap_full_tiley_sz
                dst_z_step = dst_y_step * dst_y_num # Need CNHW data format
                dst_z_num = conv_op.Tn  # Need CNHW data format
            num_partitions = conv_op.ofmap_count
        else:
            print("ERROR: expecting a convolution/matmul before activation at %s!"%act_op.data['layer_name'])
            exit -1
        waveop_name = op.data['layer_name']+"/ResAdd_"+tile_id.id_string()
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : 'ResAdd',
              'waveop_name'             : waveop_name,
              'layer_name'              : op.data['layer_name'],
              'tile_id_format'          : tile_id.format,
              'tile_id'                 : tile_id.show(),
              'in_a_dtype'              : in_a_dtype,
              'in_b_dtype'              : in_b_dtype,
              'out_dtype'               : out_dtype,
              'src_a_is_psum'           : False,
              'src_a_psum_bank_id'      : 0,
              'src_a_psum_bank_offset'  : 0,
              'src_a_sb_address'        : self.statebuffer.circbuf_residue.get_sb_address(data_start),
              'src_a_start_at_mid_part' : start_at_mid_part,
              'src_a_x_step'            : 1,
              'src_a_x_num'             : dst_x_num,
              'src_a_y_step'            : dst_y_step,
              'src_a_y_num'             : dst_y_num,
              'src_a_z_step'            : dst_z_step,
              'src_a_z_num'             : dst_z_num,
              'src_b_is_psum'           : True,
              'src_b_psum_bank_id'      : psum_bank_src,
              'src_b_psum_bank_offset'  : 0,
              'src_b_sb_address'        : 0,
              'src_b_x_step'            : 1,
              'src_b_x_num'             : dst_x_num,
              'src_b_y_step'            : dst_y_step,
              'src_b_y_num'             : dst_y_num,
              'src_b_z_step'            : dst_z_step,
              'src_b_z_num'             : dst_z_num,
              'dst_is_psum'             : dst_is_psum,
              'dst_psum_bank_id'        : psum_bank_dst,
              'dst_psum_bank_offset'    : 0,
              'dst_sb_address'          : 0, # Need to adjust this after allocating atoms
              'dst_start_at_mid_part'   : start_at_mid_part,
              'dst_x_step'              : 1,
              'dst_x_num'               : dst_x_num,
              'dst_y_step'              : dst_y_step,
              'dst_y_num'               : dst_y_num,
              'dst_z_step'              : dst_z_step,
              'dst_z_num'               : dst_z_num,
              'num_partitions'          : num_partitions,
            }
        self.waveop_stream.add_linked(instr, dram_resadd_waveops)

    def gen_fused_pool_waveop_inline (self, fused_ops, tile_id, psum_bank_src, start_at_mid_part):
        pool_waveop = fused_ops.gen_pool_waveop(self, tile_id, True, psum_bank_src, start_at_mid_part)
        self.waveop_stream.add_linked(pool_waveop, [])

    def gen_unfused_pool_waveop_inline (self, fused_ops, tile_id, dram_waveops, start_at_mid_part):
        pool_waveop = fused_ops.gen_pool_waveop(self, tile_id, False, 0, start_at_mid_part)
        self.waveop_stream.add_linked(pool_waveop, dram_waveops)

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
            #stats.total_dram_latency_cycles = stats.num_of_reads_elems*self.statebuffer.circbuf_weights.item_sz / 10 # assuming 10GB/s BW available per TPB
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
            #print("STATS summary %s: %d %d %d %d %d %d %d %d %f %f %f %d %f"%(layer_name, self.pearray.num_wave_fp16_mm, num_of_wave_ops, self.pearray.num_of_ops_executed, wave_op_efficiency, num_of_weight_reads, num_of_reads_elems, num_of_writes_elems, num_of_weights_elem, total_weight_ifmaps_elems, actual_to_min_read_ratio, ideal_compute_to_load_ratio, tpb.pearray.total_pearray_latency_cycles, total_dram_latency_cycles))

    # Execute softmax (second part, which includes Sum, Reciprocate, Scale)
    def execute_softmax2(self, inputs, result_file):
        # create and save ones as weights file, then load them back
        ones_shape = [op_list[0].C, 1, 1, 1]
        ones_tensor = np.ones(ones_shape, dtype=self.statebuffer.circbuf_ifmaps.data_type)
        ones_file = op_list[0].data['ref_file'].replace(".npy", "-ones.npy")
        if (not args.inference):
            np.save(ones_file, ones_tensor)
        weights = []
        if (op_list.has_conv):
            op_list[0].data['kernel_file'] = ones_file
            op_list[0].data['kernel_format'] = "CRSM"
            op_list[0].data['kernel_shape'] = ones_shape
            weights = self.statebuffer.circbuf_weights.load_data(op_list.conv_op)
            weight_cols_per_wave = min(op_list.conv_op.M, PEArray.NUM_COLS)
            ifmap_cols_per_wave = min(op_list.conv_op.M, PEArray.NUM_COLS)

        # save result to create a scratch space (in DRAM), then use circular buffer load to populate params
        result = self.statebuffer.circbuf_scratch.load_data(op_list[-1], result_file)

        # reallocate statebuffer resources
        #self.statebuffer.reallocate_capacities()

        # initial psum bank is 0
        op_list.conv_op.set_psum_bank(0)
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
                        # free portion of requested data (but retain data such that we can still read it)
                        for z in range(op_list.conv_op.Tn):
                            self.statebuffer.circbuf_ifmaps.free_data_region(op_list.conv_op.ifmap_tile_lower_addr[z], op_list.conv_op.ifmap_tile_upper_addr[z], self.waveop_stream.last_main_waveop)
                        self.statebuffer.circbuf_weights.free_data_region(op_list.conv_op.weight_tile_lower_addr, op_list.conv_op.weight_tile_upper_addr, self.waveop_stream.last_main_waveop)
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
                                dram_output_waveops += self.statebuffer.circbuf_scratch.write_data_region(
                                                            tile_id, 
                                                            output_params_op.ofmap_tile_lower_addr[z], 
                                                            output_params_op.ofmap_tile_upper_addr[z], 
                                                            output_params_op.ifmap_count)
                        # The scale_add destination need to be adjusted after the above writes to data region
                        if (self.waveop_stream.last_main_waveop['waveop_type'] == "ScaleAdd"):
                            #self.waveop_stream.last_main_waveop['dst_sb_address'] = self.statebuffer.circbuf_scratch.start \
                            #                                                        + self.statebuffer.circbuf_scratch.current_atom_id*self.statebuffer.circbuf_scratch.atom_sz \
                            #                                                        + self.statebuffer.circbuf_scratch.get_atom_offset(output_params_op.ofmap_tile_lower_addr[0])
                            sb_addr = self.statebuffer.circbuf_scratch.get_sb_address(output_params_op.ofmap_tile_lower_addr[0])
                            if (sb_addr < 0):
                                if (len(dram_output_waveops) > 0):
                                    sb_addr = dram_output_waveops[0]['sb_address'] + self.statebuffer.circbuf_scratch.get_atom_offset(output_params_op.ofmap_tile_lower_addr[0])
                                else:
                                    print("ERROR execute_softmax2: addr %d not found in chunk2atom_map, and also not found in dram_output_waveops; giving up"%(output_params_op.ofmap_tile_lower_addr[0]))
                                    exit(-1)
                            self.waveop_stream.last_main_waveop['dst_sb_address'] = sb_addr
                        self.waveop_stream.add_outputs(dram_output_waveops)

                        # Advance to new bank (ping-pong between 0 and 1) for PEArray, while the old bank is being processed by other engines
                        op_list.conv_op.set_psum_bank((op_list.conv_op.get_psum_bank()+1)%4)
                        psum_add = False


        # save layer results to file, for retrieval by next layer                        
        np.save(result_file, result)
        if (args.golden_inputs):
            # if using golden inputs, save the ref_file instead of result_file
            self.statebuffer.saved_result_files[op_list[-1].data['layer_name']] = op_list[-1].data['ref_file']
        else:            
            self.statebuffer.saved_result_files[op_list[-1].data['layer_name']] = result_file

        # print circular buffer stats
        self.statebuffer.print_stats()
        self.collect_stats(op_list[-1].data['layer_name'])

        # reset scratch buffer for now (TODO: keep some atoms for next layer)
        #self.statebuffer.reset_all()
        self.statebuffer.keep_scratch_and_reset()
        return result

    # Execute an unfused pooling operator
    def execute_unfused_pool_op(self, inputs, result_file):
        # save result to create a scratch space (in DRAM), then use circular buffer load to populate params
        result = self.statebuffer.circbuf_scratch.load_data(op_list[-1], result_file)

        # recompute some parameters
        self.statebuffer.circbuf_ifmaps.recompute_ifmaps_params(op_list[-1]),

        # reallocate statebuffer resources
        #self.statebuffer.reallocate_capacities()

        # wave loop ordering scheme: nmhw
        pool_op = op_list[0]
        for n_id in range(pool_op.n):
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
                        #x = DBG_DUMP_PSUM_COL("PSUM before pool: ", psum_fake_extract, 0)
                        psum_temp = self.pool.pool(pool_op.data['layer_type'], psum_fake_extract, pool_op.stride_x, pool_op.pool_window_y, pool_op.Tn, input_tilex, input_tiley, output_tilex, output_tiley)
                        #x = DBG_DUMP_PSUM_COL("PSUM after pool: ", psum_temp, 0)
                        dram_ifmaps_waveops = []
                        for z in range(pool_op.Tn):
                            if (tile_id.m_id%2 == 0):
                                fmap_count = pool_op.ifmap_count
                                if (tile_id.m_id+1 != tile_id.m):
                                    fmap_count = 2*pool_op.ifmap_count
                                dram_ifmaps_waveops += tpb.statebuffer.circbuf_ifmaps.read_data_region(
                                                            wave_id, 
                                                            pool_op.ifmap_wave_lower_addr[z], 
                                                            pool_op.ifmap_wave_upper_addr[z],
                                                            fmap_count,
                                                            ifmaps_replicate=False, 
                                                            start_at_mid_part=False)
                        start_at_mid_part = tile_id.m_id%2 == 1
                        self.gen_unfused_pool_waveop_inline(op_list, tile_id, dram_ifmaps_waveops, start_at_mid_part)
                        for z in range(pool_op.Tn):
                            if (tile_id.m_id+1 == tile_id.m or tile_id.m_id%2 == 1):
                                tpb.statebuffer.circbuf_ifmaps.free_data_region(pool_op.ifmap_wave_lower_addr[z], pool_op.ifmap_wave_upper_addr[z], self.waveop_stream.last_main_waveop)
                            for j in range(PEArray.NUM_COLS):
                                M_idx = wave_id.m_id * PEArray.NUM_COLS + j
                                if (M_idx >= pool_op.M):
                                    break
                                else:
                                    # For now, multiply zeros, and at the ofmap, extract tile with zeros, then clip
                                    result_tile_tmp = (psum_temp[z * pool_op.ofmap_full_tile_sz : (z+1) * pool_op.ofmap_full_tile_sz, j])
                                    result_tile = result_tile_tmp.reshape((pool_op.ofmap_full_tiley_sz, pool_op.ofmap_full_tilex_sz))
                                    # NCHW
                                    result[n_id * pool_op.Tn + z, 
                                            M_idx, 
                                            pool_op.ofmap_tile_y_start : pool_op.ofmap_tile_y_start + pool_op.ofmap_cropped_tile_height, 
                                            pool_op.ofmap_tile_x_start : pool_op.ofmap_tile_x_start + pool_op.ofmap_cropped_tile_width]\
                                        = result_tile[0:pool_op.ofmap_cropped_tile_height, 0:pool_op.ofmap_cropped_tile_width]
                            # for scheduling, map resulting tile into portion of atom that is itself mapped to a portion in DRAM (file)
                            dram_output_waveops = self.statebuffer.circbuf_scratch.write_data_region(tile_id, pool_op.ofmap_tile_lower_addr[z], pool_op.ofmap_tile_upper_addr[z], pool_op.ofmap_count)
                        # The pooling destination need to be adjusted after the above writes to data region
                        if (self.waveop_stream.last_main_waveop['waveop_type'] == "Pool" 
                                or self.waveop_stream.last_main_waveop['waveop_type'] == "Activation"
                                or self.waveop_stream.last_main_waveop['waveop_type'] == "ResAdd"
                                ):
                            #self.waveop_stream.last_main_waveop['dst_sb_address'] = self.statebuffer.circbuf_scratch.start \
                            #                                                        + self.statebuffer.circbuf_scratch.current_atom_id*self.statebuffer.circbuf_scratch.atom_sz \
                            #                                                        + self.statebuffer.circbuf_scratch.get_atom_offset(pool_op.ofmap_tile_lower_addr[0])
                            sb_addr = self.statebuffer.circbuf_scratch.get_sb_address(pool_op.ofmap_tile_lower_addr[0])
                            if (sb_addr < 0):
                                if (len(dram_output_waveops) > 0):
                                    sb_addr = dram_output_waveops[0]['sb_address'] + self.statebuffer.circbuf_scratch.get_atom_offset(pool_op.ofmap_tile_lower_addr[0])
                                else:
                                    print("ERROR execute_softmax2: addr %d not found in chunk2atom_map, and also not found in dram_output_waveops; giving up"%(output_params_op.ofmap_tile_lower_addr[0]))
                                    exit(-1)
                            self.waveop_stream.last_main_waveop['dst_sb_address'] = sb_addr
                        self.waveop_stream.add_outputs(dram_output_waveops)

        # save layer results to file, for retrieval by next layer                        

        np.save(result_file, result)
        if (args.golden_inputs):
            # if using golden inputs, save the ref_file instead of result_file
            self.statebuffer.saved_result_files[pool_op.data['layer_name']] = pool_op.data['ref_file']
        else:            
            self.statebuffer.saved_result_files[pool_op.data['layer_name']] = result_file

        # print circular buffer stats
        self.statebuffer.print_stats()
        self.collect_stats(op_list[-1].data['layer_name'])

        # reset scratch buffer for now (TODO: keep some atoms for next layer)
        #self.statebuffer.reset_all()
        self.statebuffer.keep_scratch_and_reset()
        return result

    # Execute conv and other operations in list: for each op, load parameters and perform op with input
    def execute_conv_ops(self, inputs, result_file):
        # get weights from file
        weights = []
        if (op_list.has_conv):
            weights = self.statebuffer.circbuf_weights.load_data(op_list.conv_op)
            weight_cols_per_wave = min(op_list.conv_op.M, PEArray.NUM_COLS)
            ifmap_cols_per_wave = min(op_list.conv_op.M, PEArray.NUM_COLS)

        # load bias values
        bias = []
        if (op_list.has_biasadd):
            bias_temp = self.statebuffer.circbuf_bias.load_data(op_list.biasadd_op)
            bias = bias_temp.flatten()

        # save result to create a scratch space (in DRAM), then use circular buffer load to populate params
        result = self.statebuffer.circbuf_scratch.load_data(op_list[-1], result_file)

        # for ResAdd, retrieve the saved result file for one of the completed legs
        #if (op_list.has_resadd):
        #    self.statebuffer.circbuf_scratch.load_data(op_list.resadd_op)

        # reallocate statebuffer resources
        #self.statebuffer.reallocate_capacities()

        # initial psum bank is 0
        op_list.conv_op.set_psum_bank(0)
        # start tensor computation by clearing psum bank
        psum_add = False                               

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
                            r_id = 0
                            s_id = 0
                            while r_id < op_list.conv_op.R:
                                while s_id < op_list.conv_op.S:
                                    wave_id = WaveID(n_id, m_id, h_id, w_id, c_id, r_id, s_id)
                                    if (args.debug > 2): print (wave_id.show())
                                    # execute PEArray matrix multiply, and add to PSUM after first wave
                                    if (op_list.execute_matmul_waveop(self, wave_id, inputs, weights, psum_add)):
                                        psum_add = True
                                    s_id += self.statebuffer.circbuf_weights.replicate_multiple
                                r_id += s_id//op_list.conv_op.S
                                s_id = s_id%op_list.conv_op.S
                        # tile is done                                   
                        self.waveop_stream.last_main_waveop['stop_tensor_calc'] = True
                        self.pearray.trig_tile_done(tile_id)
                        # free portion of requested data (but retain data such that we can still read it)
                        for z in range(op_list.conv_op.Tn):
                            self.statebuffer.circbuf_ifmaps.free_data_region(op_list.conv_op.ifmap_tile_lower_addr[z], op_list.conv_op.ifmap_tile_upper_addr[z], self.waveop_stream.last_main_waveop)
                        self.statebuffer.circbuf_weights.free_data_region(op_list.conv_op.weight_tile_lower_addr, op_list.conv_op.weight_tile_upper_addr, self.waveop_stream.last_main_waveop)
                        # extract PSUM data
                        psum_bank_src = op_list.conv_op.get_psum_bank()
                        psum_temp = self.pearray.extract_psum(psum_bank_src, 0, op_list.conv_op.ofmap_full_tile_sz * op_list.conv_op.Tn)
                        #x = DBG_DUMP_PSUM_COL("PSUM after PEArray: ", psum_temp, 0)
                        # go through the remaining operations
                        psum_temp = op_list.execute_tile_waveops(tpb, wave_id, tile_id, psum_bank_src, bias, psum_temp)
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
                            dram_output_waveops += self.statebuffer.circbuf_scratch.write_data_region(
                                                        tile_id, 
                                                        output_params_op.ofmap_tile_lower_addr[z], 
                                                        output_params_op.ofmap_tile_upper_addr[z], 
                                                        output_params_op.ofmap_count)
                        # The pooling destination need to be adjusted after the above writes to data region
                        if (self.waveop_stream.last_main_waveop['waveop_type'] == "Pool"
                                or self.waveop_stream.last_main_waveop['waveop_type'] == "Activation"
                                or self.waveop_stream.last_main_waveop['waveop_type'] == "ResAdd"
                                ):
                            #self.waveop_stream.last_main_waveop['dst_sb_address'] = self.statebuffer.circbuf_scratch.start \
                            #                                                        + self.statebuffer.circbuf_scratch.current_atom_id*self.statebuffer.circbuf_scratch.atom_sz \
                            #                                                        + self.statebuffer.circbuf_scratch.get_atom_offset(output_params_op.ofmap_tile_lower_addr[0])
                            sb_addr = self.statebuffer.circbuf_scratch.get_sb_address(output_params_op.ofmap_tile_lower_addr[0])
                            if (sb_addr < 0):
                                if (len(dram_output_waveops) > 0):
                                    sb_addr = dram_output_waveops[0]['sb_address'] + self.statebuffer.circbuf_scratch.get_atom_offset(output_params_op.ofmap_tile_lower_addr[0])
                                else:
                                    print("ERROR execute_conv_op: addr %d not found in chunk2atom_map, and also not found in dram_output_waveops; giving up"%(output_params_op.ofmap_tile_lower_addr[0]))
                                    exit(-1)
                            self.waveop_stream.last_main_waveop['dst_sb_address'] = sb_addr
                        self.waveop_stream.add_outputs(dram_output_waveops)
                        if (m_id+1 == tile_id.m or m_id%2 == 1):
                            self.statebuffer.circbuf_scratch.free_data_region(
                                                        output_params_op.ofmap_tile_lower_addr[0], 
                                                        output_params_op.ofmap_tile_upper_addr[0], 
                                                        self.waveop_stream.last_main_waveop)
                        # Advance to new bank (ping-pong between 0 and 1) for PEArray, while the old bank is being processed by other engines
                        op_list.conv_op.set_psum_bank((op_list.conv_op.get_psum_bank()+1)%4)
                        psum_add = False

        # save layer results to file, for retrieval by next layer                       
        if (result.shape != tpb.statebuffer.circbuf_scratch.layer_shape):
            result = result.reshape(tpb.statebuffer.circbuf_scratch.layer_shape)
        np.save(result_file, result)
        if ((args.golden_inputs or args.inference)
                and os.path.isfile(op_list[-1].data['ref_file'])):
            # if using golden inputs, save the ref_file instead of result_file
            self.statebuffer.saved_result_files[op_list[-1].data['layer_name']] = op_list[-1].data['ref_file']
        else:            
            self.statebuffer.saved_result_files[op_list[-1].data['layer_name']] = result_file

        # print circular buffer stats
        self.statebuffer.print_stats()
        self.collect_stats(op_list[-1].data['layer_name'])
        #self.calculate_compute_to_load_ratio(op_list[-1].data['layer_name'])

        # reset scratch buffer for now (TODO: keep some atoms for next layer)
        #self.statebuffer.reset_all()
        self.statebuffer.keep_scratch_and_reset(op_list.begin_of_first_leg, op_list.end_of_first_leg)

        if (args.debug > 1): print("DBG: Total wave elements: ", op_list.conv_op.ofmap_wave_total_elems)

        return result                   

    # Execute conv and other operations in list: for each op, load parameters and perform op with input
    def execute_multiply(self, inputs, result_file):
        # save result to create a scratch space (in DRAM), then use circular buffer load to populate params
        result = self.statebuffer.circbuf_scratch.load_data(op_list[-1], result_file)
        tile_id = TileID(0,0,0,0,1,1,1,1)
        self.gen_scaleadd_waveop_inline(op_list[0], tile_id, 0, 0, 0 ,0 ,0 ,0 , 0)
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
    parser.add_argument("--debug", type=int, default=DEBUG_LEVEL_DEFAULT, help="Debug level")
    parser.add_argument("--golden_inputs", action='store_true', help="Use golden files as inputs for each layer")
    parser.add_argument("--save_layer_output", action='store_true', help="Save intermediate layer output into files")
    parser.add_argument("--inference", action='store_true', help="Inference mode: don't write intermediate -midout.npy and -ones.npy, except for the last -midout.npy")
    args = parser.parse_args()

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

    # go through all layers and add the fusable operations
    tpb = TPBSched()
    result_file = None
    num_mismatches = 0
    while (not kgraph.walk_ended()):
        op_list = kgraph.get_fused_ops()

        # get the result file for the fused operation
        last_op = op_list[-1]
        result_file = last_op.data['ref_file'].replace(".npy", "-midout.npy")
        print("Output file for layer %s is %s"%(last_op.data['layer_name'], result_file))

        # Check init op
        first_op = op_list[0]
        first_op_type = first_op.data['layer_type'] 
        if (first_op_type == "Input"):
            tpb.statebuffer.saved_result_files[first_op.data['layer_name']] = first_op.data['ref_file']
        elif (first_op_type == "Reshape"):
            for j in first_op.prev:
                if j.data['layer_name'] in tpb.statebuffer.saved_result_files:
                    tpb.statebuffer.saved_result_files[first_op.data['layer_name']] = tpb.statebuffer.saved_result_files[j.data['layer_name']]
                    break
        # Check conv fused op
        elif (first_op_type == "Conv" or first_op_type == "MatMul"):
            if (tpb.statebuffer.circbuf_ifmaps.dram_data_in_file != None):
                inputs = tpb.statebuffer.circbuf_ifmaps.dram_data
            else:                
                inputs = tpb.statebuffer.circbuf_ifmaps.load_data(first_op)
            results = tpb.execute_conv_ops(inputs, result_file)
        elif (first_op_type == "AvgPool" or first_op_type == "MaxPool"):
            if (tpb.statebuffer.circbuf_ifmaps.dram_data_in_file != None):
                inputs = tpb.statebuffer.circbuf_ifmaps.dram_data
            else:                
                inputs = tpb.statebuffer.circbuf_ifmaps.load_data(first_op)
            results = tpb.execute_unfused_pool_op(inputs, result_file)
        elif (first_op_type == "Softmax2"):
            if (tpb.statebuffer.circbuf_ifmaps.dram_data_in_file != None):
                inputs = tpb.statebuffer.circbuf_ifmaps.dram_data
            else:                
                inputs = tpb.statebuffer.circbuf_ifmaps.load_data(first_op)
            results = tpb.execute_softmax2(inputs, result_file)
        elif (first_op_type == "Multiply"):
            inputs = tpb.statebuffer.circbuf_ifmaps.load_data(first_op)
            results = tpb.execute_multiply(inputs, result_file)
        else:        
            print("ERROR: Unrecognized first operation %s"%first_op_type)
            exit(-1)

        # Check results against pre-computed results           
        if (first_op_type != "Input" and first_op_type != "Reshape"):
            if 'ref_file' in last_op.data and os.path.isfile(last_op.data['ref_file']):
                outputs = np.load(last_op.data['ref_file'])
                diff = results - outputs
                if (args.debug > 2): print("\nInput IFMAPS:\n", inputs)
                if (args.debug > 1): print("\nComputed OFMAPS:\n", results)
                if (args.debug > 1): print("\nExpected OFMAPS:\n", outputs)
                if (args.debug > 1): print("\nDiffed   OFMAPS:\n", diff)
                if (not npu.allclose(results, outputs, 1/100, 1e-5, verbose=True)):
                    print("\nERROR: layer %s computed OFMAPS is not equal to expected OFMAPS!\n"%(last_op.data['layer_name']))
                    num_mismatches += 1

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
    if (args.debug > 1):
        #print("STATS summary headings: num_wave num_waves_x_max_pe_ops num_of_ops_executed num_of_weight_reads num_of_reads_elems num_of_writes_elems total_weight_elems total_weight_ifmaps_elems actual_to_min_read_ratio ideal_compute_to_load_ratio wave_op_efficiency total_pearray_latency_cycles total_dram_latency_cycles")
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
    if (num_mismatches > 0):
        print("\nFAILED (num mismatches %d)"%num_mismatches)
    else:        
        print("\nPASSED")
