import json
import os
import math
import re
import numpy as np
import copy
import argparse
from skimage.util.shape import view_as_windows
from graphviz import Digraph

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

    def __init__(self, n_id, m_id, h_id, w_id):
        self.format = "nmhw"
        self.n_id, self.m_id, self.h_id, self.w_id = n_id, m_id, h_id, w_id

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
    num_ops_executed = 0

    def __init__(self):
        self.psum_buf = np.zeros((self.PSUM_NUM_BANKS, self.MAX_WAVE_SIZE, self.NUM_COLS), dtype=np.float32)

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

    def avg(self, in_array, stride, pool_window_size, Tn, ofmap_tilex_sz, ofmap_tiley_sz):
        num_cols = in_array.shape[1]
        # view_as_windows needs in_array to be in the same dimension as window_shape
        # need to make sure the third dimension of stride_shape to be '1' since that is the column direction
        tile_array = in_array.reshape(ofmap_tilex_sz,ofmap_tiley_sz,Tn,num_cols)
        window_shape = (pool_window_size,pool_window_size,Tn,num_cols)
        stride_shape = (stride, stride, Tn, 1)
        pool_result_temp = view_as_windows(tile_array, window_shape, stride_shape)
        pool_result = pool_result_temp.mean(axis=(4,5)).reshape(-1,num_cols)
        return pool_result

    def resadd(self, array_a, array_b):
        return array_a + array_b

    def max(self, in_array, stride, pool_window_size, Tn, ifmap_tilex_sz, ifmap_tiley_sz, ofmap_tilex_sz, ofmap_tiley_sz):
        num_cols = in_array.shape[1]
        # view_as_windows needs in_array to be in the same dimension as window_shape
        # need to make sure the third dimension of stride_shape to be '1' since that is the column direction
        #print("ifmap_tilex_sz ", ifmap_tilex_sz, " ifmap_tiley_sz ", ifmap_tiley_sz)
        input_tilex_with_pad = ofmap_tilex_sz * stride + pool_window_size - stride
        input_tiley_with_pad = ofmap_tiley_sz * stride + pool_window_size - stride
        input_tile_with_pad_sz = input_tilex_with_pad*input_tiley_with_pad
        tile_array = np.empty((input_tiley_with_pad, input_tilex_with_pad))
        tile_array[:] = -np.inf  # set all padding values to -inf to allow only actual tile values to be analyzed
        pool_result = np.zeros((ofmap_tilex_sz*ofmap_tiley_sz, num_cols))
        for i in range(num_cols):
            tile_array[0:ifmap_tiley_sz, 0:ifmap_tilex_sz] = in_array[0:ifmap_tilex_sz*ifmap_tiley_sz,i].reshape(ifmap_tiley_sz, ifmap_tilex_sz) # ignoring Tn for now
            window_shape = (pool_window_size, pool_window_size)
            stride_shape = (stride, stride)
            pool_result_temp = view_as_windows(tile_array, window_shape, stride_shape)
            pool_result[:,i] = pool_result_temp.max(axis=(2,3)).reshape(-1)
        return pool_result

##################################################################################
# Bias-Add and Activate properties and methods
class BiasAddAct:

    def wait_tile_done(self, tile_id):
        pass

    def biasadd(self, in_array, bias_array):
        return in_array + bias_array

    def act(self, type, in_array):
        if (type == 'Relu'):
            return self.relu(in_array)
        elif (type == 'Sigmoid'):
            return 1/(1 + math.exp(-in_array))

    def relu(self, in_array):
        return np.maximum(np.zeros(in_array.shape, dtype=in_array.dtype), in_array)

##################################################################################
# State buffer memory manager
class StateBuffer:

    SB_NUM_PARTITIONS = 128
    SB_PARTITION_SZ = 96*1024 # 96KB per partition
    SB_ATOM_SZ = 1024 # can be down to 256B for maximum DMA efficiency
    SB_NUM_1K_ATOMS = SB_PARTITION_SZ/SB_ATOM_SZ

    def __init__(self):
        #self.data = np.zeros((self.SB_NUM_PARTITIONS, self.SB_PARTITION_SZ))
        self.circbuf_ifmaps  = CircularBuffer(self, "ifmaps",  24,         self.SB_ATOM_SZ, 0)
        self.circbuf_weights = CircularBuffer(self, "weights", 96-16-4-24, self.SB_ATOM_SZ, 24)
        self.circbuf_bias    = CircularBuffer(self, "bias",    4,          self.SB_ATOM_SZ, 96-16-4)
        self.circbuf_scratch = CircularBuffer(self, "scratch", 16,         self.SB_ATOM_SZ, 96-16)
        self.saved_result_files = {}

    def print_stats(self):        
        self.circbuf_ifmaps.print_stats()
        self.circbuf_weights.print_stats()
        self.circbuf_bias.print_stats()
        self.circbuf_scratch.print_stats()

    def reset_all(self):        
        self.circbuf_ifmaps.reset()
        self.circbuf_weights.reset()
        self.circbuf_bias.reset()
        self.circbuf_scratch.reset()

##################################################################################
class CircularBuffer:
    def __init__(self, parent, circbuf_type, capacity, atom_sz, start):
        self.parent = parent
        self.capacity = capacity
        self.atom_sz = atom_sz
        self.start = start
        self.circbuf_type = circbuf_type
        self.reset()
        self.DRAM_elem_read = 0
        self.DRAM_elem_written = 0

    def reset(self):
        self.head_pointer = self.start
        self.tail_pointer = self.start
        self.current_atom_id = self.start
        self.atom_data_sz = self.atom_sz
        self.need_spare_atoms = 0
        self.need_skip_atoms = False
        self.count = 0
        self.eviction_count = 0
        self.max_count = 0
        self.allocated = np.zeros(self.capacity, dtype=bool)
        self.skipped = np.zeros(self.capacity, dtype=bool)
        self.dram_data_in_file = None
        self.dram_data_out_file = None
        self.dram_data = None
        self.dram_data_len = 0
        self.ifmap_data_len = 0
        self.ofmap_data_len = 0
        self.tracked_lower_addr = -1
        self.tracked_lower_addr_chunked = 0
        self.layer_name = ""
        self.layer_type = "Output"
        self.layer_format = ""
        self.layer_shape = []
        self.chunk2atom_map = {}
        self.chunk2spare_map = {}
        self.chunk2skip_map = {}
        self.item_sz = 2
        self.data_type = 'float16'

    def get_atom(self, addr):
        addr_chunked = self.get_chunk_addr(addr) 
        if (addr_chunked in self.chunk2atom_map):
            return self.chunk2atom_map[addr_chunked]
        else:
            print("ERROR %s: addr/atom_data_sz %d (addr %d) not found in chunk2atom_map of %s:"%(self.circbuf_type, addr_chunked, addr, self.layer_name))
            for i in self.chunk2atom_map.keys():
                print("     %s: %d"%(i, self.chunk2atom_map[i]))

    def get_atom_offset(self, addr):
        return addr % self.atom_data_sz

    def load_data(self, op, file_name = None):
        fmap_full_tiley_sz = 0
        fmap_full_tilex_sz = 0
        filter_x = 1
        stride_x = 1
        #self.reset()
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
                print("ERROR: Unrecognized operation %s"%op.data['layer_type'])
                exit(-1)
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
            self.layer_type = "Output" #op.data['layer_type']
            # Ensure atom data size is multiple of tile Y for pooling
            if (op_list.has_pool):
                fmap_full_tiley_sz = op_list.pool_op.ofmap_full_tiley_sz
                data_type = op_list.pool_op.data_type
            else:    
                fmap_full_tiley_sz = op_list.conv_op.ofmap_full_tiley_sz
                data_type = op_list.conv_op.data_type
            if (op.data['layer_type'] == "ResAdd"):
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
        self.dram_data_out_file = file_name
        return self.dram_data

    def load_file(self, file, fmap_full_tiley_sz = 0, fmap_full_tilex_sz = 0, filter_sz=1, stride_sz=1):
        self.dram_data_in_file = file
        self.dram_data = np.load(self.dram_data_in_file)
        self.item_sz = self.dram_data.dtype.itemsize   
        self.data_type = self.dram_data.dtype.name
        self.dram_data_len = self.dram_data.size * self.item_sz
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
            else:
                print("ERROR: wrong weights format %s"%self.layer_format)
                exit(-1)
            assert(C * R * S * M * self.item_sz == self.dram_data_len)                
            self.ifmap_data_len = self.dram_data_len//C
            m_data_len = M * self.item_sz
            sm_data_len = S * m_data_len
            # For NCHW, just use ifmap size as atom size (see rule above: "different FMAPs folds will be in different atoms")
            # Here ifmap is RSM
            if (self.ifmap_data_len <= self.atom_sz):
                self.atom_data_sz = self.ifmap_data_len
            # Else find the largest   
            elif (sm_data_len <= self.atom_sz):
                multiple = self.atom_sz // sm_data_len
                self.atom_data_sz = sm_data_len * min(R, multiple)
            elif (m_data_len <= self.atom_sz):
                multiple = self.atom_sz // m_data_len
                self.atom_data_sz = m_data_len * min(S, multiple)
            else:
                self.atom_data_sz = self.atom_sz
        else:
            if (self.layer_format == 'NCHW'):
                N, C, H, W = self.dram_data.shape
            elif (self.layer_format == 'CNHW'):    
                C, N, H, W = self.dram_data.shape
            else:
                print("ERROR in load_file: Unrecognized layer %s type %s format %s"%(self.layer_name, self.layer_type, self.layer_format))
                exit(-1)
            assert(N * C * H * W * self.item_sz == self.dram_data_len)                
            self.ifmap_data_len = self.dram_data_len//(N*C)
            # layer_shape is the ofmap_shape, in the format of N, M, E, F
            assert(self.layer_shape[0] == N)
            self.ofmap_data_len = self.layer_shape[2]*self.layer_shape[3]*self.item_sz
            ifmap_width_data_len = W * self.item_sz
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
                self.need_spare_atoms = max(1, (fmap_full_tiley_sz * fmap_full_tilex_sz * self.item_sz) // self.atom_sz)
                print("Reserve %d as spare atoms"%self.need_spare_atoms)
                if (C > 128):
                    self.need_skip_atoms = True
                    print("Reserving last atom for each wave load as skip atom")
            # make atom size multiple of width data length if it is smaller than default atom size
            elif (ifmap_width_data_len <= self.atom_sz):
                multiple = self.atom_sz // ifmap_width_data_len
                multiple = min(H, multiple)
                if (fmap_full_tiley_sz != 0):
                    if (fmap_full_tiley_sz < multiple):
                        multiple = (multiple//fmap_full_tiley_sz)*fmap_full_tiley_sz
                self.atom_data_sz = ifmap_width_data_len * min(H, multiple)
            else:
                self.atom_data_sz = self.atom_sz
        print("Loaded %s for layer %s, first data is %f, data size is %d bytes, atom size %d bytes, atom data size %d bytes"%(self.dram_data_in_file, self.layer_name, self.dram_data[0,0,0,0], self.item_sz, self.atom_sz, self.atom_data_sz)) 
        return self.dram_data

    def gen_dram_read_waveop(self, wave_id, atom_id, chunk_id, ifmap_count):
        offset_in_file = chunk_id*self.atom_data_sz
        length = self.atom_data_sz
        adjust_for_folding = wave_id.c_id * self.ifmap_data_len * PEArray.NUM_ROWS
        offset_in_fold = offset_in_file - adjust_for_folding
        # if address is larger than IFMAP size (H*W) for the case that IFMAP size is larger than Atom Data Size,
        # then try to get the modulo; but if the modulo is 0, then keep length = Atom Data Size
        if ((offset_in_fold + length) > self.ifmap_data_len and self.ifmap_data_len > self.atom_data_sz):
            length = self.ifmap_data_len % self.atom_data_sz

        if (length == 0): length = self.atom_data_sz
        assert (length > 0)            
        self.DRAM_elem_read += length * ifmap_count / self.item_sz
        #print("gen_dram_read_waveop - DRAM_elem_read: ", self.DRAM_elem_read, "length: ", length, "ifmap_count: ",ifmap_count)
        #print("fmap_data_len",self.ifmap_data_len, "atom_data_sz",self.atom_data_sz)
        #print("chunk_id", chunk_id, "offset", offset)
        if (args.golden_inputs):            
            simout_file = self.dram_data_in_file.replace("-midout.", ".")
        else:            
            simout_file = self.dram_data_in_file.replace("-midout.", "-simout.")
        waveop_name = self.layer_name+"/SBAtomFile_%s_%d_%s"%(self.circbuf_type, atom_id, wave_id.id_string())            
        return {
              'previous_waveops' : [],
              'waveop_type'      : "SBAtomFile",
              'waveop_name'      : waveop_name,
              'layer_name'       : self.layer_name,
              'atom_id'          : atom_id,
              'atom_size'        : self.atom_sz,
              'data_type'        : self.data_type,
              'ref_file'         : simout_file,
              'ref_file_format'  : self.layer_format,
              'ref_file_shape'   : self.layer_shape,
              'offset_in_file'   : offset_in_file,
              'length'           : length,
              'ifmaps_replicate' : False,
              'ifmaps_fold_idx'  : wave_id.c_id,
              'batch_fold_idx'   : wave_id.n_id,
              'ifmap_count'      : ifmap_count,
              'partition_step_bytes': self.ifmap_data_len,
            }

    def gen_dram_save_waveop(self, tile_id, atom_id, chunk_id, ofmap_count):
        offset_in_file = chunk_id*self.atom_data_sz
        length = self.atom_data_sz
        adjust_for_folding = tile_id.m_id * self.ofmap_data_len * PEArray.NUM_COLS
        offset_in_fold = offset_in_file - adjust_for_folding
        # if address is larger than IFMAP size (H*W) for the case that IFMAP size is larger than Atom Data Size,
        # then try to get the modulo; but if the modulo is 0, then keep length = Atom Data Size
        if ((offset_in_fold + length) > self.ofmap_data_len and self.ofmap_data_len > self.atom_data_sz):
            length = self.ofmap_data_len % self.atom_data_sz
            if (length == 0): length = self.atom_data_sz
        assert (length > 0)            
        self.DRAM_elem_written += length

        assert(self.dram_data_out_file != None)
        simout_file = self.dram_data_out_file.replace("-midout.", "-simout.")
        waveop_name = self.layer_name + "/SBAtomSave_%s_%d_%s"%(self.circbuf_type, atom_id, tile_id.id_string())
        return {
              'previous_waveops' : [],
              'waveop_type'      : "SBAtomSave",
              'waveop_name'      : waveop_name,
              'layer_name'       : self.layer_name,
              'atom_id'          : atom_id,
              'atom_size'        : self.atom_sz,
              'data_type'        : self.data_type,
              'ref_file'         : simout_file,
              'ref_file_format'  : self.layer_format,
              'ref_file_shape'   : self.layer_shape,
              'offset_in_file'   : offset_in_file,
              'length'           : length,
              'ofmaps_fold_idx'  : tile_id.m_id,
              'batch_fold_idx'   : tile_id.n_id,
              'ofmap_count'      : ofmap_count,
              'partition_step_bytes': self.ofmap_data_len,
            }

    def get_chunk_addr(self, addr):
        return addr // self.atom_data_sz

    def is_a_spare_atom(self, atom_id):
        return (self.need_spare_atoms > 0
                and atom_id >= self.start + self.capacity - self.need_spare_atoms
                and atom_id < self.start + self.capacity)

    def is_a_skip_atom(self, atom_id):
        return (self.need_skip_atoms and self.skipped[atom_id - self.start])
    
    def map_chunk_to_nonspare_atom(self, atom_id, chunk_id):
        for k in self.chunk2atom_map.keys():
            if (self.chunk2atom_map[k] == atom_id):
                if (args.debug > 2): print("%s: evicting %s at nonspare atom_id %d, replacing with chunk %d"%(self.circbuf_type, k, atom_id, chunk_id))
                self.eviction_count += 1
                del self.chunk2atom_map[k]
                break
        self.chunk2atom_map[chunk_id] = atom_id

    def map_chunk_to_spare_atom(self, atom_id, chunk_id):
        for k in self.chunk2spare_map.keys():
            if (self.chunk2spare_map[k] == atom_id):
                if (args.debug > 2): print("%s: evicting %s at spare atom_id %d, replacing with chunk %d"%(self.circbuf_type, k, atom_id, chunk_id))
                self.eviction_count += 1
                del self.chunk2spare_map[k]
                break
        self.chunk2spare_map[chunk_id] = atom_id

    def map_chunk_to_skip_atom(self, atom_id, chunk_id):
        for k in self.chunk2skip_map.keys():
            if (self.chunk2skip_map[k] == atom_id):
                if (args.debug > 2): print("%s: evicting %s at skip atom_id %d, replacing with chunk %d"%(self.circbuf_type, k, atom_id, chunk_id))
                self.eviction_count += 1
                del self.chunk2skip_map[k]
                break
        self.chunk2skip_map[chunk_id] = atom_id

    def read_data_region(self, wave_id, lower_addr, upper_addr, ifmap_count):
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
            while (self.is_a_spare_atom(self.tail_pointer) or self.is_a_skip_atom(self.tail_pointer)):
                self.tail_pointer += 1
                if (self.tail_pointer == self.start + self.capacity):
                    self.tail_pointer = self.start
                assert (self.tail_pointer != self.head_pointer)                    
            atom_id = self.allocate_atom()
            dram_waveops.append(self.gen_dram_read_waveop(wave_id, atom_id, lower_addr_chunked, ifmap_count))
            self.map_chunk_to_nonspare_atom(atom_id, lower_addr_chunked)
        else:
            if (args.debug > 2): print("%s: chunk %d is already mapped to atom %d"%(self.circbuf_type, lower_addr_chunked, self.chunk2atom_map[lower_addr_chunked]));
            atom_id = self.chunk2atom_map[lower_addr_chunked]
        # allocate the remaining chunks
        # check whether chunk is already in the spares
        starting_spares = False
        for i in range(lower_addr_chunked+1, upper_addr_chunked+1):
            if (self.need_skip_atoms 
                    and i in self.chunk2skip_map
                    and self.chunk2skip_map[i] == atom_id+1
                    and i == upper_addr_chunked
                    ):
                if (args.debug > 2): print("%s: reusing atom_id %d as skip for chunk %d (range %d-%d)"%(self.circbuf_type, self.chunk2skip_map[i], i, lower_addr, upper_addr))
                atom_id = self.chunk2skip_map[i]
            elif i in self.chunk2atom_map:
                atom_id = self.chunk2atom_map[i]
            else:
                if (self.is_a_spare_atom(self.tail_pointer)):
                    starting_spares = True
                    if (i not in self.chunk2spare_map):
                        atom_id = self.allocate_atom()
                        dram_waveops.append(self.gen_dram_read_waveop(wave_id, atom_id, i, ifmap_count))
                        self.allocated[atom_id - self.start] = False
                        self.count -= 1
                        self.map_chunk_to_spare_atom(atom_id, i)
                        if (args.debug > 2): print("%s: keeping atom_id %d as spare for chunk %d (range %d-%d)"%(self.circbuf_type, atom_id, i, lower_addr, upper_addr))
                    else:                        
                        if (args.debug > 2): print("%s: reusing atom_id %d as spare for chunk %d (range %d-%d)"%(self.circbuf_type, self.chunk2spare_map[i], i, lower_addr, upper_addr))
                else:
                    assert(starting_spares == False)    # indicate fragmented space (bad!)
                    # if multiple atoms used, and need skip atoms, then keep the last atom as skip-atom
                    if (self.need_skip_atoms and i == upper_addr_chunked):
                        atom_id = self.allocate_atom()
                        dram_waveops.append(self.gen_dram_read_waveop(wave_id, atom_id, i, ifmap_count))
                        if (args.debug > 2): print("%s: keeping last atom_id %d as skip for chunk %d (range %d-%d)"%(self.circbuf_type, atom_id, i, lower_addr, upper_addr))
                        self.allocated[atom_id - self.start] = False
                        self.skipped[atom_id - self.start] = True
                        self.count -= 1
                        self.map_chunk_to_skip_atom(atom_id, i)
                    else:                        
                        atom_id = self.allocate_atom()
                        dram_waveops.append(self.gen_dram_read_waveop(wave_id, atom_id, i, ifmap_count))
                        self.map_chunk_to_nonspare_atom(atom_id, i)
        return dram_waveops
    
    # hit_end_addr is used in write_data_region; so it should use ofmap_data_len
    def hit_end_addr(self, upper_addr):
        upper_addr_chunked = self.get_chunk_addr(upper_addr)
        # if upper addr is larger than IFMAP size, then it is in a different channel or batch item,
        # so use the modulo to check the end address
        upper_addr_mod = upper_addr % self.ofmap_data_len
        if ((upper_addr_mod == (self.ofmap_data_len - self.item_sz)) or (upper_addr_mod == (upper_addr_chunked+1)*self.atom_data_sz - self.item_sz)):
            return True
        return False

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
        if self.hit_end_addr(upper_addr):
            if (args.debug > 2): print("%s: freeing range %d to %d"%(self.circbuf_type, self.tracked_lower_addr, upper_addr))
            for i in range(self.tracked_lower_addr_chunked, upper_addr_chunked+1):
                atom_id = self.chunk2atom_map[i]
                dram_waveops.append(self.gen_dram_save_waveop(tile_id, atom_id, i, ofmap_count))
                self.free_atom(atom_id)
                self.tracked_lower_addr = -1
                del self.chunk2atom_map[i]
        return dram_waveops

    def free_data_region(self, lower_addr, upper_addr):
        if (args.debug > 2): print("%s: free byte range %d to %d"%(self.circbuf_type, lower_addr, upper_addr))
        lower_addr_chunked = self.get_chunk_addr(lower_addr)
        upper_addr_chunked = self.get_chunk_addr(upper_addr)
        for i in range(lower_addr_chunked, upper_addr_chunked+1):
            if i in self.chunk2atom_map:
                self.free_atom(self.chunk2atom_map[i])
                if (args.debug > 2): print("%s: freeing atom_id %d for chunk %d (lower_addr %d, upper_addr %d), but keep around for any subsequent read"%(self.circbuf_type, self.chunk2atom_map[i], i, lower_addr, upper_addr))
                # keep data around just in case, but allow pointers to wrap around
                #del self.chunk2atom_map[i]

    def allocate_atom(self):
        if (self.count == self.capacity):
            print ("ERROR %s: no more space during allocate_atom for layer %s!"%(self.circbuf_type, self.layer_name))
            self.print_stats()
            exit(-1)
            return -1
        self.current_atom_id = self.tail_pointer
        if (self.allocated[self.current_atom_id - self.start]):
            print("ERROR %s: allocating a still allocated (non-free) atom at atom_id %d"%(self.circbuf_type, self.current_atom_id))
            self.print_stats()
            exit(-1)
        self.allocated[self.current_atom_id - self.start] = True
        if (args.debug > 2): print ("%s: Added atom_id %d for layer %s"%(self.circbuf_type, self.current_atom_id, self.layer_name))
        self.tail_pointer += 1
        if (self.tail_pointer == self.start + self.capacity):
            self.tail_pointer = self.start
        self.count += 1
        if (self.count > self.max_count):
            self.max_count = self.count
        return self.current_atom_id            

    def free_atom(self, atom_id):   
        if (self.allocated[atom_id - self.start]):
            self.allocated[atom_id - self.start] = False
            self.count -= 1
            if (args.debug > 2): print ("%s: Freed atom_id %d for layer %s"%(self.circbuf_type, atom_id, self.layer_name))
        #else:
        #    print ("ERROR %s: cannot free atom ID %d since it is unallocated for layer %s!"%(self.circbuf_type, atom_id, self.layer_name))
        #    return -1
        # garbage collection: advance head pointer until it sees allocated atom
        if (not self.allocated[self.head_pointer - self.start]):
            self.head_pointer += 1            
            if (self.head_pointer == self.start + self.capacity):
                self.head_pointer = self.start

    def print_stats(self):
        print("STATS circular buffer type %s layer %s: capacity %d atom size %d atom data size %d atom count %d max count %d eviction count %d DRAM file data length %d IFMAP data length %d"%(self.circbuf_type, self.layer_name, self.capacity, self.atom_sz, self.atom_data_sz, self.count, self.max_count, self.eviction_count, self.dram_data_len, self.ifmap_data_len))

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
        assert (self.prev[0] != None)
        input_layer = self.prev[0].data
        self.ifmap_shape = input_layer['ofmap_shape']
        if (input_layer['ofmap_format'] == 'NCHW'):
            self.N, self.C, self.H, self.W = input_layer['ofmap_shape']
        elif (layer_info['ofmap_format'] == 'CNHW'):            
            self.C, self.N, self.H, self.W = input_layer['ofmap_shape']
        else:
            print("ERROR in populate_common_params: Unrecognized previous layer %s format %s"%(input_layer['layer_name'], input_layer['ofmap_format']))
            exit(-1)
        # get output shape from current layer's data
        layer_info = self.data
        self.ofmap_shape = layer_info['ofmap_shape']
        if (layer_info['ofmap_format'] == 'NCHW'):
            self.N, self.M, self.E, self.F = layer_info['ofmap_shape']
        elif (layer_info['ofmap_format'] == 'CNHW'):            
            self.M, self.N, self.E, self.F = layer_info['ofmap_shape']
        else:
            print("ERROR in populate_common_params: Unrecognized current layer %s format %s"%(layer_info['layer_name'], layer_info['ofmap_format']))
            exit(-1)
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
        self.ofmap_full_tilex_sz = min(self.F * self.Tn, PEArray.MAX_WAVE_SIZE)
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
            self.ofmap_full_tilex_sz = min(self.F * self.Tn, PEArray.MAX_WAVE_SIZE // self.ofmap_full_tiley_sz)
            #self.ofmap_full_tilex_sz = PEArray.MAX_WAVE_SIZE // self.ofmap_full_tiley_sz
        self.ofmap_full_tile_sz = self.ofmap_full_tilex_sz * self.ofmap_full_tiley_sz
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
        print("Common params for layer %s:  N=%d, M=%d, H=%d, W=%d, C=%d, E=%d, F=%d, stride_x=%d, stride_y=%d, ofmap_full_tilex_sz=%d, ofmap_full_tiley_sz=%d, ofmap_full_tile_sz=%d"
                %(self.data['layer_name'], self.N, self.M, self.H, self.W, self.C, self.E, self.F, self.stride_x, self.stride_y, self.ofmap_full_tilex_sz, self.ofmap_full_tiley_sz, self.ofmap_full_tile_sz))

    # Compute Conv looping params
    def populate_conv_params(self):
        # convolution kernel shape
        layer_info = self.data
        if (layer_info['kernel_format'] == 'CRSM'):
            self.C, self.R, self.S, self.M = layer_info['kernel_shape']
        elif (layer_info['kernel_format'] == 'CM'):
            self.C, self.M = layer_info['kernel_shape']
            self.R, self.S = 1, 1
        else:
            print("ERROR: wrong weights format %s"%layer_info['kernel_format'])
            exit(-1)
        self.populate_common_params(False)
        print("Conv params for layer %s: n=%d, m=%d, h=%d, w=%d, c=%d, R=%d, S=%d, Tn=%d"
                %(self.data['layer_name'], self.n, self.m, self.h, self.w, self.c, self.R, self.S, self.Tn))

    # Compute pooling params
    def populate_pooling_params(self):
        # are the dimensions from layer info correct?
        layer_info = self.data
        self.pool_window_y = layer_info['kernel_shape'][2]
        self.pool_window_x = layer_info['kernel_shape'][3]
        self.ifmap_wave_lower_addr = -1
        self.ifmap_wave_upper_addr = -1
        self.populate_common_params(True)
        print("Pooling params for layer %s: ofmap_full_tilex_sz=%d, ofmap_full_tiley_sz=%d, pool_window_x=%d, pool_window_y=%d"
                %(self.data['layer_name'], self.ofmap_full_tilex_sz, self.ofmap_full_tiley_sz, self.pool_window_x, self.pool_window_y))

    # Recompute conv tile params due to fused pooling
    def recompute_conv_params(self, pool_window_x, pool_window_y):        
        # For pooling using PSUM (fused), max tile size must be a multiple of pooling window
        self.ofmap_full_tiley_sz = (self.ofmap_full_tiley_sz // pool_window_y) * pool_window_y
        self.ofmap_full_tilex_sz = (self.ofmap_full_tilex_sz // pool_window_x) * pool_window_x
        self.ofmap_full_tile_sz = self.ofmap_full_tilex_sz * self.ofmap_full_tiley_sz
        print("Recomputed Conv params due to fused pooling: pool_window_x=%d, pool_window_y=%d, ofmap_full_tiley_sz=%d"
                %(pool_window_x, pool_window_y, self.ofmap_full_tiley_sz))

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
        self.ofmap_tile_lower_addr = int(np.ravel_multi_index(
                                            (tile_id.n_id * self.Tn, 
                                                tile_id.m_id * PEArray.NUM_COLS,
                                                self.ofmap_tile_y_start, 
                                                self.ofmap_tile_x_start),
                                        dims=self.ofmap_shape) * self.item_sz)
        # NCHW
        self.ofmap_tile_upper_addr = int(np.ravel_multi_index(
                                            (self.N - 1, 
                                                tile_id.m_id * PEArray.NUM_COLS,
                                                self.ofmap_tile_y_start + self.ofmap_cropped_tile_height - 1, 
                                                self.ofmap_tile_x_start + self.ofmap_cropped_tile_width - 1),
                                        dims=self.ofmap_shape) * self.item_sz)

        # compute the address bounds for IFMAP tile within IFMAPs tensor
        # TODO: for Tn>1, need to have multiple bounds for each batch item
        # NCHW
        ifmap_tile_lower_coordx = self.ofmap_tile_x_start * self.stride_x
        ifmap_tile_lower_coordy = self.ofmap_tile_y_start * self.stride_y
        self.ifmap_tile_lower_addr = int(np.ravel_multi_index(
                                            (tile_id.n_id * self.Tn, 
                                                0,
                                                ifmap_tile_lower_coordy,
                                                ifmap_tile_lower_coordx),
                                        dims=self.ifmap_shape) * self.item_sz)

        ifmap_tile_upper_coordx = ifmap_tile_lower_coordx + self.ofmap_cropped_tile_width * self.stride_x - 1
        ifmap_tile_upper_coordy = ifmap_tile_lower_coordy + self.ofmap_cropped_tile_height * self.stride_y - 1
        if (ifmap_tile_upper_coordx > self.W-1):
            ifmap_tile_upper_coordx = self.W-1
        if (ifmap_tile_upper_coordy > self.H-1):
            ifmap_tile_upper_coordy = self.H-1
        # NCHW
        self.ifmap_tile_upper_addr = int(np.ravel_multi_index(
                                            (self.N - 1,    # TODO: for Tn>1, need to have multiple bounds for each batch item
                                                (self.c-1) * PEArray.NUM_ROWS,
                                                ifmap_tile_upper_coordy,
                                                ifmap_tile_upper_coordx),
                                        dims=self.ifmap_shape) * self.item_sz)

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
    def pack_wave_ifmaps(self, ifmaps, wave_id, for_unfused_pooling):
        # If we are not doing convolution (aka pooling), set out_array_dim_y to be PEArray.NUM_COLS to match pooling/activation engines dimension
        if (for_unfused_pooling):
            out_array_dim_y = PEArray.NUM_COLS
            fmap_folding_idx = wave_id.m_id
            fmap_total_count = self.M
        else:            
            out_array_dim_y = PEArray.NUM_ROWS
            fmap_folding_idx = wave_id.c_id
            fmap_total_count = self.C
        out_array = np.zeros((PEArray.MAX_WAVE_SIZE, out_array_dim_y))
        # remember to extract IFMAPs starting at r_id, s_id (which should be zero for non-conv op)
        # also need to add zeros for padding
        self.ifmap_wave_lower_addr = -1
        self.ifmap_wave_upper_addr = -1
        self.ofmap_wave_lower_coordx = 0
        self.ofmap_wave_lower_coordy = 0
        self.ofmap_wave_upper_coordx = 0
        self.ofmap_wave_upper_coordy = 0
        self.psum_bank_offset = 0
        # for pooling, the "row" below actually means output columns
        pe_row_start = fmap_folding_idx * out_array_dim_y
        pe_row_stop = min(fmap_total_count, pe_row_start + out_array_dim_y)
        assert(pe_row_start < pe_row_stop)
        for row in range(pe_row_start, pe_row_stop):
            #out_array[:,row] = self.pack_wave_ifmap(ifmaps[:, wave_id.c_id * out_array_dim_y + row], wave_id)
            ifmap = ifmaps[:, row]  # NCHW
            pe_row_offset = row - pe_row_start
            ofmap_full_tilex_sz_per_batchitem = self.ofmap_full_tilex_sz//self.Tn
            for i in range(self.Tn):
                for x in range(ofmap_full_tilex_sz_per_batchitem):
                    for y in range(self.ofmap_full_tiley_sz):
                        ifmap_tilex = (wave_id.w_id * ofmap_full_tilex_sz_per_batchitem + x) * self.stride_x + wave_id.s_id - self.pad_west
                        ifmap_tiley = (wave_id.h_id * self.ofmap_full_tiley_sz + y) * self.stride_y + wave_id.r_id - self.pad_north
                        ifmap_addr = i * self.ofmap_full_tile_sz//self.Tn + y * ofmap_full_tilex_sz_per_batchitem + x
                        if (ifmap_tilex < 0 or ifmap_tilex >= self.W):
                            out_array[ifmap_addr, pe_row_offset] = 0
                        elif (ifmap_tiley < 0 or ifmap_tiley >= self.H):
                            out_array[ifmap_addr, pe_row_offset] = 0
                        else:
                            out_array[ifmap_addr, pe_row_offset] = ifmap[(wave_id.n_id * self.Tn) + i, ifmap_tiley, ifmap_tilex]
                            # Check bounds of actual pixels within the original ifmaps for the first ifmap (which should reside in first SB partition)
                            # TODO: check how N/C are arrange in memory; batching within waves may cause different atoms to be accessed by same wave
                            # TODO: for Tn>1, need to have multiple bounds for each batch item
                            if (row == pe_row_start):                                
                                # NCHW
                                self.ifmap_wave_upper_addr = int(np.ravel_multi_index(((wave_id.n_id * self.Tn) + i, row, ifmap_tiley, ifmap_tilex),
                                                                    dims=ifmaps.shape) * ifmaps.dtype.itemsize)
                                self.ofmap_wave_upper_coordx = x
                                self.ofmap_wave_upper_coordy = y
                                if (self.ifmap_wave_lower_addr < 0):
                                    self.ifmap_wave_lower_addr = self.ifmap_wave_upper_addr
                                    self.ofmap_wave_lower_coordx = x
                                    self.ofmap_wave_lower_coordy = y
                                    self.psum_bank_offset = (y * ofmap_full_tilex_sz_per_batchitem + x)
                        #print("x %d y %d ifmap_tilex %d ifmap_tiley %d wave_lower_coordx %d wave_upper_coordy %d wave_upper_coordx %d wave_upper_coordy %d"%(x, y, ifmap_tilex, ifmap_tiley, self.ofmap_wave_lower_coordx, self.ofmap_wave_lower_coordy, self.ofmap_wave_upper_coordx, self.ofmap_wave_upper_coordy))                                    
        return out_array

    def pack_wave_ifmaps_unfused_pooling (self, ifmaps, wave_id, for_unfused_pooling):
        # If we are not doing convolution (aka pooling), set out_array_dim_y to be PEArray.NUM_COLS to match pooling/activation engines dimension
        if (for_unfused_pooling):
            out_array_dim_y = PEArray.NUM_COLS
            fmap_folding_idx = wave_id.m_id
            fmap_total_count = self.M
            out_array = np.zeros((PEArray.MAX_WAVE_SIZE * self.stride_x * self.stride_y * 2, out_array_dim_y))
        else:            
            out_array_dim_y = PEArray.NUM_ROWS
            fmap_folding_idx = wave_id.c_id
            fmap_total_count = self.C
            out_array = np.zeros((PEArray.MAX_WAVE_SIZE, out_array_dim_y))
        # remember to extract IFMAPs starting at r_id, s_id (which should be zero for non-conv op)
        # also need to add zeros for padding
        self.ifmap_wave_lower_addr = -1
        self.ifmap_wave_upper_addr = -1
        # TODO: make the coordinates definition more clear
        self.ifmap_wave_lower_coordx = 0
        self.ifmap_wave_lower_coordy = 0
        self.ifmap_wave_upper_coordx = 0
        self.ifmap_wave_upper_coordy = 0
        self.psum_bank_offset = 0
        # for pooling, the "row" below actually means output columns
        pe_row_start = fmap_folding_idx * out_array_dim_y
        pe_row_stop = min(fmap_total_count, pe_row_start + out_array_dim_y)
        self.ifmap_count = pe_row_stop - pe_row_start
        assert(pe_row_start < pe_row_stop)
        for row in range(pe_row_start, pe_row_stop):
            #out_array[:,row] = self.pack_wave_ifmap(ifmaps[:, wave_id.c_id * out_array_dim_y + row], wave_id)
            ifmap = ifmaps[:, row]  # NCHW
            pe_row_offset = row - pe_row_start
            ofmap_full_tilex_sz_per_batchitem = self.ofmap_full_tilex_sz//self.Tn
            for i in range(self.Tn): #ignore Tn for now
                self.ifmap_wave_lower_coordx = (wave_id.w_id * ofmap_full_tilex_sz_per_batchitem) * self.stride_x 
                self.ifmap_wave_lower_coordy = (wave_id.h_id * self.ofmap_full_tiley_sz) * self.stride_y
                self.ifmap_wave_upper_coordx = ((wave_id.w_id+1) * ofmap_full_tilex_sz_per_batchitem) * self.stride_x + (self.pool_window_x - self.stride_x) - 1
                self.ifmap_wave_upper_coordy = ((wave_id.h_id+1) * self.ofmap_full_tiley_sz) * self.stride_y + (self.pool_window_y - self.stride_y) - 1 
                if (self.ifmap_wave_upper_coordx > self.W-1):
                    self.ifmap_wave_upper_coordx = self.W-1
                if (self.ifmap_wave_upper_coordy > self.H-1):
                    self.ifmap_wave_upper_coordy = self.H-1
                row_temp = ifmap[(wave_id.n_id * self.Tn) + i, 
                                                            self.ifmap_wave_lower_coordy:self.ifmap_wave_upper_coordy+1,
                                                            self.ifmap_wave_lower_coordx:self.ifmap_wave_upper_coordx+1].flatten()
                out_array[0:len(row_temp), pe_row_offset] = row_temp
                if (row == pe_row_start):                               
                    # NCHW
                    self.ifmap_wave_lower_addr = int(np.ravel_multi_index(((wave_id.n_id * self.Tn) + i, row, self.ifmap_wave_lower_coordy, self.ifmap_wave_lower_coordx),
                                                        dims=ifmaps.shape) * ifmaps.dtype.itemsize)
                    self.ifmap_wave_upper_addr = int(np.ravel_multi_index(((wave_id.n_id * self.Tn) + i, row, self.ifmap_wave_upper_coordy, self.ifmap_wave_upper_coordx),
                                                        dims=ifmaps.shape) * ifmaps.dtype.itemsize)
        #print(self.ifmap_wave_lower_coordx, self.ifmap_wave_lower_coordy, self.ifmap_wave_upper_coordx, self.ifmap_wave_upper_coordy)                    
        return out_array

    # Pack the conv weights in columns to create a PE-Array weights array for a particular wave number
    #   weights: conv weights in CRSM format
    #   wave_id: current wave ID, [n_id, m_id, h_id, w_id, c_id, r_id, s_id]
    #   return: a 128x64 array
    def pack_wave_conv_weights(self, weights, wave_id):
        out_array = np.zeros((PEArray.NUM_ROWS, PEArray.NUM_COLS))
        pe_row_start = wave_id.c_id * PEArray.NUM_ROWS
        pe_row_stop = min(self.C, pe_row_start + PEArray.NUM_ROWS)
        pe_col_start = wave_id.m_id * PEArray.NUM_COLS
        pe_col_stop = min(self.M, pe_col_start + PEArray.NUM_COLS)
        self.ifmap_count = pe_row_stop - pe_row_start
        self.ofmap_count = pe_col_stop - pe_col_start
        for row in range(pe_row_start, pe_row_stop):
            for col in range(pe_col_start, pe_col_stop):
                out_array[row - pe_row_start, col - pe_col_start] = weights[row, wave_id.r_id, wave_id.s_id, col] # CRSM
        self.weight_wave_lower_addr = int(np.ravel_multi_index(
                                            (pe_row_start, wave_id.r_id, wave_id.s_id, pe_col_start), # CRSM
                                            dims=weights.shape) 
                                            * weights.dtype.itemsize)
        self.weight_wave_upper_addr = int(np.ravel_multi_index(
                                            (pe_row_start, wave_id.r_id, wave_id.s_id, pe_col_stop-1), # CRSM
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
            input_list.append(i['waveop_name'])
            self.append_check(i)
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
        self.has_matmul = False
        self.has_biasadd = False
        self.pool_op = None
        self.resadd_op = None
        self.conv_op = None
        self.matmul_op = None
        self.biasadd_op = None
        self.out_data_type = out_data_type 

    # Add operation to list of fused operations.
    # Returns True if successful; False if cannot add (i.e. Pool cannot be fused)
    def add(self, op):
        if (args.debug > 2):
            print("DBG: adding layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
        if (op.data['layer_type'] == 'AvgPool' or op.data['layer_type'] == 'MaxPool'):
            op.populate_pooling_params()
            # If not first op, pool cannot be fused with previous op if stride != pooling window
            if (len(self) != 0 and 
                    (op.stride_x != op.pool_window_x or op.stride_y != op.pool_window_y)):
                if (args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            elif (self.has_pool):
                if (args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            else:
                # recompute Conv params due to constrained Pooling tile dimensions
                # (only if it is not identity pool, where window/stride are both 1)
                if (op.pool_window_y > 1 and self.has_conv):
                    self.conv_op.recompute_conv_params(op.pool_window_x,op.pool_window_y)
                self.pool_op = op
                self.has_pool = True
        elif (op.data['layer_type'] == 'Conv' or op.data['layer_type'] == 'MatMul'):
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
        elif (op.data['layer_type'] == 'MatMul'):
            if (self.has_matmul):
                if (args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            else:
                self.matmul_op = op
                self.has_matmul = True
        elif (op.data['layer_type'] == 'BiasAdd'):
            if (self.has_biasadd):
                if (args.debug > 2):
                    print("DBG: refusing to add layer_type ", op.data["layer_type"], " layer_name ", op.data["layer_name"])
                return False
            else:
                self.biasadd_op = op
                self.has_biasadd = True
        self.append(op)
        return True            

    def show(self):
        print("DBG: fused_ops collected: ",)
        for i in self:
            print("    ", i.data["layer_type"],":",i.data["layer_name"], )

    # generate MatMul waveop and add it to waveop stream
    def gen_matmul_waveop(self, tpb, wave_id, psum_add):
        ofmap_wave_width = self.conv_op.ofmap_wave_upper_coordx - self.conv_op.ofmap_wave_lower_coordx + 1
        ofmap_wave_height = self.conv_op.ofmap_wave_upper_coordy - self.conv_op.ofmap_wave_lower_coordy + 1
        self.conv_op.ofmap_wave_total_elems += ofmap_wave_width*ofmap_wave_height
        if (self.conv_op.item_sz == 2):
            in_dtype = "float16"
            out_dtype = "float32"
        elif (self.conv_op.item_sz == 4):
            in_dtype = "float32"
            out_dtype = "float32"
        else:            
            print("ERROR: item_sz %d not yet supported"%self.conv_op.item_sz)
        waveop_name = self.conv_op.data['layer_name']+"/MatMul_"+wave_id.id_string()            
        matmul_waveop = {
              'previous_waveops'        : [],   # to be added later
              'waveop_type'             : 'MatMul',
              'waveop_name'             : waveop_name,
              'layer_name'              : self.conv_op.data['layer_name'],
              'weights_atom_id'         : tpb.statebuffer.circbuf_weights.get_atom(self.conv_op.weight_wave_lower_addr),
              'weights_offset_in_atom'  : tpb.statebuffer.circbuf_weights.get_atom_offset(self.conv_op.weight_wave_lower_addr),  # TODO: -1 means don't load new weights
              'ifmaps_atom_id'          : tpb.statebuffer.circbuf_ifmaps.get_atom(self.conv_op.ifmap_wave_lower_addr), # if multiple atoms loaded, pick the first one
              'ifmaps_offset_in_atom'   : tpb.statebuffer.circbuf_ifmaps.get_atom_offset(self.conv_op.ifmap_wave_lower_addr),
              'ifmaps_atom_size'        : tpb.statebuffer.circbuf_ifmaps.atom_sz,
              'wave_id_format'          : wave_id.format,
              'wave_id'                 : wave_id.show(),
              'in_dtype'                : in_dtype,
              'out_dtype'               : out_dtype,
              'start'                   : not(psum_add),
              'stride_x'                : self.conv_op.stride_x,
              'stride_y'                : self.conv_op.stride_y,
              'psum_bank_id'            : self.conv_op.psum_bank_dst,
              'psum_bank_offset'        : self.conv_op.psum_bank_offset,
              'ifmap_count'             : self.conv_op.ifmap_count, # to be removed
              'ifmap_tile_width'        : ofmap_wave_width, # to be removed 
              'ifmap_tile_height'       : ofmap_wave_height, # to be removed
              'ofmap_count'             : self.conv_op.ofmap_count, # to be removed
              'ofmap_tile_width'        : ofmap_wave_width, # to be removed
              'ofmap_tile_height'       : ofmap_wave_height,  # to be removed
              'batching_in_wave'        : self.conv_op.Tn,
              'start_tensor_calc'       : not(psum_add),
              'stop_tensor_calc'        : False,
              'fmap_x_step'             : self.conv_op.stride_x,
              'fmap_x_num'              : ofmap_wave_width,
              'fmap_y_step'             : self.conv_op.W * self.conv_op.stride_y,
              'fmap_y_num'              : ofmap_wave_height,
              'fmap_z_step_atoms'       : 1,    # 1 for now; may need more if input needs more than one atom at once 
              'fmap_z_num'              : self.conv_op.Tn,  # need CNHW format for this
              'num_row_partitions'      : self.conv_op.ifmap_count,
              'psum_x_step'             : 1,
              'psum_x_num'              : ofmap_wave_width,
              'psum_y_step'             : self.conv_op.ofmap_cropped_tile_width,
              'psum_y_num'              : ofmap_wave_height,
              'num_column_partitions'   : self.conv_op.ofmap_count,
            }
        return matmul_waveop

    # generate Pool waveop and add it to waveop stream
    # TODO: currently, always go to SB after Pooling
    def gen_pool_waveop(self, tpb, tile_id, src_is_psum, src_psum_bank_id):
        if (src_is_psum):
            src_ifmap_width = self.pool_op.ifmap_cropped_tile_width
            src_ifmap_height = self.pool_op.ifmap_cropped_tile_height
            src_sb_atom_id = 0
            src_sb_offset_in_atom = 0
            if (self.pool_op.item_sz == 2):
                in_dtype = "float32"
            else:    
                in_dtype = "float32"
        else:
            src_ifmap_width = self.pool_op.W
            src_ifmap_height = self.pool_op.H
            src_sb_atom_id = tpb.statebuffer.circbuf_ifmaps.get_atom(self.pool_op.ifmap_wave_lower_addr)
            src_sb_offset_in_atom = tpb.statebuffer.circbuf_ifmaps.get_atom_offset(self.pool_op.ifmap_wave_lower_addr)
            in_dtype = self.out_data_type
        psum_step_multiplier = 1   # kaena-174, tonga-310: after Inkling fix, no need for multiplier         
        waveop_name = self.pool_op.data['layer_name']+"/Pool_"+tile_id.id_string()
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
              'src_sb_atom_id'          : src_sb_atom_id, 
              'src_sb_offset_in_atom'   : src_sb_offset_in_atom,
              'src_x_step'              : 1 * psum_step_multiplier,
              'src_x_num'               : self.pool_op.pool_window_x,
              'src_y_step'              : src_ifmap_width * psum_step_multiplier,
              'src_y_num'               : self.pool_op.pool_window_y,
              'src_z_step'              : self.pool_op.stride_x * psum_step_multiplier,
              'src_z_num'               : self.pool_op.ofmap_cropped_tile_width,
              'src_w_step'              : src_ifmap_width * self.pool_op.stride_y * psum_step_multiplier,
              'src_w_num'               : self.pool_op.ofmap_cropped_tile_height,
              'pool_frequency'          : self.pool_op.pool_window_x * self.pool_op.pool_window_y,
              'num_partitions'          : self.pool_op.ofmap_count,
              'dst_sb_atom_id'          : 0, # Need to adjust this after allocating atoms
              'dst_sb_offset_in_atom'   : tpb.statebuffer.circbuf_scratch.get_atom_offset(self.pool_op.ofmap_tile_lower_addr),
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
        pearray_packed_weights = self.conv_op.pack_wave_conv_weights(weights, wave_id)
        pearray_packed_ifmaps = self.conv_op.pack_wave_ifmaps(inputs, wave_id, for_unfused_pooling=False)
        #print("\npearray_packed_ifmaps", wave_id.show(), "\n", pearray_packed_ifmaps)
        #print("\npearray_packed_weights", wave_id.show(), "\n", pearray_packed_weights)
        if (self.conv_op.ifmap_wave_lower_addr < 0 or self.conv_op.ifmap_wave_upper_addr < 0):
            print("WARNING layer %s: IFMAP wave (%s) has no data, so don't create waveops for this wave"%(op_list[0].data['layer_name'], wave_id.id_string()))
            return False
        else:
            dram_weights_waveops = tpb.statebuffer.circbuf_weights.read_data_region(
                                        wave_id, 
                                        self.conv_op.weight_wave_lower_addr, 
                                        self.conv_op.weight_wave_upper_addr,
                                        self.conv_op.ifmap_count)
            dram_ifmaps_waveops = tpb.statebuffer.circbuf_ifmaps.read_data_region(
                                        wave_id, 
                                        self.conv_op.ifmap_wave_lower_addr, 
                                        self.conv_op.ifmap_wave_upper_addr,
                                        self.conv_op.ifmap_count)
            tpb.pearray.wave_fp16_mm(pearray_packed_ifmaps, pearray_packed_weights, self.conv_op.psum_bank_dst, psum_add)
            matmul_waveop = self.gen_matmul_waveop(tpb, wave_id, psum_add)
            tpb.waveop_stream.add_linked(matmul_waveop, dram_ifmaps_waveops + dram_weights_waveops)
            # collect statistics
            tpb.pearray.num_ops_executed += self.conv_op.ofmap_count * self.conv_op.ofmap_full_tile_sz * self.conv_op.ifmap_count
            return True
        
    # execute remaining fused ops
    def execute_tile_waveops (self, tpb, wave_id, tile_id, psum_bank_src, bias, psum_temp):
        op_list_iter = iter(range(1, len(self)))
        op_list = self
        for i in op_list_iter:
            layer_type = self[i].data['layer_type'] 
            if (re.search(r"Relu|Tanh|Sigmoid|Exp|Identity|Lrelu|Prelu", layer_type)):
                tpb.activate.wait_tile_done(tile_id)
                psum_temp = tpb.activate.relu(psum_temp)
                psum_bank_dst = 2
                dst_is_psum = False
                if (i != len(op_list)-1):
                    dst_is_psum = True
                    tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz, psum_temp)
                tpb.gen_act_waveop_inline(None, op_list[i], self.conv_op, tile_id, 
                                          psum_bank_src, dst_is_psum, psum_bank_dst, [], 0)
                psum_bank_src = psum_bank_dst
            elif (layer_type == 'BiasAdd'):
                tpb.activate.wait_tile_done(tile_id)
                bias_chan_start = tile_id.m_id * PEArray.NUM_COLS
                bias_chan_end = min(bias_chan_start + PEArray.NUM_COLS, self.conv_op.M)
                bias_extracted = np.zeros(PEArray.NUM_COLS)
                bias_extracted[0 : bias_chan_end - bias_chan_start] = bias[bias_chan_start : bias_chan_end]
                bias_addr = bias_chan_start * op_list[i].item_sz
                dram_bias_waveops = tpb.statebuffer.circbuf_bias.read_data_region(wave_id, bias_addr, bias_addr, self.conv_op.ifmap_count)
                #x = DBG_DUMP_PSUM_COL("PSUM col0 before BiasAdd (FP32): ", psum_temp, 0)
                psum_temp = tpb.activate.biasadd(psum_temp, bias_extracted)
                #y = DBG_DUMP_PSUM_COL("PSUM col0 after BiasAdd: ", psum_temp, 0)
                #print(y-x)
                psum_bank_dst = 2
                dst_is_psum = False
                if (i+1 < len(op_list) and re.search(r"Relu|Tanh|Sigmoid|Exp|Identity|Lrelu|Prelu", op_list[i+1].data['layer_type'])):
                    psum_temp = tpb.activate.act(op_list[i+1].data['layer_type'], psum_temp)
                    if (i+1 != len(op_list)-1):
                        dst_is_psum = True
                        tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz, psum_temp)
                    tpb.gen_act_waveop_inline(op_list[i], op_list[i+1], self.conv_op, tile_id, 
                                              psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_addr)
                    tpb.statebuffer.circbuf_bias.free_data_region(bias_addr, bias_addr)
                    psum_bank_src = psum_bank_dst
                    next(op_list_iter)
                else:                                    
                    if (i != len(op_list)-1):
                        dst_is_psum = True
                        tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz, psum_temp)
                    tpb.gen_act_waveop_inline(op_list[i], None, self.conv_op, tile_id, 
                                              psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_addr)
                    tpb.statebuffer.circbuf_bias.free_data_region(bias_addr, bias_addr)
                    psum_bank_src = psum_bank_dst
            elif (layer_type == 'ResAdd'):
                tpb.pool.wait_tile_done(tile_id)
                dram_resadd_waveop = tpb.statebuffer.circbuf_scratch.read_data_region(wave_id, self.conv_op.ofmap_tile_lower_addr, self.conv_op.ofmap_tile_upper_addr, self.conv_op.ifmap_count)
                residue_tile = np.zeros((self.conv_op.ofmap_full_tile_sz, PEArray.NUM_COLS))
                residue_ifmaps = np.zeros((self.conv_op.ofmap_full_tile_sz, PEArray.NUM_COLS), dtype=np.float32)
                for j in range(PEArray.NUM_COLS):
                    M_idx = tile_id.m_id * PEArray.NUM_COLS + j
                    if (M_idx >= self.conv_op.M):
                        break
                    else:
                        # NCHW
                        residue_tile_ifmap = np.zeros((self.conv_op.ofmap_full_tiley_sz, self.conv_op.ofmap_full_tilex_sz), dtype=np.float32)
                        residue_tile_ifmap[0:self.conv_op.ofmap_cropped_tile_height, 0:self.conv_op.ofmap_cropped_tile_width] = tpb.statebuffer.circbuf_scratch.dram_data[
                                tile_id.n_id, 
                                M_idx, 
                                self.conv_op.ofmap_tile_y_start : self.conv_op.ofmap_tile_y_start + self.conv_op.ofmap_cropped_tile_height, 
                                self.conv_op.ofmap_tile_x_start : self.conv_op.ofmap_tile_x_start + self.conv_op.ofmap_cropped_tile_width]
                        residue_ifmaps[:,j] = residue_tile_ifmap.flatten()
                x1 = DBG_DUMP_PSUM_COL("PSUM col0 before ResAdd (FP32): ", psum_temp, 0)
                x2 = DBG_DUMP_PSUM_COL("Residue col0 before ResAdd (FP32): ", residue_ifmaps, 0)
                psum_temp = tpb.pool.resadd(psum_temp, residue_ifmaps)
                y1 = DBG_DUMP_PSUM_COL("PSUM col0 after RessAdd (FP32): ", psum_temp, 0)
                psum_bank_dst = 3
                dst_is_psum = False
                if (i != len(op_list)-1):
                    dst_is_psum = True
                    tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz, psum_temp)
                tpb.gen_resadd_waveop_inline(op_list[i], self.conv_op, tile_id, psum_bank_src, dst_is_psum, psum_bank_dst, self.conv_op.ofmap_tile_lower_addr)
                tpb.statebuffer.circbuf_scratch.free_data_region(self.conv_op.ofmap_tile_lower_addr, self.conv_op.ofmap_tile_upper_addr)
                psum_bank_src = psum_bank_dst
            elif ((layer_type == 'AvgPool') or (layer_type == 'MaxPool')):
                tpb.activate.wait_tile_done(tile_id)
                self[i].compute_ofmap_tile_info(tile_id)
                #tilex = self.conv_op.ofmap_cropped_tile_width
                #tiley = self.conv_op.ofmap_cropped_tile_height
                tilex = self[i].ofmap_full_tilex_sz * self[i].stride_x
                tiley = self[i].ofmap_full_tiley_sz * self[i].stride_y
                if (layer_type == 'AvgPool'):
                    psum_temp = tpb.pool.avg(psum_temp, self[i].stride_x, self[i].pool_window_y, self[i].Tn, tilex, tiley)
                else:
                    psum_temp = tpb.pool.max(psum_temp, self[i].stride_x, self[i].pool_window_y, self[i].Tn, tilex, tiley, self[i].ofmap_full_tilex_sz, self[i].ofmap_full_tiley_sz)
                tpb.gen_fused_pool_waveop_inline(op_list, tile_id, psum_bank_src)
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
        'AvgPool': "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|ResAdd",
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

    # add forward edges for forward traversals        
    def add_forward_refs(self, starting_node):
        if (starting_node != None):
            #print (starting_node.data['layer_name'], len(starting_node.prev))
            if (len(starting_node.prev) > 0):
                for i in starting_node.prev:
                    i.add_next(starting_node)
                    self.add_forward_refs(i)

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
                    if (l['layer_type'] == "Input"):
                        self.first_node = new_node
                # assume the last node is the last one processed (JSON graph is in order), at least for the last one
                self.last_node = new_node                
                self.node_dict[ l['layer_name'] ] = new_node
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
        fused_ops.add(self.current_node)
        for i in self.current_node.next:
            print(i.data['layer_type'], ":", i.data['layer_name'])
        fused_ops = self.get_next_fused_op(fused_ops)
        # if there are multiple next nodes
        next_nodes = fused_ops[-1].next
        last_node_type = fused_ops[-1].data['layer_type']
        if (len(next_nodes) == 1):
            self.current_node = next_nodes[0]   
        elif (len(next_nodes) > 1):
            # follow the leg that goes to ResAdd directly first, if it exists
            for i in range(len(next_nodes)):
                if (next_nodes[i].data['layer_type'] == "ResAdd"):
                    resadd_node = next_nodes[i]
                    del next_nodes[i]
                    next_nodes.insert(0, resadd_node)
            # pick the first leg as current_node                        
            self.current_node = next_nodes[0]
            # save the remaining legs in a list
            self.last_split_next_nodes = next_nodes[1:]
        else:
            self.current_node = None
            self.last_split_next_nodes = []
        # if the last node is Conv or MatMul, add an identity pool op
        if (last_node_type == "Conv" or last_node_type == "MatMul"):
            fused_ops.add(self.gen_id_pool_op(fused_ops[-1]))
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
    def gen_act_waveop_inline(self, biasadd_op, act_op, conv_op, tile_id, psum_bank_src, dst_is_psum, psum_bank_dst, dram_bias_waveops, bias_start):
        layer_name = ""
        bias_add_en = False
        bias_atom_id = 0
        bias_offset_in_atom = 0
        # TODO: update in_dtype when src_is_psum is added
        in_dtype = "float32"
        out_dtype = "float32"
        if (biasadd_op != None):
            bias_add_en = True
            bias_atom_id = self.statebuffer.circbuf_bias.get_atom(bias_start)
            bias_offset_in_atom = bias_start % self.statebuffer.circbuf_bias.atom_data_sz
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
              'out_dtype'               : out_dtype,
              'src_psum_bank_id'        : psum_bank_src,
              'src_x_step'              : 1,
              'src_x_num'               : dst_x_num,
              'src_y_step'              : dst_y_step,
              'src_y_num'               : dst_y_num * dst_z_num,
              'dst_is_psum'             : dst_is_psum,
              'dst_psum_bank_id'        : psum_bank_dst,
              'dst_sb_atom_id'          : 0, # Need to adjust this after allocating atoms
              'dst_sb_offset_in_atom'   : tpb.statebuffer.circbuf_scratch.get_atom_offset(conv_op.ofmap_tile_lower_addr),
              'dst_x_step'              : 1,
              'dst_x_num'               : dst_x_num,
              'dst_y_step'              : dst_y_step,
              'dst_y_num'               : dst_y_num,
              'dst_z_step'              : dst_z_step,
              'dst_z_num'               : dst_z_num,
              'num_partitions'          : num_partitions,
              'bias_add_en'             : bias_add_en,
              'bias_atom_id'            : bias_atom_id,
              'bias_offset_in_atom'     : bias_offset_in_atom,
            }
        self.waveop_stream.add_linked(instr, dram_bias_waveops)

    # generate ResAdd instruction and add it to instruction stream
    def gen_resadd_waveop_inline(self, op, conv_op, tile_id, psum_bank_src, dst_is_psum, psum_bank_dst, data_start):
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
              'src_a_sb_atom_id'        : self.statebuffer.circbuf_scratch.get_atom(data_start),
              'src_a_sb_offset_in_atom' : self.statebuffer.circbuf_scratch.get_atom_offset(data_start),
              'src_a_x_step'            : 1,
              'src_a_x_num'             : dst_x_num,
              'src_a_y_step'            : dst_y_step,
              'src_a_y_num'             : dst_y_num,
              'src_a_z_step'            : dst_z_step,
              'src_a_z_num'             : dst_z_num,
              'src_b_is_psum'           : True,
              'src_b_psum_bank_id'      : psum_bank_src,
              'src_b_psum_bank_offset'  : 0,
              'src_b_sb_atom_id'        : self.statebuffer.circbuf_scratch.get_atom(data_start),
              'src_b_sb_offset_in_atom' : self.statebuffer.circbuf_scratch.get_atom_offset(data_start),
              'src_b_x_step'            : 1,
              'src_b_x_num'             : dst_x_num,
              'src_b_y_step'            : dst_y_step,
              'src_b_y_num'             : dst_y_num,
              'src_b_z_step'            : dst_z_step,
              'src_b_z_num'             : dst_z_num,
              'dst_is_psum'             : dst_is_psum,
              'dst_psum_bank_id'        : psum_bank_dst,
              'dst_psum_bank_offset'    : 0,
              'dst_sb_atom_id'          : self.statebuffer.circbuf_scratch.get_atom(data_start),
              'dst_sb_offset_in_atom'   : self.statebuffer.circbuf_scratch.get_atom_offset(data_start),
              'dst_x_step'              : 1,
              'dst_x_num'               : dst_x_num,
              'dst_y_step'              : dst_y_step,
              'dst_y_num'               : dst_y_num,
              'dst_z_step'              : dst_z_step,
              'dst_z_num'               : dst_z_num,
              'num_partitions'          : num_partitions,
            }
        self.waveop_stream.add_linked(instr, [])

    def gen_fused_pool_waveop_inline (self, fused_ops, tile_id, psum_bank_src):
        pool_waveop = fused_ops.gen_pool_waveop(self, tile_id, True, psum_bank_src)
        self.waveop_stream.add_linked(pool_waveop, [])

    def gen_unfused_pool_waveop_inline (self, fused_ops, tile_id, dram_waveops):
        pool_waveop = fused_ops.gen_pool_waveop(self, tile_id, False, 0)
        self.waveop_stream.add_linked(pool_waveop, dram_waveops)

    # reset statistics
    def reset_stats(self):        
        self.pearray.num_wave_fp16_mm = 0
        self.pearray.num_ops_executed = 0
        if (self.statebuffer.circbuf_weights.DRAM_elem_read):
            self.statebuffer.circbuf_weights.DRAM_elem_read = 0
        self.statebuffer.circbuf_ifmaps.DRAM_elem_read = 0

    # print out statistics
    def calculate_compute_to_load_ratio(self):
        num_of_wave_ops = self.pearray.num_wave_fp16_mm * self.pearray.NUM_ROWS * self.pearray.NUM_COLS * self.pearray.MAX_WAVE_SIZE
        num_of_reads = self.statebuffer.circbuf_weights.DRAM_elem_read + self.statebuffer.circbuf_ifmaps.DRAM_elem_read
        minimum_reads = self.statebuffer.circbuf_weights.dram_data.size + self.statebuffer.circbuf_ifmaps.dram_data.size
        ideal_compute_to_load_ratio = self.pearray.num_ops_executed / self.statebuffer.circbuf_weights.dram_data.size * 2
        actual_to_min_read_ratio = num_of_reads / minimum_reads
        wave_op_efficiency = self.pearray.num_ops_executed / num_of_wave_ops
        print("num_of_wave_ops: ",num_of_wave_ops, "num_of_op_executed: ", self.pearray.num_ops_executed, "wave_op_efficiency", wave_op_efficiency)
        print("num_of_reads: ",num_of_reads,"minimum_reads: ",minimum_reads, "actual_to_min_read_ratio: ",actual_to_min_read_ratio,\
              "weights_read:",self.statebuffer.circbuf_weights.DRAM_elem_read,"ifmaps_read: ",self.statebuffer.circbuf_ifmaps.DRAM_elem_read)
        print("min weight read:",self.statebuffer.circbuf_weights.dram_data.size, "min ifmap read: ",self.statebuffer.circbuf_ifmaps.dram_data.size)
        print("ideal_compute_to_load_ratio: ",ideal_compute_to_load_ratio)
        print("number_of_waves: ",self.pearray.num_wave_fp16_mm)
        self.reset_stats()

    # Execute an unfused pooling operator
    def execute_unfused_pool_op(self, inputs, result_file):
        # save result to create a scratch space (in DRAM), then use circular buffer load to populate params
        result = self.statebuffer.circbuf_scratch.load_data(op_list[-1], result_file)

        # wave loop ordering scheme: nmhw
        pool_op = op_list[0]
        for n_id in range(pool_op.n):
            for m_id in range(pool_op.m):
                for h_id in range(pool_op.h):
                    for w_id in range(pool_op.w):
                        tile_id = TileID(n_id, m_id, h_id, w_id)
                        pool_op.compute_ofmap_tile_info(tile_id)
                        # set r_id and s_id in wave_id to zero since we are not doing convolution
                        wave_id = WaveID(n_id, m_id, h_id, w_id, 0, 0, 0)
                        psum_fake = pool_op.pack_wave_ifmaps_unfused_pooling(inputs, wave_id, for_unfused_pooling=True)
                        input_tiley = pool_op.ifmap_wave_upper_coordy - pool_op.ifmap_wave_lower_coordy + 1
                        input_tilex = pool_op.ifmap_wave_upper_coordx - pool_op.ifmap_wave_lower_coordx + 1
                        output_tiley = pool_op.ofmap_full_tiley_sz
                        output_tilex = pool_op.ofmap_full_tilex_sz
                        psum_fake_extract = psum_fake [0:input_tiley*input_tilex, :]
                        psum_temp = self.pool.max(psum_fake_extract, pool_op.stride_x, pool_op.pool_window_y, pool_op.Tn, input_tilex, input_tiley, output_tilex, output_tiley)
                        dram_ifmaps_waveops = tpb.statebuffer.circbuf_ifmaps.read_data_region(
                                                    wave_id, 
                                                    pool_op.ifmap_wave_lower_addr, 
                                                    pool_op.ifmap_wave_upper_addr,
                                                    pool_op.ifmap_count)
                        self.gen_unfused_pool_waveop_inline(op_list, tile_id, dram_ifmaps_waveops)
                        tpb.statebuffer.circbuf_ifmaps.free_data_region(pool_op.ifmap_wave_lower_addr, pool_op.ifmap_wave_upper_addr)
                        for j in range(PEArray.NUM_COLS):
                            M_idx = wave_id.m_id * PEArray.NUM_COLS + j
                            if (M_idx >= pool_op.M):
                                break
                            else:
                                # For now, multiply zeros, and at the ofmap, extract tile with zeros, then clip
                                result_tile = (psum_temp[0 : pool_op.ofmap_full_tile_sz, j]).reshape((pool_op.ofmap_full_tiley_sz, pool_op.ofmap_full_tilex_sz))
                                # NCHW
                                result[n_id, 
                                        M_idx, 
                                        pool_op.ofmap_tile_y_start : pool_op.ofmap_tile_y_start + pool_op.ofmap_cropped_tile_height, 
                                        pool_op.ofmap_tile_x_start : pool_op.ofmap_tile_x_start + pool_op.ofmap_cropped_tile_width]\
                                    = result_tile[0:pool_op.ofmap_cropped_tile_height, 0:pool_op.ofmap_cropped_tile_width]
                        # for scheduling, map resulting tile into portion of atom that is itself mapped to a portion in DRAM (file)
                        dram_output_waveops = self.statebuffer.circbuf_scratch.write_data_region(tile_id, pool_op.ofmap_tile_lower_addr, pool_op.ofmap_tile_upper_addr, pool_op.ofmap_count)
                        # The pooling destination need to be adjusted after the above writes to data region
                        if (self.waveop_stream.last_main_waveop['waveop_type'] == "Pool" or self.waveop_stream.last_main_waveop['waveop_type'] == "Activation"):
                            self.waveop_stream.last_main_waveop['dst_sb_atom_id'] = self.statebuffer.circbuf_scratch.current_atom_id
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
        self.reset_stats()

        # reset scratch buffer for now (TODO: keep some atoms for next layer)
        self.statebuffer.reset_all()
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
        if (op_list.has_resadd):
            self.statebuffer.circbuf_scratch.load_data(op_list.resadd_op)

        # initial psum bank is 0
        op_list.conv_op.set_psum_bank(0)
        # start tensor computation by clearing psum bank
        psum_add = False                               

        # wave loop ordering scheme: nmhwcRS
        for n_id in range(op_list.conv_op.n):
            for m_id in range(op_list.conv_op.m):
                for h_id in range(op_list.conv_op.h):
                    for w_id in range(op_list.conv_op.w):
                        tile_id = TileID(n_id, m_id, h_id, w_id)
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
                        self.statebuffer.circbuf_ifmaps.free_data_region(op_list.conv_op.ifmap_tile_lower_addr, op_list.conv_op.ifmap_tile_upper_addr)
                        self.statebuffer.circbuf_weights.free_data_region(op_list.conv_op.weight_tile_lower_addr, op_list.conv_op.weight_tile_upper_addr)
                        # extract PSUM data
                        psum_bank_src = op_list.conv_op.get_psum_bank()
                        psum_temp = self.pearray.extract_psum(psum_bank_src, 0, op_list.conv_op.ofmap_full_tile_sz)
                        # go through the remaining operations
                        psum_temp = op_list.execute_tile_waveops(tpb, wave_id, tile_id, psum_bank_src, bias, psum_temp)
                        # if operation is the last one, dump current result into a portion of final result
                        output_params_op = op_list.conv_op
                        if (op_list.has_pool):
                            output_params_op = op_list.pool_op
                        for j in range(PEArray.NUM_COLS):
                            M_idx = wave_id.m_id * PEArray.NUM_COLS + j
                            if (M_idx >= output_params_op.M):
                                break
                            else:
                                # For now, multiply zeros, and at the ofmap, extract tile with zeros, then clip
                                result_tile = (psum_temp[0 : output_params_op.ofmap_full_tile_sz, j]).reshape((output_params_op.ofmap_full_tiley_sz, output_params_op.ofmap_full_tilex_sz))
                                #DBG_DUMP_ARRAY("Intermediate result (FP32): ", result_tile)
                                # NCHW
                                result[n_id, 
                                        M_idx, 
                                        output_params_op.ofmap_tile_y_start : output_params_op.ofmap_tile_y_start + output_params_op.ofmap_cropped_tile_height, 
                                        output_params_op.ofmap_tile_x_start : output_params_op.ofmap_tile_x_start + output_params_op.ofmap_cropped_tile_width]\
                                    = result_tile[0:output_params_op.ofmap_cropped_tile_height, 0:output_params_op.ofmap_cropped_tile_width]
                        # for scheduling, map resulting tile into portion of atom that is itself mapped to a portion in DRAM (file)
                        dram_output_waveops = self.statebuffer.circbuf_scratch.write_data_region(tile_id, output_params_op.ofmap_tile_lower_addr, output_params_op.ofmap_tile_upper_addr, output_params_op.ofmap_count)
                        # The pooling destination need to be adjusted after the above writes to data region
                        if (self.waveop_stream.last_main_waveop['waveop_type'] == "Pool" or self.waveop_stream.last_main_waveop['waveop_type'] == "Activation"):
                            self.waveop_stream.last_main_waveop['dst_sb_atom_id'] = self.statebuffer.circbuf_scratch.current_atom_id
                        self.waveop_stream.add_outputs(dram_output_waveops)
                        # Advance to new bank (ping-pong between 0 and 1) for PEArray, while the old bank is being processed by other engines
                        op_list.conv_op.set_psum_bank((op_list.conv_op.get_psum_bank()+1)%2)
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
        self.calculate_compute_to_load_ratio()

        # reset scratch buffer for now (TODO: keep some atoms for next layer)
        self.statebuffer.reset_all()

        if (args.debug > 1): print("DBG: Total wave elements: ", op_list.conv_op.ofmap_wave_total_elems)

        return result                    

# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kgraph", help="K-graph Json file to read")
    parser.add_argument("--wavegraph", help="Wave-graph Json file to write")
    parser.add_argument("--dot", help="Dot file to write")
    parser.add_argument("--debug", type=int, default=DEBUG_LEVEL_DEFAULT, help="Debug level")
    parser.add_argument("--golden_inputs", action='store_true', help="Use golden files as inputs for each layer")
    args = parser.parse_args()

    if (args.debug > 5): np.set_printoptions(threshold=np.nan)

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
        result_file = op_list[-1].data['ref_file'].replace(".npy", "-midout.npy")
        print("Output file for layer %s is %s"%(op_list[-1].data['layer_name'], result_file))

        # Check init op
        if (op_list[0].data['layer_type'] == "Input"):
            tpb.statebuffer.saved_result_files[op_list[0].data['layer_name']] = op_list[0].data['ref_file']
        # Check conv fused op
        elif (op_list[0].data['layer_type'] == "Conv" or op_list[0].data['layer_type'] == "MatMul"):
            inputs = tpb.statebuffer.circbuf_ifmaps.load_data(op_list[0])
            results = tpb.execute_conv_ops(inputs, result_file)
        elif (op_list[0].data['layer_type'] == "AvgPool" or op_list[0].data['layer_type'] == "MaxPool"):
            inputs = tpb.statebuffer.circbuf_ifmaps.load_data(op_list[0])
            results = tpb.execute_unfused_pool_op(inputs, result_file)
        else:        
            print("ERROR: Unrecognized first operation %s"%op_list[0].data['layer_type'])
            exit(-1)

        # Check results against pre-computed results           
        if (op_list[0].data['layer_type'] != "Input"):
            if 'ref_file' in op_list[-1].data:
                outputs = np.load(op_list[-1].data['ref_file'])
                diff = results - outputs
                if (args.debug > 2): print("\nInput IFMAPS:\n", inputs)
                if (args.debug > 1): print("\nComputed OFMAPS:\n", results)
                if (args.debug > 1): print("\nExpected OFMAPS:\n", outputs)
                if (args.debug > 1): print("\nDiffed   OFMAPS:\n", diff)
                if (not npu.allclose(results, outputs, 1/100, 1e-5, verbose=True)):
                    print("\nERROR: layer %s computed OFMAPS is not equal to expected OFMAPS!\n"%(op_list[-1].data['layer_name']))
                    num_mismatches += 1

    # write out wavegraph           
    wavegraph_json = kgraph_json
    if (args.wavegraph != None): 
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

    # write out dot graph in SVG format
    if (args.dot != None):            
        dot = Digraph()
        for i in tpb.waveop_stream:
            dot.node(i['waveop_name'], i['waveop_name'])
            for j in i['previous_waveops']:
                dot.edge(j, i['waveop_name'])
        (dotfile_root, dotfile_ext) = os.path.splitext(args.dot)                
        dot.format = dotfile_ext[1:]
        dot.render(dotfile_root)
        print("INFO: Wrote " + args.dot)

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

    # check for comparison errors
    if (num_mismatches > 0):
        print("\nFAILED (num mismatches %d)"%num_mismatches)
    else:        
        print("\nPASSED")
