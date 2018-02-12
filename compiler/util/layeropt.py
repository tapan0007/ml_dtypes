import json
import os
import math
import re
import numpy as np
import argparse
from graphviz import Digraph

DEBUG_LEVEL_DEFAULT=1

#np.set_printoptions(threshold=np.nan)

#kgraph_file = os.environ['KAENA_PATH'] + "/compiler/tffe/rundir/0-1conv0/trivnet_compiler.json"

# TODO: use datatype from K-Graph to cast everything to that datatype
# TODO: multiple atoms within waveop (batching)

def ceildiv(a,b):
    return (a//b) + (a%b != 0)

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

##################################################################################
# Pooling properties and methods
class Pool:
    def wait_tile_done(self, tile_id):
        pass
    def avg(self, in_array, stride, pool_window_size):
        # cannot use the below assertion because we only have layer property for conv layer
        # assert (stride == pool_window_size)
        num_cols = in_array.shape[1]
        print("pooling fun")
        print(in_array.shape," ",num_cols)
        # will reshape the tensor into pool_window_size "box" and average within the box
        return in_array.reshape(pool_window_size,pool_window_size,-1,num_cols).mean(axis=(0,1))
    def resadd(self, array_a, array_b):
        return array_a + array_b
    def max(self, in_array):
        return in_array

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
        self.circbuf_ifmaps  = CircularBuffer("ifmaps",  8,        self.SB_ATOM_SZ, 0)
        self.circbuf_weights = CircularBuffer("weights", 96-8-8-8, self.SB_ATOM_SZ, 8)
        self.circbuf_bias    = CircularBuffer("bias",    8,        self.SB_ATOM_SZ, 96-8-8)
        self.circbuf_scratch = CircularBuffer("scratch", 8,        self.SB_ATOM_SZ, 96-8)
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
    def __init__(self, circbuf_type, capacity, atom_sz, start):
        self.capacity = capacity
        self.atom_sz = atom_sz
        self.item_sz = 2
        self.start = start
        self.circbuf_type = circbuf_type
        self.reset()

    def reset(self):
        self.head_pointer = self.start
        self.tail_pointer = self.start
        self.current_atom_id = self.start
        self.atom_data_sz = self.atom_sz
        self.count = 0
        self.max_count = 0
        self.allocated = np.zeros(self.capacity, dtype=bool)
        self.dram_data_file = None
        self.dram_data = None
        self.dram_data_len = 0
        self.layer_name = ""
        self.layer_type = "Output"
        self.addr2atom = {}

    def load_data(self, waveop):
        self.reset()
        self.layer_name = waveop.data['layer_name']
        self.layer_type = waveop.data['layer_type']
        if (self.layer_type == 'Input' or self.layer_type == 'Const'):
            self.dram_data_file = waveop.data['ref_file']
        else:            
            self.dram_data_file = waveop.data['kernel_file']
        self.load_file(self.dram_data_file)                
        #print("Loaded %s for layer %s, first data is %f, data size is %d bytes, atom size %d bytes, atom data size %d bytes"%(self.dram_data_file, self.layer_name, self.dram_data[0,0,0,0], self.item_sz, self.atom_sz, self.atom_data_sz)) 
        return self.dram_data

    def load_file(self, file, ofmap_full_tiley_sz = 0):      # use waveop instead of waveop 
        self.dram_data_file = file
        self.dram_data = np.load(self.dram_data_file)
        self.item_sz = self.dram_data.dtype.itemsize   
        self.dram_data_len = self.dram_data.size * self.item_sz
        # Determine the actual amount of data per atom
        # TODO: come up with a better formula for atom_data_sz to take care of all cases
        # Constraints for atom_data_sz for Conv IFMAPs/OFMAPs: 
        #   * less than or equal to 1KB
        #   * IFMAP: multiple of H*W, else multiple of W
        #   * OFMAP: multiple of E*F, else multiple of OFMAP tile size
        #   * Filter: multiple of R*S*M, else multiple of S*M, else multiple of M
        #   * different FMAPs in one batch will be in different atoms (for now)
        #   * different FMAPs folds will be in different atoms (for now)
        # TODO: refactor the following function since it is used in multiple places
        if (self.layer_type == 'Input' or self.layer_type == 'Const' or self.layer_type == 'Output'):
            N, C, H, W = [i*self.item_sz for i in self.dram_data.shape]
            if (H*W <= self.atom_sz):
                multiple = self.atom_sz // (H*W)
                self.atom_data_sz = (H*W) * min(C, multiple)
            elif (W <= self.atom_sz):
                multiple = self.atom_sz // W
                if (self.layer_type == 'Output'):
                    multiple = min(H, multiple)
                    if (ofmap_full_tiley_sz < multiple):
                        multiple = (multiple//ofmap_full_tiley_sz)*ofmap_full_tiley_sz
                    self.atom_data_sz = W * min(H, multiple)
                else:                    
                    self.atom_data_sz = W * min(H, multiple)
            else:
                self.atom_data_sz = self.atom_sz
        else:            
            C, R, S, M = [i*self.item_sz for i in self.dram_data.shape]
            if (R*S*M <= self.atom_sz):
                multiple = self.atom_sz // (R*S*M)
                self.atom_data_sz = (R*S*M) * min(C, multiple)
            elif (S*M <= self.atom_sz):
                multiple = self.atom_sz // (S*M)
                self.atom_data_sz = (S*M) * min(R, multiple)
            elif (M <= self.atom_sz):
                multiple = self.atom_sz // M
                self.atom_data_sz = M * min(S, multiple)
            else:
                self.atom_data_sz = self.atom_sz
        print("Loaded %s for layer %s, first data is %f, data size is %d bytes, atom size %d bytes, atom data size %d bytes"%(self.dram_data_file, self.layer_name, self.dram_data[0,0,0,0], self.item_sz, self.atom_sz, self.atom_data_sz)) 
        return self.dram_data

    def gen_dram_read_waveop(self, wave_id, atom_id, chunk_id):
        offset = chunk_id*self.atom_data_sz
        length = self.atom_data_sz
        if ((offset + length) > self.dram_data_len):
            length = self.dram_data_len % self.atom_data_sz
        return {
              'previous_waveops' : [],
              'waveop_type'      : "SBAtomFile",
              'waveop_name'      : self.layer_name+"/SBAtomFile_%d"%chunk_id,
              'layer_name'       : self.layer_name,
              'atom_id'          : atom_id,
              'ref_file'         : self.dram_data_file,
              'offset_in_file'   : offset,
              'length'           : length,
              'ifmaps_replicate' : False,
              'ifmaps_fold_idx'  : wave_id.c_id,
              'batch_fold_idx'   : wave_id.n_id 
            }

    def gen_dram_save_waveop(self, wave_id, atom_id, chunk_id):
        offset = chunk_id*self.atom_data_sz
        length = self.atom_data_sz
        if ((offset + length) > self.dram_data_len):
            length = self.dram_data_len % self.atom_data_sz
        return {
              'previous_waveops' : [],
              'waveop_type'      : "SBAtomSave",
              'waveop_name'      : self.layer_name+"/SBAtomSave_%d"%chunk_id,
              'layer_name'       : self.layer_name,
              'atom_id'          : atom_id,
              'ref_file'         : self.dram_data_file,
              'offset_in_file'   : offset,
              'length'           : length,
              'ofmaps_fold_idx'  : wave_id.m_id,
              'batch_fold_idx'   : wave_id.n_id 
            }

    def read_data_region(self, wave_id, lower_addr, upper_addr):
        dram_waveops = []
        if (args.debug > 2): print("%s: read byte range %d to %d"%(self.circbuf_type, lower_addr, upper_addr))
        lower_addr_chunked = lower_addr // self.atom_data_sz
        upper_addr_chunked = upper_addr // self.atom_data_sz
        for i in range(lower_addr_chunked, upper_addr_chunked+1):
            if i not in self.addr2atom:
                atom_id = self.allocate_atom()
                dram_waveops.append(self.gen_dram_read_waveop(wave_id, atom_id, i))
                self.addr2atom[i] = atom_id
        return dram_waveops

    def write_data_region(self, wave_id, lower_addr, upper_addr):
        if (args.debug > 2): print("%s: write byte range %d to %d"%(self.circbuf_type, lower_addr, upper_addr))
        dram_waveops = []
        lower_addr_chunked = lower_addr // self.atom_data_sz
        upper_addr_chunked = upper_addr // self.atom_data_sz
        for i in range(lower_addr_chunked, upper_addr_chunked+1):
            if i not in self.addr2atom:
                atom_id = self.allocate_atom()
                dram_waveops.append(self.gen_dram_save_waveop(wave_id, atom_id, i))
                self.addr2atom[i] = atom_id
        # assuming that we always write to the last piece of atom last, when 
        # there's a write to last piece of atom, trigger to dump to DRAM and deallocate atom
        # TODO: optimize by keep some atoms between layers
        if (upper_addr == self.dram_data_len - self.item_sz or upper_addr == (upper_addr_chunked+1)*self.atom_data_sz - self.item_sz):
            for i in range(lower_addr_chunked, upper_addr_chunked+1):
                self.free_atom(self.addr2atom[i])
                del self.addr2atom[i]
        return dram_waveops

    def free_data_region(self, lower_addr, upper_addr):
        if (args.debug > 2): print("%s: free byte range %d to %d"%(self.circbuf_type, lower_addr, upper_addr))
        lower_addr_chunked = lower_addr // self.atom_data_sz
        upper_addr_chunked = upper_addr // self.atom_data_sz
        for i in range(lower_addr_chunked, upper_addr_chunked+1):
            if i in self.addr2atom:
                self.free_atom(self.addr2atom[i])
                # keep data around just in case, but allow pointers to wrap around
                #del self.addr2atom[i]

    def allocate_atom(self):
        if (self.count == self.capacity):
            print ("ERROR %s: no more space during allocate_atom for layer %s!"%(self.circbuf_type, self.layer_name))
            exit(-1)
            return -1
        self.current_atom_id = self.tail_pointer
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
        print("STATS circular buffer type %s layer %s: capacity %d atom size %d atom data size %d atom count %d max count %d DRAM data length %d"%(self.circbuf_type, self.layer_name, self.capacity, self.atom_sz, self.atom_data_sz, self.count, self.max_count, self.dram_data_len))

##################################################################################
# Neural network node, containing data read from JSON
class KNode:
    def __init__(self, data, item_sz):
        self.prev = []
        self.next = []
        self.data = data
        self.psum_bank_dst = 0
        self.item_sz = item_sz
    def add_prev(self, prev_node):
        self.prev.append(prev_node)
    def add_next(self, next_node):
        self.next.append(next_node)

    # set/get dest PSUM bank
    def set_psum_bank(self, dest):
        self.psum_bank_dst = dest
    def get_psum_bank(self):
        return self.psum_bank_dst

    # populate common parameters for Conv and Pool
    def populate_common_params(self):
        # get input shape from previous layer's data
        assert (self.prev[0] != None)
        input_layer = self.prev[0].data
        assert (input_layer['ofmap_format'] == 'NCHW')
        self.N, self.C, self.H, self.W = input_layer['ofmap_shape']
        # get output shape from current layer's data
        layer_info = self.data
        assert (layer_info['ofmap_format'] == 'NCHW')
        self.N, self.M, self.E, self.F = layer_info['ofmap_shape']
        self.pad_north, self.pad_south = layer_info['padding'][2]
        self.pad_west, self.pad_east = layer_info['padding'][3]
        self.stride_y = layer_info['stride'][2]
        self.stride_x = layer_info['stride'][3]
        # IFMAP and OFMAP total areas
        self.HW = self.H * self.W
        self.EF = self.E * self.F
        # per kaena-85, use noodle shapes for tiles
        self.ofmap_full_tilex_sz = min(self.F, PEArray.MAX_WAVE_SIZE)   # full tile (not clipped)  
        self.ofmap_full_tiley_sz = min(self.E, PEArray.MAX_WAVE_SIZE // self.ofmap_full_tilex_sz) # full tile (not clipped)
        self.ofmap_full_tile_sz = self.ofmap_full_tilex_sz * self.ofmap_full_tiley_sz  # full tile (not clipped)
        # compute the IFMAP folds
        self.c = ceildiv(self.C, PEArray.NUM_ROWS)
        # compute the OFMAP folds
        self.m = ceildiv(self.M, PEArray.NUM_COLS)
        # computing the input map tiling       
        self.h, self.w, self.e, self.f = 1, 1, 1, 1
        # compute batch folding and batching within wave
        self.n, self.Tn = 1, 1
        if (self.EF >= PEArray.MAX_WAVE_SIZE):
            self.e = ceildiv(self.E, self.ofmap_full_tiley_sz)
            self.f = ceildiv(self.F, self.ofmap_full_tilex_sz)
        else:
            self.Tn = PEArray.MAX_WAVE_SIZE // self.EF
            if (self.Tn > self.N):
                self.Tn = self.N
            self.n = ceildiv(self.N, self.Tn)
        # heigh/width folding is the same for IFMAP and OFMAP            
        self.h = self.e
        self.w = self.f
        print("Common params for layer %s:  N=%d, M=%d, H=%d, W=%d, C=%d, E=%d, F=%d, stride_x=%d, stride_y=%d, ofmap_full_tilex_sz=%d, ofmap_full_tiley_sz=%d"
                %(self.data['layer_name'], self.N, self.M, self.H, self.W, self.C, self.E, self.F, self.stride_x, self.stride_y, self.ofmap_full_tilex_sz, self.ofmap_full_tiley_sz))

    # Compute Conv looping params
    def populate_conv_params(self):
        self.populate_common_params()
        # convolution kernel shape
        layer_info = self.data
        assert (layer_info['kernel_format'] == 'CRSM')
        self.C, self.R, self.S, self.M = layer_info['kernel_shape']
        print("Conv params for layer %s: n=%d, m=%d, h=%d, w=%d, c=%d, R=%d, S=%d, Tn=%d"
                %(self.data['layer_name'], self.n, self.m, self.h, self.w, self.c, self.R, self.S, self.Tn))

    # Compute pooling params
    def populate_pooling_params(self):
        self.populate_common_params()
        # are the dimensions from layer info correct?
        layer_info = self.data
        self.pool_window_y = layer_info['kernel_shape'][2]
        self.pool_window_x = layer_info['kernel_shape'][3]
        print("Pooling params for layer %s: ofmap_full_tilex_sz=%d, ofmap_full_tiley_sz=%d, pool_window_x=%d, pool_window_y=%d"
                %(self.data['layer_name'], self.ofmap_full_tilex_sz, self.ofmap_full_tiley_sz, self.pool_window_x, self.pool_window_y))

    # Recompute conv tile params due to fused pooling
    def recompute_conv_params(self, pooling_ifmap_width):        
        # For pooling using PSUM (fused), max tile size must be a multiple of pooling window
        self.ofmap_full_tiley_sz = (self.ofmap_full_tiley_sz // pooling_ifmap_width) * pooling_ifmap_width
        self.ofmap_full_tile_sz = self.ofmap_full_tilex_sz * self.ofmap_full_tiley_sz
        print("Recomputed Conv params due to fused pooling: ofmap_full_tiley_sz=%d"
                %(self.ofmap_full_tiley_sz))

    # compute output tile info
    def compute_ofmap_tile_info(self, tile_id):        
        self.tile_x_start = tile_id.w_id * self.ofmap_full_tilex_sz
        self.tile_y_start = tile_id.h_id * self.ofmap_full_tiley_sz
        self.tile_height = self.ofmap_full_tiley_sz
        self.tile_width = self.ofmap_full_tilex_sz
        if ((tile_id.h_id+1) * self.ofmap_full_tiley_sz > self.E):
            self.tile_height = self.E - self.tile_y_start
        if ((tile_id.w_id+1) * self.ofmap_full_tilex_sz > self.F):
            self.tile_width = self.F - self.tile_x_start
        self.tile_size = self.tile_height * self.tile_width

        # compute the address bounds for OFMAP tile within OFMAPs tensor
        # TODO: for Tn>1, need to have multiple bounds for each batch item
        self.ofmap_tile_lower_addr = int(np.ravel_multi_index(
                                            (tile_id.n_id * self.Tn, 
                                                tile_id.m_id * PEArray.NUM_COLS,
                                                self.tile_y_start, 
                                                self.tile_x_start),
                                        dims=self.data['ofmap_shape']) * self.item_sz)
        self.ofmap_tile_upper_addr = int(np.ravel_multi_index(
                                            (self.N - 1, 
                                                tile_id.m_id * PEArray.NUM_COLS,
                                                self.tile_y_start + self.tile_height - 1, 
                                                self.tile_x_start + self.tile_width - 1),
                                        dims=self.data['ofmap_shape']) * self.item_sz)

        # compute the address bounds for IFMAP tile within IFMAPs tensor
        # TODO: for Tn>1, need to have multiple bounds for each batch item
        self.ifmap_tile_lower_addr = int(np.ravel_multi_index(
                                            (tile_id.n_id * self.Tn, 
                                                0,
                                                self.tile_y_start * self.stride_y, 
                                                self.tile_x_start * self.stride_x),
                                        dims=[self.N, self.C, self.H, self.W]) * self.item_sz)
        self.ifmap_tile_upper_addr = int(np.ravel_multi_index(
                                            (self.N - 1,    # TODO: for Tn>1, need to have multiple bounds for each batch item
                                                (self.c-1) * PEArray.NUM_ROWS,
                                                self.tile_y_start * self.stride_y + self.tile_height * self.stride_y - 1, 
                                                self.tile_x_start * self.stride_x + self.tile_width * self.stride_x - 1),
                                        dims=[self.N, self.C, self.H, self.W]) * self.item_sz)

    # Pack the IFMAPs in columns to create a PE-Array IFMAPs input for a particular wave number
    #   ifmaps: IFMAPs in NCHW format
    #   wave_id: current wave ID, [n_id, m_id, h_id, w_id, c_id, r_id, s_id]
    #   return: a 256x128 array
    def pack_wave_ifmaps(self, ifmaps, wave_id):
        out_array = np.zeros((PEArray.MAX_WAVE_SIZE, PEArray.NUM_ROWS))
        # remember to extract IFMAPs starting at r_id, s_id (which should be zero for non-conv op)
        # also need to add zeros for padding
        self.ifmap_wave_lower_addr = -1
        self.ifmap_wave_upper_addr = -1
        self.ofmap_wave_lower_coord = (0, 0)
        self.ofmap_wave_upper_coord = (0, 0)
        self.psum_bank_offset = 0
        pe_row_start = wave_id.c_id * PEArray.NUM_ROWS
        pe_row_stop = min(self.C, pe_row_start + PEArray.NUM_ROWS)
        assert(pe_row_start < pe_row_stop)
        for row in range(pe_row_start, pe_row_stop):
            #out_array[:,row] = self.pack_wave_ifmap(ifmaps[:, wave_id.c_id * PEArray.NUM_ROWS + row], wave_id)
            ifmap = ifmaps[:, row]
            pe_row_offset = row - pe_row_start
            for i in range(self.Tn):
                for x in range(self.ofmap_full_tilex_sz):
                    for y in range(self.ofmap_full_tiley_sz):
                        ifmap_tilex = (wave_id.w_id * self.ofmap_full_tilex_sz + x) * self.stride_x + wave_id.s_id - self.pad_west
                        ifmap_tiley = (wave_id.h_id * self.ofmap_full_tiley_sz + y) * self.stride_y + wave_id.r_id - self.pad_north
                        ifmap_addr = i * self.ofmap_full_tile_sz + y * self.ofmap_full_tilex_sz + x
                        #print("x %d y %d ifmap_tilex %d ifmap_tiley %d"%(x, y, ifmap_tilex, ifmap_tiley))                                    
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
                                self.ifmap_wave_upper_addr = int(np.ravel_multi_index(((wave_id.n_id * self.Tn) + i, row, ifmap_tiley, ifmap_tilex),
                                                                    dims=ifmaps.shape) * ifmaps.dtype.itemsize)
                                self.ofmap_wave_upper_coord = (x, y)
                                if (self.ifmap_wave_lower_addr < 0):
                                    self.ifmap_wave_lower_addr = self.ifmap_wave_upper_addr
                                    self.ofmap_wave_lower_coord = (x, y)
                                    self.psum_bank_offset = (y*self.ofmap_full_tilex_sz + x) * ifmaps.dtype.itemsize
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
        self.ifmap_count = pe_row_stop - pe_row_start + 1
        self.ofmap_count = pe_col_stop - pe_col_start + 1
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

    def add_linked(self, waveop, side_waveops):
        input_list = []
        for i in side_waveops:
            input_list.append(i['waveop_name'])
            self.append(i)
        if (self.last_main_waveop != None):
            input_list.append(self.last_main_waveop['waveop_name'])
        waveop['previous_waveops'] = input_list
        self.append(waveop)
        self.last_main_waveop = waveop

    def add_outputs(self, waveops):
        for i in waveops:
            i['previous_waveops'].append(self.last_main_waveop['waveop_name'])
            self.append(i)

    def add_group(self, waveops):
        if (len(waveops)>0):
            if (self.last_main_waveop != None):
                waveops[0]['previous_waveops'].append(self.last_main_waveop)
            self = self + waveops
            self.last_main_waveop = waveops.last_main_waveop

##################################################################################
# FusedOp: consist of list of K-Nodes that are fused (communicate through PSUM buffers)
class FusedOp(list):

    def __init__(self):
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

    # Add operation to list of fused operations.
    # Returns True if successful; False if cannot add (i.e. Pool cannot be fused)
    def add(self, op):
        if (op.data['layer_type'] == 'AvgPool' or op.data['layer_type'] == 'MaxPool'):
            op.populate_pooling_params()
            # If not first op, pool cannot be fused with previous op if stride != pooling window
            if (len(self) != 0 and 
                    (op.stride_x != op.pool_window_x or op.stride_y != op.pool_window_y)):
                return False
            elif (self.has_pool):
                return False
            else:
                if (self.has_conv):
                    self.conv_op.recompute_conv_params(pooling_ifmap_width = op.F)
                self.pool_op = op
                self.has_pool = True
        elif (op.data['layer_type'] == 'Conv'):
            if (len(self) != 0):
                return False
            elif (self.has_conv):
                return False
            else:
                op.populate_conv_params()
                self.conv_op = op
                self.has_conv = True
        elif (op.data['layer_type'] == 'ResAdd'):
            if (self.has_resadd):
                return False
            else:
                self.resadd_op = op
                self.has_resadd = True
        elif (op.data['layer_type'] == 'MatMul'):
            if (self.has_matmul):
                return False
            else:
                self.matmul_op = op
                self.has_matmul = True
        elif (op.data['layer_type'] == 'BiasAdd'):
            if (self.has_biasadd):
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
        ofmap_wave_width = self.conv_op.ofmap_wave_upper_coord[0] - self.conv_op.ofmap_wave_lower_coord[0] + 1
        ofmap_wave_height = self.conv_op.ofmap_wave_upper_coord[1] - self.conv_op.ofmap_wave_lower_coord[1] + 1
        matmul_waveop = {
              'previous_waveops'        : [],   # to be added later
              'waveop_type'             : 'MatMul',
              'waveop_name'             : self.conv_op.data['layer_name']+"/MatMul_"+wave_id.id_string(),
              'layer_name'              : self.conv_op.data['layer_name'],
              'weights_atom_id'         : tpb.statebuffer.circbuf_weights.current_atom_id,
              'ifmaps_atom_id'          : tpb.statebuffer.circbuf_ifmaps.current_atom_id, # if multiple atoms loaded, pick the first one
              'weights_offset_in_atom'  : self.conv_op.weight_wave_lower_addr % tpb.statebuffer.circbuf_weights.atom_data_sz,  # TODO: -1 means don't load new weights
              'ifmaps_offset_in_atom'   : self.conv_op.ifmap_wave_lower_addr % tpb.statebuffer.circbuf_ifmaps.atom_data_sz,
              'wave_id_format'          : wave_id.format,
              'wave_id'                 : wave_id.show(),
              'start'                   : not(psum_add),
              'stop'                    : False,
              'stride_x'                : self.conv_op.stride_x,
              'stride_y'                : self.conv_op.stride_y,
              'psum_bank_id'            : self.conv_op.psum_bank_dst,
              'psum_bank_offset'        : self.conv_op.psum_bank_offset,
              'ifmap_count'             : self.conv_op.ifmap_count,
              'ifmap_tile_width'        : ofmap_wave_width,
              'ifmap_tile_height'       : ofmap_wave_height,
              'ofmap_count'             : self.conv_op.ofmap_count,
              'ofmap_tile_width'        : ofmap_wave_width,
              'ofmap_tile_height'       : ofmap_wave_height, 
            }
        return matmul_waveop

    # execute PEArray matrix multiply; returns True if successful (IFMAP wave is non-zero)
    def execute_matmul_waveop(self, tpb, wave_id, inputs, weights, psum_add):
        pearray_packed_weights = self.conv_op.pack_wave_conv_weights(weights, wave_id)
        pearray_packed_ifmaps = self.conv_op.pack_wave_ifmaps(inputs, wave_id)
        #print("\npearray_packed_ifmaps", wave_id.show(), "\n", pearray_packed_ifmaps)
        #print("\npearray_packed_weights", wave_id.show(), "\n", pearray_packed_weights)
        if (self.conv_op.ifmap_wave_lower_addr < 0 or self.conv_op.ifmap_wave_upper_addr < 0):
            print("WARNING layer %s: IFMAP wave (%s) has no data, so don't create waveops for this wave"%(op_list[0].data['layer_name'], wave_id.id_string()))
            return False
        else:
            dram_weights_waveops = tpb.statebuffer.circbuf_weights.read_data_region(
                                        wave_id, 
                                        self.conv_op.weight_wave_lower_addr, 
                                        self.conv_op.weight_wave_upper_addr)
            dram_ifmaps_waveops = tpb.statebuffer.circbuf_ifmaps.read_data_region(
                                        wave_id, 
                                        self.conv_op.ifmap_wave_lower_addr, 
                                        self.conv_op.ifmap_wave_upper_addr)
            tpb.pearray.wave_fp16_mm(pearray_packed_ifmaps, pearray_packed_weights, self.conv_op.psum_bank_dst, psum_add)
            matmul_waveop = self.gen_matmul_waveop(tpb, wave_id, psum_add)
            tpb.waveop_stream.add_linked(matmul_waveop, dram_weights_waveops+dram_ifmaps_waveops)
            return True

    # execute remaining fused ops
    def execute_tile_waveops (self, tpb, wave_id, tile_id, psum_bank_src, bias, psum_temp):
        op_list_iter = iter(range(1, len(self)))
        op_list = self
        for i in op_list_iter:
            layer_type = self[i].data['layer_type'] 
            if (re.search(r"Relu|Tanh|Sigmoid", layer_type)):
                tpb.activate.wait_tile_done(tile_id)
                psum_temp = tpb.activate.relu(psum_temp)
                psum_bank_dst = 2
                tpb.gen_act_waveop_inline(None, op_list[i], tile_id, psum_bank_src, psum_bank_dst, [], 0)
                if (i != len(op_list)-1):
                    tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz, psum_temp)
                psum_bank_src = psum_bank_dst
            elif (layer_type == 'BiasAdd'):
                tpb.activate.wait_tile_done(tile_id)
                bias_start = tile_id.m_id * PEArray.NUM_COLS
                bias_end = min(bias_start + PEArray.NUM_COLS, self.conv_op.M)
                bias_extracted = np.zeros(PEArray.NUM_COLS)
                bias_extracted[0 : bias_end - bias_start] = bias[bias_start : bias_end]
                dram_bias_waveops = tpb.statebuffer.circbuf_bias.read_data_region(wave_id, bias_start, bias_end)
                psum_temp = tpb.activate.biasadd(psum_temp, bias_extracted)
                psum_bank_dst = 2
                if (i+1 < len(op_list) and re.search(r"Relu|Tanh|Sigmoid", op_list[i+1].data['layer_type'])):
                    psum_temp = tpb.activate.act(op_list[i+1].data['layer_type'], psum_temp)
                    tpb.gen_act_waveop_inline(op_list[i], op_list[i+1], tile_id, psum_bank_src, psum_bank_dst, dram_bias_waveops, bias_start)
                    next(op_list_iter)
                else:                                    
                    tpb.gen_act_waveop_inline(op_list[i], None, tile_id, psum_bank_src, psum_bank_dst, dram_bias_waveops, bias_start)
                if (i != len(op_list)-1):
                    tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz, psum_temp)
                psum_bank_src = psum_bank_dst
            elif (layer_type == 'ResAdd'):
                tpb.pool.wait_tile_done(tile_id)
                dram_resadd_waveop = tpb.statebuffer.circbuf_scratch.read_data_region(wave_id, self.conv_op.ofmap_tile_lower_addr, self.conv_op.ofmap_tile_upper_addr)
                residue_tile = np.zeros((self.conv_op.ofmap_full_tile_sz, PEArray.NUM_COLS))
                for j in range(PEArray.NUM_COLS):
                    M_idx = tile_id.m_id * PEArray.NUM_COLS + j
                    if (M_idx >= self.conv_op.M):
                        break
                    else:
                        residue_tile_ifmap = tpb.statebuffer.circbuf_scratch.dram_data[
                                tile_id.n_id, 
                                j, 
                                self.conv_op.tile_y_start : self.conv_op.tile_y_start + self.conv_op.tile_height, 
                                self.conv_op.tile_x_start : self.conv_op.tile_x_start + self.conv_op.tile_width]
                        residue_tile[0:self.conv_op.tile_height*self.conv_op.tile_width,j] = residue_tile_ifmap.flatten()
                psum_temp = tpb.pool.resadd(psum_temp, residue_tile)
                psum_bank_dst = 3
                tpb.gen_resadd_waveop_inline(op_list[i], tile_id, psum_bank_src, psum_bank_dst, self.conv_op.ofmap_tile_lower_addr)
                if (i != len(op_list)-1):
                    tpb.pearray.write_psum(psum_bank_dst, 0, self.conv_op.ofmap_full_tile_sz, psum_temp)
                psum_bank_src = psum_bank_dst
            elif (layer_type == 'AvgPool'):
                tpb.activate.wait_tile_done(tile_id)
                self[i].compute_ofmap_tile_info(tile_id)
                psum_temp = tpb.pool.avg(psum_temp, self[i].stride_x, self[i].pool_window_y)
                psum_bank_dst = 3
                # TODO: generate AvgPool instruction inline
                if (i != len(op_list)-1):
                    tpb.pearray.write_psum(psum_bank_dst, 0, self[i].ofmap_full_tile_sz, psum_temp)
                psum_bank_src = psum_bank_dst
            else:
                print ("ERROR: %s is currently not yet implemented"%layer_type)
                exit(-1)
        return psum_temp

##################################################################################
# RegExs to determine whether next node is fusable or not
next_is_fusable = {
        'Conv'   : "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
        'MatMul' : "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
        'BiasAdd': "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
        'Add'    : "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
        'ResAdd' : "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
        'AvgPool': "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
        'MaxPool': "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
        'Relu'   : "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
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
        self.last_split_next_nodes = None

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
                new_node = KNode(l, self.item_sz)
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
                    new_node = KNode(l, self.item_sz)
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
            if (last_node_type in next_is_fusable):
                regex = next_is_fusable[last_node_type]
                if (re.search(regex, next_nodes[0].data['layer_type'])):               
                    # TODO: don't fuse if pool size != stride size
                    if (fused_ops.add(next_nodes[0])):
                        fused_ops = self.get_next_fused_op(fused_ops)
        return fused_ops                    

    # starting from current node position, collect as many operations as possible            
    def get_fused_ops(self):
        fused_ops = FusedOp()
        if (self.current_node == None):
            print("ERROR: found zero operations to fuse")
            exit(-1)
        # when we see ResAdd, backtrack to the last split and follow the next leg in list
        if (self.current_node.data['layer_type'] == "ResAdd" and self.last_split_next_nodes != []):
            if (args.debug > 0): print("DBG: found ResAdd, back-track to last split and follow next leg")
            self.current_node = self.last_split_next_nodes[0] 
            self.last_split_next_nodes = self.last_split_next_nodes[1:]
        fused_ops.add(self.current_node)
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
        if (args.debug > 0):
            fused_ops.show()
        return fused_ops                   

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
        self.last_psum_waveop = None

    # generate activation instruction and add it to instruction stream
    def gen_act_waveop_inline(self, biasadd_op, act_op, tile_id, psum_bank_src, psum_bank_dst, dram_bias_waveops, bias_start):
        layer_name = ""
        bias_add_en = False
        bias_atom_id = 0
        bias_offset_in_atom = 0
        if (biasadd_op != None):
            bias_add_en = True
            bias_atom_id = self.statebuffer.circbuf_bias.current_atom_id
            bias_offset_in_atom = bias_start % self.statebuffer.circbuf_bias.atom_data_sz
            layer_name = biasadd_op.data['layer_name']
        act_type = "none"    
        if (act_op != None):
            act_type = act_op.data['layer_type']
            layer_name = act_op.data['layer_name']
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : 'Activation',
              'waveop_name'             : layer_name+"/Activation_"+tile_id.id_string(),
              'layer_name'              : layer_name,
              'tile_id_format'          : tile_id.format,
              'tile_id'                 : tile_id.show(),
              'psum_bank_id_src'        : psum_bank_src,
              'psum_bank_id_dst'        : psum_bank_dst,
              'act_type'                : act_type,
              'bias_add_en'             : bias_add_en,
              'bias_atom_id'            : bias_atom_id,
              'bias_offset_in_atom'     : bias_offset_in_atom,
            }
        self.waveop_stream.add_linked(instr, dram_bias_waveops)

    # generate ResAdd instruction and add it to instruction stream
    def gen_resadd_waveop_inline(self, op, tile_id, psum_bank_src, psum_bank_dst, data_start):
        instr = {
              'previous_waveops'        : [],
              'waveop_type'             : 'ResAdd',
              'waveop_name'             : op.data['layer_name']+"/ResAdd_"+tile_id.id_string(),
              'layer_name'              : op.data['layer_name'],
              'data_atom_id'            : self.statebuffer.circbuf_scratch.current_atom_id,
              'data_offset_in_atom'     : data_start % self.statebuffer.circbuf_scratch.atom_data_sz,
              'tile_id_format'          : tile_id.format,
              'tile_id'                 : tile_id.show(),
              'psum_bank_id_src'        : psum_bank_src,
              'psum_bank_id_dst'        : psum_bank_dst,
            }
        self.waveop_stream.add_linked(instr, [])

    # Execute conv and other operations in list: for each op, load parameters and perform op with input
    def execute_conv_ops(self, inputs, result_file):
        assert (op_list[0].data['layer_type'] == 'Conv')

        # get weights from file
        weights = self.statebuffer.circbuf_weights.load_data(op_list.conv_op)
        weight_cols_per_wave = min(op_list.conv_op.M, PEArray.NUM_COLS)
        ifmap_cols_per_wave = min(op_list.conv_op.M, PEArray.NUM_COLS)

        # load bias values
        bias = []
        if (op_list.has_biasadd):
            for j in op_list.biasadd_op.prev:
                if (j.data['layer_type'] == "Const"): # assert sizes can be flattened
                    bias_temp = self.statebuffer.circbuf_bias.load_data(j)
                    bias = bias_temp.flatten()

        # initial psum bank is 0
        op_list.conv_op.set_psum_bank(0)
        # start tensor computation by clearing psum bank
        psum_add = False                               

        # for ResAdd, retrieve the saved result file for one of the completed legs
        if (op_list.has_resadd):
            for j in op_list.resadd_op.prev:
                if j.data['layer_name'] in self.statebuffer.saved_result_files:
                    self.statebuffer.circbuf_scratch.load_file(self.statebuffer.saved_result_files[j.data['layer_name']], op_list.conv_op.ofmap_full_tiley_sz)
                    break

        # save result to create a scratch space (in DRAM), then use circular buffer load to populate params
        if (op_list.has_pool):
            result = np.zeros((op_list.pool_op.N, op_list.pool_op.M, op_list.pool_op.E, op_list.pool_op.F), dtype=inputs.dtype)
        else:     
            result = np.zeros((op_list.conv_op.N, op_list.conv_op.M, op_list.conv_op.E, op_list.conv_op.F), dtype=inputs.dtype)
        np.save(result_file, result)
        self.statebuffer.circbuf_scratch.layer_type = "Output"
        self.statebuffer.circbuf_scratch.layer_name = op_list[-1].data['layer_name']
        # only clear the scratch buffer if there's no ResAdd input there
        if (self.statebuffer.circbuf_scratch.dram_data_file == None):                    
            if (op_list.has_pool):
                self.statebuffer.circbuf_scratch.load_file(result_file, op_list.pool_op.ofmap_full_tiley_sz)
            else:                
                self.statebuffer.circbuf_scratch.load_file(result_file, op_list.conv_op.ofmap_full_tiley_sz)

        # wave loop ordering scheme: nmhwcRS
        for n_id in range(op_list.conv_op.n):
            for m_id in range(op_list.conv_op.m):
                for h_id in range(op_list.conv_op.h):
                    for w_id in range(op_list.conv_op.w):
                        tile_id = TileID(n_id, m_id, h_id, w_id)
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
                        self.pearray.trig_tile_done(tile_id)
                        # compute ofmap tile information (tile startx, starty, height, width)
                        op_list.conv_op.compute_ofmap_tile_info(tile_id)
                        # free portion of requested data (but retain data such that we can still read it)
                        self.statebuffer.circbuf_ifmaps.free_data_region(op_list.conv_op.ifmap_tile_lower_addr, op_list.conv_op.ifmap_tile_upper_addr)
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
                                result[n_id, 
                                        j, 
                                        output_params_op.tile_y_start : output_params_op.tile_y_start + output_params_op.tile_height, 
                                        output_params_op.tile_x_start : output_params_op.tile_x_start + output_params_op.tile_width]\
                                    = result_tile[0:output_params_op.tile_height, 0:output_params_op.tile_width]
                        # for scheduling, map resulting tile into portion of atom that is itself mapped to a portion in DRAM (file)
                        dram_output_waveops = self.statebuffer.circbuf_scratch.write_data_region(wave_id, output_params_op.ofmap_tile_lower_addr, output_params_op.ofmap_tile_upper_addr)
                        self.waveop_stream.add_outputs(dram_output_waveops)
                        # Advance to new bank (ping-pong between 0 and 1) for PEArray, while the old bank is being processed by other engines
                        op_list.conv_op.set_psum_bank((op_list.conv_op.get_psum_bank()+1)%2)
                        psum_add = False

        # save layer results to file, for retrieval by next layer                        
        np.save(result_file, result)
        self.statebuffer.saved_result_files[op_list[-1].data['layer_name']] = result_file

        # print circular buffer stats
        self.statebuffer.print_stats()

        # reset scratch buffer for now (TODO: keep some atoms for next layer)
        self.statebuffer.reset_all()
        return result                    

# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kgraph", help="K-graph Json file to read")
    parser.add_argument("--wavegraph", help="Wave-graph Json file to write")
    parser.add_argument("--dot", help="Dot file to write")
    parser.add_argument("--debug", default=DEBUG_LEVEL_DEFAULT, help="Debug level")
    args = parser.parse_args()

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
    last_result_file = None
    num_mismatches = 0
    while (not kgraph.walk_ended()):
        op_list = kgraph.get_fused_ops()
        if (result_file != None):
            last_result_file = result_file
        result_file = "save_" + op_list[-1].data['layer_name'].replace("/", "__") + ".npy"

        # Check init op
        if (re.search(r"Input", op_list[0].data['layer_type'])):
            inputs = tpb.statebuffer.circbuf_ifmaps.load_data(op_list[0])
            tpb.statebuffer.saved_result_files[op_list[0].data['layer_name']] = op_list[0].data['ref_file']
            results = inputs
        # Check conv fused op
        # TODO: add matrix multiply
        elif (re.search(r"Conv", op_list[0].data['layer_type'])):
            if (tpb.statebuffer.circbuf_ifmaps.dram_data_file == None):                    
                tpb.statebuffer.circbuf_ifmaps.layer_name = op_list[0].data['layer_name']
                tpb.statebuffer.circbuf_ifmaps.layer_type = op_list[0].data['layer_type']
                for j in op_list[0].prev:
                    if j.data['layer_name'] in tpb.statebuffer.saved_result_files:
                        tpb.statebuffer.circbuf_ifmaps.load_file(tpb.statebuffer.saved_result_files[j.data['layer_name']])
                        break
            if (tpb.statebuffer.circbuf_ifmaps.dram_data_file == None):                    
                print("ERROR: ifmaps are not loaded for layer %s"%op_list[0].data['layer_name'])
                exit(-1)
            # TODO: add selecting among pre-derived looping schemes
            results = tpb.execute_conv_ops(results, result_file)

        elif (re.search(r"MatMult", op_list[0].data['layer_type'])):
            print("ERROR: MatMult operation is unimplemented")
            exit(-1)
        elif (re.search(r".*Pool", op_list[0].data['layer_type'])):
            print("ERROR: Pool (unfused) operation is unimplemented")
            exit(-1)
        else:        
            print("ERROR: Unrecognized first operation %s"%op_list[0].data['layer_type'])
            exit(-1)

        # Check results against pre-computed results            
        if 'ref_file' in op_list[-1].data:
            outputs = np.load(op_list[-1].data['ref_file'])
            diff = results - outputs
            if (args.debug > 2): print("\nInput IFMAPS:\n", inputs)
            if (args.debug > 1): print("\nComputed OFMAPS:\n", results)
            if (args.debug > 1): print("\nExpected OFMAPS:\n", outputs)
            if (args.debug > 1): print("\nDiffed   OFMAPS:\n", diff)
            if (not np.allclose(results, outputs, 1/100, 1e-6)):
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
                s = re.sub(r'\s+(\d+,)\n', r' \1', s, flags=re.S)
                s = re.sub(r',\s+(\d+)\n\s+\]', r', \1 ]', s, flags=re.S)
                s = re.sub(r'\s+(\[ \d+, \d+ \],)\n', r' \1', s, flags=re.S)
                s = re.sub(r',\s+(\[ \d+, \d+ \])\n\s+\]', r', \1 ]', s, flags=re.S)
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
