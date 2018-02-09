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
        self.psum_buf = np.zeros((self.PSUM_NUM_BANKS, self.MAX_WAVE_SIZE, self.NUM_COLS))
    def trig_tile_done(self, tile_id):
        if (args.debug > 2): print("Tile done %s"%tile_id.id_string())
        pass
    def extract_psum (self, psum_bank, start_entry, num_entries):
        assert(start_entry < self.MAX_WAVE_SIZE)
        #assert((start_entry+num_entries) < self.MAX_WAVE_SIZE)
        return self.psum_buf[psum_bank, start_entry:start_entry+num_entries, :]
    def write_psum (self, psum_bank, start_entry, num_entries, op_result):
        assert(start_entry < self.MAX_WAVE_SIZE)
        #assert((start_entry+num_entries) < self.MAX_WAVE_SIZE)
        self.psum_buf[psum_bank, start_entry:start_entry+num_entries, :] = op_result
    # Do wave fp16->fp32 matrix-multiply        
    #   packed_ifmaps: must be 256x128 matrix, float16
    #   packet_weights: must be 128x64 matrix, float16
    #   psum_bank: the PSUM bank number to write result to
    #   psum_add: if True, add to PSUM value in buffer; if False, replace with new value
    def wave_fp16_mm(self, packed_ifmaps, packet_weights, psum_bank, psum_add):
        assert (packed_ifmaps.shape == (self.MAX_WAVE_SIZE, self.NUM_ROWS))
        assert (packet_weights.shape == (self.NUM_ROWS, self.NUM_COLS))
        assert (psum_bank < self.PSUM_NUM_BANKS)
        self.matmul_result = np.matmul(
                packed_ifmaps.astype(np.float32), 
                packet_weights.astype(np.float32))
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
            return np.maximum(np.zeros(in_array.shape), in_array)
        elif (type == 'Sigmoid'):
            return 1/(1 + math.exp(-in_array))
    def relu(self, in_array):
        return np.maximum(np.zeros(in_array.shape), in_array)

##################################################################################
# State buffer memory manager
class StateBuffer:
    SB_NUM_PARTITIONS = 128
    SB_PARTITION_SZ = 96*1024 # 96KB per partition
    SB_ATOM_SZ = 1024 # can be down to 256B for maximum DMA efficiency
    SB_NUM_1K_ATOMS = SB_PARTITION_SZ/SB_ATOM_SZ
    def __init__(self):
        self.data = np.zeros((self.SB_NUM_PARTITIONS, self.SB_PARTITION_SZ))
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
        self.allocated = np.zeros(self.capacity)
        self.dram_data_file = None
        self.dram_data = None
        self.dram_data_len = 0
        self.layer_name = ""
        self.layer_type = "Output"
        self.addr2atom = {}

    def load_data(self, waveop):      # use waveop instead of waveop 
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

    def load_file(self, file):      # use waveop instead of waveop 
        self.dram_data_file = file
        self.dram_data = np.load(self.dram_data_file)
        self.item_sz = self.dram_data.dtype.itemsize   
        self.dram_data_len = self.dram_data.size * self.item_sz
        # TODO: come up with a better formula for atom_data_sz to take care of all cases
        # Constraints for atom_data_sz: 
        #   * less than 1KB
        #   * multiple of width
        #   * different IFMAPs in one batch will be in different atoms (for now)
        #   * different IFMAPs folds will be in different atoms (for now)
        # TODO: refactor the following function since it is used in multiple places
        if (self.layer_type == 'Input' or self.layer_type == 'Const' or self.layer_type == 'Output'):
            N, C, H, W = [i*self.item_sz for i in self.dram_data.shape]
            if (H*W <= self.atom_sz):
                num_dim3 = self.atom_sz // (H*W)
                self.atom_data_sz = (H*W) * min(C, num_dim3)
            elif (W <= self.atom_sz):
                num_dim3 = self.atom_sz // W
                self.atom_data_sz = W * min(H, num_dim3)
            else:
                self.atom_data_sz = self.atom_sz
        else:            
            C, R, S, M = [i*self.item_sz for i in self.dram_data.shape]
            if (M <= self.atom_sz):
                # find the largest multiple of M that satisfies constraints
                num_dim3 = self.atom_sz // M
                if (R*S < num_dim3):
                    num_dim3 = R*S
                elif (S < num_dim3):
                    num_dim3 = S
                self.atom_data_sz = M * num_dim3
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
        self.allocated[self.current_atom_id - self.start] = 1
        if (args.debug > 2): print ("%s: Added atom_id %d for layer %s"%(self.circbuf_type, self.current_atom_id, self.layer_name))
        self.tail_pointer += 1
        if (self.tail_pointer == self.start + self.capacity):
            self.tail_pointer = self.start
        self.count += 1
        if (self.count > self.max_count):
            self.max_count = self.count
        return self.current_atom_id            

    def free_atom(self, atom_id):   
        if (self.allocated[atom_id - self.start] == 1):
            self.allocated[atom_id - self.start] = 0
            self.count -= 1
            if (args.debug > 2): print ("%s: Freed atom_id %d for layer %s"%(self.circbuf_type, atom_id, self.layer_name))
        #else:
        #    print ("ERROR %s: cannot free atom ID %d since it is unallocated for layer %s!"%(self.circbuf_type, atom_id, self.layer_name))
        #    return -1
        # garbage collection: advance head pointer until it sees allocated atom
        if (self.allocated[self.head_pointer - self.start] == 0):
            self.head_pointer += 1            
            if (self.head_pointer == self.start + self.capacity):
                self.head_pointer = self.start

    def print_stats(self):
        print("STATS circular buffer type %s layer %s: capacity %d atom size %d atom data size %d atom count %d max count %d DRAM data length %d"%(self.circbuf_type, self.layer_name, self.capacity, self.atom_sz, self.atom_data_sz, self.count, self.max_count, self.dram_data_len))

##################################################################################
# Neural network node, containing data read from JSON
class KNode:
    def __init__(self, data):
        self.prev = []
        self.next = []
        self.data = data
        self.psum_bank_dst = 0
    def add_prev(self, prev_node):
        self.prev.append(prev_node)
    def add_next(self, next_node):
        self.next.append(next_node)

    # set/get dest PSUM bank
    def set_psum_bank(self, dest):
        self.psum_bank_dst = dest
    def get_psum_bank(self):
        return self.psum_bank_dst

    # Compute conv looping and tiling params
    def compute_conv_loops(self):
        assert (self.prev[0] != None)
        input_layer = self.prev[0].data
        self.N, self.C, self.H, self.W = input_layer['ofmap_shape']
        layer_info = self.data
        assert (layer_info['kernel_format'] == 'CRSM')
        assert (layer_info['ofmap_format'] == 'NCHW')
        self.C, self.R, self.S, self.M = layer_info['kernel_shape']
        self.N, self.M, self.E, self.F = layer_info['ofmap_shape']
        self.pad_north, self.pad_south = layer_info['padding'][2]
        self.pad_west, self.pad_east = layer_info['padding'][3]
        self.stride_y = layer_info['stride'][2]
        self.stride_x = layer_info['stride'][3]
        # compute the IFMAP folds
        self.c = ceildiv(self.C, PEArray.NUM_ROWS)
        # compute the OFMAP folds
        self.m = ceildiv(self.M, PEArray.NUM_COLS)
        # computing the input map tiling       
        #self.num_tiles = 1
        self.HW = self.H * self.W
        self.EF = self.E * self.F
        self.h, self.w, self.e, self.f = 1, 1, 1, 1
        self.n, self.Tn = 1, 1
        # TODO: determine tile size and aspect ratio taking into account pooling size
        #self.ofmap_tiley_sz = min(self.E, int(math.sqrt(PEArray.MAX_WAVE_SIZE)))
        #self.ofmap_tilex_sz = min(self.F, int(math.sqrt(PEArray.MAX_WAVE_SIZE)))
        # per kaena-85, use noodle shapes for tiles (TODO: if pooling, then need to make num rows = pooling rows)
        self.ofmap_tilex_sz = min(self.F, PEArray.MAX_WAVE_SIZE)
        self.ofmap_tiley_sz = min(self.E, PEArray.MAX_WAVE_SIZE // self.ofmap_tilex_sz)
        self.ofmap_tile_sz = self.ofmap_tilex_sz * self.ofmap_tiley_sz
        if (self.EF >= PEArray.MAX_WAVE_SIZE):
            # for now, send in noodles that span width of IFMAP, which has implications for pooling
            #self.num_tiles = ceildiv(H*W, PEArray.MAX_WAVE_SIZE)
            # for now, send in 16x16 tiles, which has implications for pooling
            # for now, use h/w for both IFMAP and OFMAP
            self.e = ceildiv(self.E, self.ofmap_tiley_sz)
            self.f = ceildiv(self.F, self.ofmap_tilex_sz)
        else:
            self.Tn = PEArray.MAX_WAVE_SIZE // self.EF
            if (self.Tn > self.N):
                self.Tn = self.N
            self.n = ceildiv(self.N, self.Tn)
        self.h = self.e
        self.w = self.f
        self.ifmap_tiley_sz = self.ofmap_tiley_sz * self.stride_y
        self.ifmap_tilex_sz = self.ofmap_tilex_sz * self.stride_x
        self.data['batching_in_wave'] = self.Tn
        self.data['batch_fold_count'] = self.n
        self.data['ofmap_fold_count'] = self.m
        self.data['ifmap_fold_count'] = self.c
        self.data['width_fold_count'] = self.w
        self.data['height_fold_count'] = self.h
        print("Conv params: n=%d, m=%d, h=%d, w=%d, c=%d, R=%d, S=%d, Tn=%d, stride_x=%d, stride_y=%d, ofmap_tilex_sz=%d, ofmap_tiley_sz=%d"
                %(self.n, self.m, self.h, self.w, self.c, self.R, self.S, self.Tn, self.stride_x, self.stride_y, self.ofmap_tilex_sz, self.ofmap_tiley_sz))

    # Compute pooling params
    def compute_pooling_params(self):
        assert (self.prev[0] != None)
        input_layer = self.prev[0].data
        self.N, self.C, self.H, self.W = input_layer['ofmap_shape']
        layer_info = self.data
        self.N, self.M, self.E, self.F = layer_info['ofmap_shape']
        self.pad_north, self.pad_south = layer_info['padding'][2]
        self.pad_west, self.pad_east = layer_info['padding'][3]
        # are the dimensions from layer info correct?
        self.pool_window_y = layer_info['kernel_shape'][2]
        self.pool_window_x = layer_info['kernel_shape'][3]
        self.stride_y = layer_info['stride'][2]
        self.stride_x = layer_info['stride'][3]
        # computing the input map tiling       
        #self.num_tiles = 1
        self.HW = self.H * self.W
        self.EF = self.E * self.F
        # TODO: determine tile size and aspect ratio taking into account pooling size
        #self.ofmap_tiley_sz = min(self.E, int(math.sqrt(PEArray.MAX_WAVE_SIZE)))
        #self.ofmap_tilex_sz = min(self.F, int(math.sqrt(PEArray.MAX_WAVE_SIZE)))
        # per kaena-85, use noodle shapes for tiles (TODO: if pooling, then need to make num rows = pooling rows)
        self.ofmap_tilex_sz = min(self.F, PEArray.MAX_WAVE_SIZE)
        self.ofmap_tiley_sz = min(self.E, PEArray.MAX_WAVE_SIZE // self.ofmap_tilex_sz)
        # For pooling using PSUM, max tile size must be a multiple of pooling window
        self.ofmap_tiley_sz = (self.ofmap_tiley_sz // self.F) * self.F
        self.ofmap_tile_sz = self.ofmap_tilex_sz * self.ofmap_tiley_sz
        print("Pooling params: ofmap_tilex_sz=%d, ofmap_tiley_sz=%d"%(self.ofmap_tilex_sz, self.ofmap_tiley_sz))

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
        pe_row_start = wave_id.c_id * PEArray.NUM_ROWS
        pe_row_stop = min(self.C, pe_row_start + PEArray.NUM_ROWS)
        assert(pe_row_start < pe_row_stop)
        for row in range(pe_row_start, pe_row_stop):
            #out_array[:,row] = self.pack_wave_ifmap(ifmaps[:, wave_id.c_id * PEArray.NUM_ROWS + row], wave_id)
            ifmap = ifmaps[:, row]
            pe_row_offset = row - pe_row_start
            for i in range(self.Tn):
                for x in range(self.ofmap_tilex_sz):
                    for y in range(self.ofmap_tiley_sz):
                        ifmap_tilex = wave_id.w_id * self.ifmap_tilex_sz + (x * self.stride_x) + wave_id.s_id - self.pad_west
                        ifmap_tiley = wave_id.h_id * self.ifmap_tiley_sz + (y * self.stride_y) + wave_id.r_id - self.pad_north
                        ifmap_addr = i * self.ofmap_tile_sz + y * self.ofmap_tilex_sz + x
                        #print("x %d y %d ifmap_tilex %d ifmap_tiley %d"%(x, y, ifmap_tilex, ifmap_tiley))                                    
                        if (ifmap_tilex < 0 or ifmap_tilex >= self.W):
                            out_array[ifmap_addr, pe_row_offset] = 0
                        elif (ifmap_tiley < 0 or ifmap_tiley >= self.H):
                            out_array[ifmap_addr, pe_row_offset] = 0
                        else:
                            out_array[ifmap_addr, pe_row_offset] = ifmap[(wave_id.n_id * self.Tn) + i, ifmap_tiley, ifmap_tilex]
                            # Check bounds of actual pixels within the original ifmaps for the first ifmap (which should reside in first SB partition)
                            # TODO: check how N/C are arrange in memory; batching within waves may cause different atoms to be accessed by same wave
                            if (row == pe_row_start):                                
                                self.ifmap_wave_upper_addr = int(np.ravel_multi_index(((wave_id.n_id * self.Tn) + i, row, ifmap_tiley, ifmap_tilex),
                                                                    dims=ifmaps.shape) * ifmaps.dtype.itemsize)
                                if (self.ifmap_wave_lower_addr < 0):
                                    self.ifmap_wave_lower_addr = self.ifmap_wave_upper_addr
                        # TODO: for optimization, don't send padding zeros by using the following clipped version
                        #ifmap_tilex_clip = max(0, min(ifmap_tilex, self.W-1))
                        #ifmap_tiley_clip = max(0, min(ifmap_tiley, self.H-1))
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
        self.has_pool = False
        self.has_resadd = False
        self.has_conv = False
        self.has_matmul = False
        self.pool_op = None
        self.resadd_op = None
        self.conv_op = None
        self.matmul_op = None
        self.count = 0

    # Add operation to list of fused operations.
    # Returns True if successful; False if cannot add (i.e. Pool cannot be fused)
    def add(self, op):
        if (op.data['layer_type'] == 'AvgPool' or op.data['layer_type'] == 'MaxPool'):
            op.compute_pooling_params()
            # If not first op, pool cannot be fused with previous op if stride != pooling window
            if (len(self) != 0 and 
                    (op.stride_x != op.pool_window_x or op.stride_y != op.pool_window_y)):
                return False
            else:
                self.pool_op = op
        elif (op.data['layer_type'] == 'ResAdd'):
            self.resadd_op = op
        elif (op.data['layer_type'] == 'MatMul'):
            self.matmul_op = op
        elif (op.data['layer_type'] == 'Conv'):
            if (self.count != 0):
                print("ERROR: Conv must be the first op in the fused operation")
                self.show()
                exit(-1)
            op.compute_conv_loops()
            self.conv_op = op
        self.count += 1
        self.append(op)
        return True            

    def show(self):
        print("DBG: fused_ops collected: ",)
        for i in self:
            print("    ", i.data["layer_type"],":",i.data["layer_name"], )

    # generate MatMul waveop and add it to waveop stream
    def gen_matmul_waveop(self, tpb, wave_id, psum_add):
        matmul_waveop = {
              'previous_waveops'        : [],   # to be added later
              'waveop_type'             : 'MatMul',
              'waveop_name'             : self.conv_op.data['layer_name']+"/MatMul_"+wave_id.id_string(),
              'layer_name'              : self.conv_op.data['layer_name'],
              'weights_atom_id'         : tpb.statebuffer.circbuf_weights.current_atom_id,
              'ifmaps_atom_id'          : tpb.statebuffer.circbuf_ifmaps.current_atom_id,
              'weights_offset_in_atom'  : self.conv_op.weight_wave_lower_addr % tpb.statebuffer.circbuf_weights.atom_data_sz,  # TODO: -1 means don't load new weights
              'ifmaps_offset_in_atom'   : self.conv_op.ifmap_wave_lower_addr % tpb.statebuffer.circbuf_ifmaps.atom_data_sz,
              'wave_id_format'          : wave_id.format,
              'wave_id'                 : wave_id.show(),
              'start'                   : not(psum_add),
              'psum_bank_id'            : self.conv_op.psum_bank_dst,
              'psum_bank_offset'        : 0,    # TODO: compute and put the correct value here
              'ifmap_count'             : 1,    # TODO: compute and put the correct value here
              'ifmap_tile_width'        : 1,    # TODO: compute and put the correct value here
              'ifmap_tile_height'       : 1,    # TODO: compute and put the correct value here
              'ofmap_count'             : 1,    # TODO: compute and put the correct value here
              'ofmap_tile_width'        : 1,    # TODO: compute and put the correct value here
              'ofmap_tile_height'       : 1,    # TODO: compute and put the correct value here
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
        # process layers
        layers = kgraph_json["layers"]
        num_layers = len(layers)
        if (num_layers >= 1):
            for l in layers:
                new_node = KNode(l)
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
                    new_node = KNode(l)
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
        weights = self.statebuffer.circbuf_weights.load_data(op_list[0])
        weight_cols_per_wave = min(op_list[0].M, PEArray.NUM_COLS)
        ifmap_cols_per_wave = min(op_list[0].M, PEArray.NUM_COLS)

        # load bias values
        bias = []
        for i in range(1, len(op_list)):
            layer_type = op_list[i].data['layer_type'] 
            if (layer_type == 'BiasAdd'):
                for j in op_list[i].prev:
                    if (j.data['layer_type'] == "Const"): # assert sizes can be flattened
                        bias_temp = self.statebuffer.circbuf_bias.load_data(j)
                        bias = bias_temp.flatten()

        # initial values
        op_list[0].set_psum_bank(0)
        psum_add = False                               
        result = np.zeros((op_list[0].N, op_list[0].M, op_list[0].E, op_list[0].F), dtype=inputs.dtype)
        # save result to create a scratch space (in DRAM), then use circular buffer load to populate params
        np.save(result_file, result)
        self.statebuffer.circbuf_scratch.layer_type = "Output"
        self.statebuffer.circbuf_scratch.layer_name = op_list[-1].data['layer_name']
        # for ResAdd, retrieve the saved result file for one of the completed legs
        for i in op_list:
            if (i.data['layer_type'] == 'ResAdd'):
                for j in i.prev:
                    if j.data['layer_name'] in self.statebuffer.saved_result_files:
                        self.statebuffer.circbuf_scratch.load_file(self.statebuffer.saved_result_files[j.data['layer_name']])
                        break
        if (self.statebuffer.circbuf_scratch.dram_data_file == None):                    
            self.statebuffer.circbuf_scratch.load_file(result_file)
        # wave loop ordering scheme: nmhwcRS
        for n_id in range(op_list[0].n):
            for m_id in range(op_list[0].m):
                for h_id in range(op_list[0].h):
                    for w_id in range(op_list[0].w):
                        tile_id = TileID(n_id, m_id, h_id, w_id)
                        # loops for constructing a tile
                        for c_id in range(op_list[0].c):
                            for r_id in range(op_list[0].R):
                                for s_id in range(op_list[0].S):
                                    wave_id = WaveID(n_id, m_id, h_id, w_id, c_id, r_id, s_id)
                                    if (args.debug > 2): print (wave_id.show())
                                    # execute PEArray matrix multiply, and add to PSUM after first wave
                                    if (op_list.execute_matmul_waveop(self, wave_id, inputs, weights, psum_add)):
                                        psum_add = True
                        # tile is done                                    
                        # TODO: refactor the following code into a tile object
                        self.pearray.trig_tile_done(tile_id)
                        tile_x_start = wave_id.w_id * op_list[0].ofmap_tilex_sz
                        tile_y_start = wave_id.h_id * op_list[0].ofmap_tiley_sz
                        tile_height = op_list[0].ofmap_tiley_sz
                        tile_width = op_list[0].ofmap_tilex_sz
                        if ((wave_id.h_id+1) * op_list[0].ofmap_tiley_sz > op_list[0].E):
                            tile_height = op_list[0].E - tile_y_start
                        if ((wave_id.w_id+1) * op_list[0].ofmap_tilex_sz > op_list[0].F):
                            tile_width = op_list[0].F - tile_x_start
                        tile_size = tile_height * tile_width

                        self.ifmap_tile_lower_addr = int(np.ravel_multi_index((wave_id.n_id * op_list[0].Tn, 0, tile_y_start * op_list[0].stride_y, tile_x_start * op_list[0].stride_x),
                                                        dims=inputs.shape) * inputs.dtype.itemsize)
                        self.ifmap_tile_upper_addr = int(np.ravel_multi_index(
                                                            (op_list[0].N - 1, 
                                                            (op_list[0].c - 1) * PEArray.NUM_ROWS,
                                                            tile_y_start * op_list[0].stride_y + tile_height * op_list[0].stride_y - 1, 
                                                            tile_x_start * op_list[0].stride_x + tile_width * op_list[0].stride_x - 1),
                                                        dims=inputs.shape) * inputs.dtype.itemsize)
                        # free portion of requested data
                        # TODO: make sure that this doesn't free needed data
                        self.statebuffer.circbuf_ifmaps.free_data_region(self.ifmap_tile_lower_addr, self.ifmap_tile_upper_addr)
                        # go through the remaining operations
                        psum_bank_src = op_list[0].get_psum_bank()
                        op_result = self.pearray.extract_psum(psum_bank_src, 0, op_list[0].ofmap_tile_sz)
                        op_list_iter = iter(range(1, len(op_list)))
                        for i in op_list_iter: #range(1, len(op_list)):
                            layer_type = op_list[i].data['layer_type'] 
                            if (re.search(r"Relu|Tanh|Sigmoid", layer_type)):
                                self.activate.wait_tile_done(tile_id)
                                op_result = self.activate.relu(op_result)
                                psum_bank_dst = 2
                                self.gen_act_waveop_inline(None, op_list[i], tile_id, psum_bank_src, psum_bank_dst, [], 0)
                                if (i != len(op_list)-1):
                                    self.pearray.write_psum(psum_bank_dst, 0, op_list[0].ofmap_tile_sz, op_result)
                            elif (layer_type == 'BiasAdd'):
                                self.activate.wait_tile_done(tile_id)
                                bias_start = m_id*PEArray.NUM_COLS
                                bias_end = min(bias_start + PEArray.NUM_COLS, op_list[0].M)
                                bias_extracted = np.zeros(PEArray.NUM_COLS)
                                bias_extracted[0 : bias_end - bias_start] = bias[bias_start : bias_end]
                                dram_bias_waveops = self.statebuffer.circbuf_bias.read_data_region(wave_id, bias_start, bias_end)
                                op_result = self.activate.biasadd(op_result, bias_extracted)
                                psum_bank_dst = 2
                                if (i+1 < len(op_list) and re.search(r"Relu|Tanh|Sigmoid", op_list[i+1].data['layer_type'])):
                                    op_result = self.activate.act(op_list[i+1].data['layer_type'], op_result)
                                    self.gen_act_waveop_inline(op_list[i], op_list[i+1], tile_id, psum_bank_src, psum_bank_dst, dram_bias_waveops, bias_start)
                                    next(op_list_iter)
                                else:                                    
                                    self.gen_act_waveop_inline(op_list[i], None, tile_id, psum_bank_src, psum_bank_dst, dram_bias_waveops, bias_start)
                                if (i != len(op_list)-1):
                                    self.pearray.write_psum(psum_bank_dst, 0, op_list[0].ofmap_tile_sz, op_result)
                            elif (layer_type == 'ResAdd'):
                                self.pool.wait_tile_done(tile_id)
                                dram_resadd_waveop = self.statebuffer.circbuf_scratch.read_data_region(wave_id, self.ofmap_tile_lower_addr, self.ofmap_tile_upper_addr)
                                residue_tile = np.zeros((op_list[0].ofmap_tile_sz, PEArray.NUM_COLS))
                                for j in range(PEArray.NUM_COLS):
                                    M_idx = wave_id.m_id * PEArray.NUM_COLS + j
                                    if (M_idx >= op_list[0].M):
                                        break
                                    else:
                                        residue_tile_ifmap = self.statebuffer.circbuf_scratch.dram_data[n_id, j, tile_y_start : tile_y_start + tile_height, tile_x_start : tile_x_start + tile_width]
                                        residue_tile[0:tile_height*tile_width,j] = residue_tile_ifmap.flatten()
                                op_result = self.pool.resadd(op_result, residue_tile)
                                psum_bank_dst = 3
                                self.gen_resadd_waveop_inline(op_list[i], tile_id, psum_bank_src, psum_bank_dst, self.ofmap_tile_lower_addr)
                                if (i != len(op_list)-1):
                                    self.pearray.write_psum(psum_bank_dst, 0, op_list[0].ofmap_tile_sz, op_result)
                            elif (layer_type == 'AvgPool'):
                                self.activate.wait_tile_done(tile_id)
                                # use the dimension E for pool_window_size since we do not have pool_window_size readily available
                                op_result = self.pool.avg(op_result, op_list[0].stride_x, op_list[0].E)
                                # need to adjust tile size since pooling operation will shrink the tile dimension
                                print ("Adjust ofmap size from pooling operation")
                                pool_ofmap_tilex_sz = op_list[0].ofmap_tilex_sz // op_list[0].E
                                pool_ofmap_tiley_sz = op_list[0].ofmap_tiley_sz // op_list[0].E
                                pool_ofmap_tile_sz = pool_ofmap_tilex_sz * pool_ofmap_tiley_sz
                                # replicated code - techinical debt
                                pool_tile_x_start = wave_id.w_id * pool_ofmap_tilex_sz
                                pool_tile_y_start = wave_id.h_id * pool_ofmap_tiley_sz
                                pool_tile_height = pool_ofmap_tiley_sz
                                pool_tile_width = pool_ofmap_tilex_sz
                                if ((wave_id.h_id+1) * pool_ofmap_tiley_sz > op_list[0].E):
                                    pool_tile_height = op_list[0].E - pool_tile_y_start
                                if ((wave_id.w_id+1) * op_list[0].ofmap_tilex_sz > op_list[0].F):
                                    pool_tile_width = op_list[0].F - pool_tile_x_start
                                pool_tile_size = pool_tile_height * pool_tile_width
                                psum_bank_dst = 3
                                # TODO: generate AvgPool instruction inline
                                if (i != len(op_list)-1):
                                    self.pearray.write_psum(psum_bank_dst, 0, pool_ofmap_tile_sz, op_result)
                            else:
                                print ("ERROR: %s is currently not yet implemented"%layer_type)
                                exit(-1)
                        # if operation is the last one, dump current result into a portion of final result
                        for j in range(PEArray.NUM_COLS):
                            M_idx = wave_id.m_id * PEArray.NUM_COLS + j
                            if (M_idx >= op_list[0].M):
                                break
                            else:
                                # For now, multiply zeros, and at the ofmap, extract tile with zeros, then clip
                                # take advantage of the fact that pool op is usually the last one in fused ops
                                layer_type = op_list[-1].data['layer_type'] 
                                if (layer_type != 'AvgPool'):
                                    result_tile = (op_result[0 : op_list[0].ofmap_tile_sz, j]).reshape((op_list[0].ofmap_tiley_sz, op_list[0].ofmap_tilex_sz))
                                    result[n_id, j, tile_y_start : tile_y_start + tile_height, tile_x_start : tile_x_start + tile_width]\
                                        = result_tile[0:tile_height, 0:tile_width]
                                else:
                                    result_tile = (op_result[0 : pool_ofmap_tile_sz, j]).reshape((pool_ofmap_tiley_sz, pool_ofmap_tilex_sz))
                                    result[n_id, j, pool_tile_y_start : pool_tile_y_start + pool_tile_height, pool_tile_x_start : pool_tile_x_start + pool_tile_width]\
                                        = result_tile[0:pool_tile_height, 0:pool_tile_width]
                                # print(wave_id.show(),"\n", result_tile)                                        
                                #print(wave_id.show(),"\n", result_tile)                                       

                        # for scheduling, dump resulting tile into atom that is mapped to DRAM (file)
                        self.ofmap_tile_lower_addr = int(np.ravel_multi_index((tile_id.n_id * op_list[0].Tn, 0, tile_y_start, tile_x_start),
                                                        dims=result.shape) * inputs.dtype.itemsize)
                        self.ofmap_tile_upper_addr = int(np.ravel_multi_index(
                                                            (op_list[0].N - 1, 
                                                            0,
                                                            tile_y_start + tile_height - 1, 
                                                            tile_x_start + tile_width - 1),
                                                        dims=result.shape) * inputs.dtype.itemsize)
                        dram_output_waveops = self.statebuffer.circbuf_scratch.write_data_region(wave_id, self.ofmap_tile_lower_addr, self.ofmap_tile_upper_addr)
                        self.waveop_stream.add_outputs(dram_output_waveops)

                        # Advance to new bank (ping-pong between 0 and 1) for PEArray, while the old bank is being processed by other engines
                        op_list[0].set_psum_bank((op_list[0].get_psum_bank()+1)%2)
                        psum_add = False
        # save layer results to file, for retrieval by next layer                        
        np.save(result_file, result)
        self.statebuffer.saved_result_files[op_list[-1].data['layer_name']] = result_file

        # print circular buffer stats
        self.statebuffer.print_stats()

        # reset scratch buffer for now (TODO: keep some atoms for next layer)
        self.statebuffer.circbuf_scratch.reset()
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
                for j in op_list[0].prev:
                    if j.data['layer_name'] in tpb.statebuffer.saved_result_files:
                        tpb.statebuffer.circbuf_scratch.load_file(tpb.statebuffer.saved_result_files[j.data['layer_name']])
                        break
            if (tpb.statebuffer.circbuf_ifmaps.dram_data_file == None):                    
                print("ERROR: ifmaps are not loaded for layer %s"%op_list[0].data['layer_name'])
                exit(-1)
            if (op_list[0].prev[0].data['ofmap_format'] == 'NCHW'):
                iN,iC,iH,iW = op_list[0].prev[0].data['ofmap_shape']
                print("Input shape",iN,iC,iH,iW)
            else:
                print("ERROR: don't understand ifmap_format %s"%op_list[0].prev[0].data['ofmap_shape'])
                exit(-1)
            conv_layer = op_list[0].data
            convN, convC, convH, convW = conv_layer['ofmap_shape']
            pad_north, pad_south = conv_layer['padding'][2]
            pad_west, pad_east = conv_layer['padding'][3]
            # stride: ignore stride_batch and stride_depth for now; also assume stride_x==stride_y
            stride_x = conv_layer['stride'][2]
            stride_y = conv_layer['stride'][3]
            assert(stride_x == stride_y)
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
            if (not np.allclose(results, outputs, 1/100, 1e-7)):
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
