import json
import os
import math
import re
import numpy as np
import argparse
from graphviz import Digraph

#np.set_printoptions(threshold=np.nan)

#kgraph_file = os.environ['KAENA_PATH'] + "/compiler/tffe/rundir/0-1conv0/trivnet_compiler.json"

def ceildiv(a,b):
    return (a//b) + (a%b != 0)

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

# ID of completed OFMAP tile
class TileID:
    def __init__(self, n_id, m_id, h_id, w_id):
        self.format = "nmhw"
        self.n_id, self.m_id, self.h_id, self.w_id = n_id, m_id, h_id, w_id
    def show(self):
        return [self.n_id, self.m_id, self.h_id, self.w_id]
    def id_string(self):
        return "n%d_m%d_h%d_w%d"%(self.n_id, self.m_id, self.h_id, self.w_id)

# PE Array properties and methods
class PEArray:
    NUM_ROWS = 128
    NUM_COLS = 64
    PSUM_NUM_BANKS = 4
    MAX_WAVE_SIZE = 256
    def __init__(self):
        self.psum_buf = np.zeros((self.PSUM_NUM_BANKS, self.MAX_WAVE_SIZE, self.NUM_COLS))
    def trig_tile_done(self, tile_id):
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

# Pooling properties and methods
class Pool:
    def wait_tile_done(self, tile_id):
        pass
    def avg(self, in_array):
        return in_array
    def max(self, in_array):
        return in_array

# Bias-Add and Activate properties and methods
class BiasAddAct:
    def wait_tile_done(self, tile_id):
        pass
    def biasadd(self, in_array, bias_array):
        return in_array+bias_array
    def relu(self, in_array):
        return np.maximum(np.zeros(in_array.shape), in_array)

# State buffer memory manager
class StateBuffer:
    SB_NUM_PARTITIONS = 128
    SB_PARTITION_SZ = 6*16*1024
    SB_ATOM_SZ = 1024
    SB_NUM_1K_ATOMS = SB_PARTITION_SZ/SB_ATOM_SZ
    def __init__(self):
        self.data = np.zeros((self.SB_NUM_PARTITIONS, self.SB_PARTITION_SZ))
        self.infbuf_weights = CircularBuffer(1024, self.SB_ATOM_SZ, 0)
        self.infbuf_ifmaps = CircularBuffer(1024, self.SB_ATOM_SZ, 24)
        self.infbuf_bias = CircularBuffer(1024, self.SB_ATOM_SZ, 48)
        self.infbuf_scratch = CircularBuffer(1024, self.SB_ATOM_SZ, 72)
    def write(self, partition, address, write_data):
        self.data[partition, address:len(write_data)] = write_data

def addr1kbyte(addr):
    return (addr//1024)

class CircularBuffer:
    # TODO: instead of using initial start atom, determine max capacity during first pass
    # then adjust the start to fit SB
    def __init__(self, capacity, atom_sz, start):
        self.capacity = capacity
        self.atom_sz = atom_sz
        self.item_sz = 2
        self.start = start
        self.reset()

    def reset(self):
        self.head_pointer = 0
        self.tail_pointer = 0
        self.head_byte_ptr = 0
        self.tail_byte_ptr = 0
        self.current_atom_id = 0
        self.count = 0
        self.max_count = 0
        self.valid = np.zeros(self.capacity)
        self.dram_data_file = None
        self.dram_data = None
        self.layer_name = ""
        self.layer_type = ""
        self.instr_list = []
        self.addr2atom = {}

    def load_data(self, node):       
        self.reset()
        if (node.data['layer_type'] == 'Input' or node.data['layer_type'] == 'Const'):
            self.dram_data_file = node.data['ref_file']
        else:            
            self.dram_data_file = node.data['kernel_file']
        self.dram_data = np.load(self.dram_data_file)
        self.item_sz = self.dram_data.dtype.itemsize
        print("Loaded %s, first data is %f, data size is %d bytes"%(self.dram_data_file, self.dram_data[0,0,0,0], self.item_sz)) 
        self.layer_name = node.data['layer_name']
        self.layer_type = node.data['layer_type']
        return self.dram_data

    def gen_dram_instr(self, wave_id, atom_id, offset1k):    
        return {
              'previous_waveops' : [],
              'waveop_type'      : "SBAtomFile",
              'waveop_name'      : self.layer_name+"/SBAtomFile_%d"%offset1k,
              'layer_name'       : self.layer_name,
              'atom_id'          : atom_id,
              'ref_file'         : self.dram_data_file,
              'offset_in_file'   : offset1k*self.atom_sz,
              'length'           : self.atom_sz,
              'ifmaps_replicate' : False,
              'ifmaps_fold_idx'  : wave_id.c_id
            }

    def cache_data(self, wave_id, lower_addr, upper_addr):
        if (self.tail_byte_ptr + self.item_sz < lower_addr):
            print("WARNING: caching byte range %d to %d, but there's a %d > %d gap between tail_byte_ptr %d and lower_addr"%(lower_addr, upper_addr, (lower_addr-self.tail_byte_ptr), self.item_sz, self.tail_byte_ptr))
        dram_instr = None
        lower_addr_1kbyte = lower_addr//self.atom_sz
        upper_addr_1kbyte = upper_addr//self.atom_sz
        if lower_addr_1kbyte not in self.addr2atom:
            self.add_vatom()
            dram_instr = self.gen_dram_instr(wave_id, self.current_atom_id, lower_addr_1kbyte)
            self.addr2atom[lower_addr_1kbyte] = dram_instr
            #assert(lower_addr_1kbyte == upper_addr_1kbyte)
        # the following assume that the lower_addr and upper_addr cannot be in different 1kbyte blocks
        elif upper_addr_1kbyte not in self.addr2atom:
            self.add_vatom()
            dram_instr = self.gen_dram_instr(wave_id, self.current_atom_id, upper_addr_1kbyte)
            self.addr2atom[upper_addr_1kbyte] = dram_instr
        self.tail_byte_ptr = upper_addr
        return dram_instr

    def uncache_data(self, lower_addr, upper_addr):
        if (self.head_byte_ptr + self.item_sz < lower_addr):
            print("WARNING: freeing byte range %d to %d, but there's a %d > %d gap between head_byte_ptr %d and lower_addr"%(lower_addr, upper_addr, (lower_addr-self.head_byte_ptr), self.item_sz, self.head_byte_ptr))
        if ((upper_addr//self.atom_sz) > (self.head_byte_ptr//self.atom_sz)):
            self.free_vatom(self.head_pointer)
        self.head_byte_ptr = upper_addr

    def add_vatom(self):
        if (self.count == self.capacity):
            print ("ADD VATOM ERROR: no more space!")
            return -1
        self.current_atom_id = self.start + self.tail_pointer
        if (args.debug > 0): print ("Added atom_id %d"%(self.current_atom_id))
        self.tail_pointer += 1
        self.count += 1
        if (self.count > self.max_count):
            self.max_count = self.count

    def free_vatom(self, atom_id):   
        if (self.valid[atom_id] == 1):
            self.valid[atom_id] = 0
            self.count -= 1
            if (args.debug > 0): print ("Freed atom_id %d"%(atom_id))
        else:
            print ("FREE VATOM ERROR: atom ID %d is already freed!"%atom_id)
            return -1
        self.head_pointer += 1            

# Neural network node, containing data read from JSON
class KNode:
    def __init__(self, data):
        self.prev = []
        self.next = []
        self.data = data
    def add_prev(self, prev_node):
        self.prev.append(prev_node)
    def add_next(self, next_node):
        self.next.append(next_node)

# RegExs to determine whether next node is fusable or not
next_is_fusable = {
        'Conv'   : "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
        'MatMul' : "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
        'BiasAdd': "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
        'Add'    : "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
        'ResAdd' : "BiasAdd|Relu|Sigmoid|Tanh|.*Pool|Add|ResAdd",
        }

# graph: nodes, edges, and operations
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
                    fused_ops.append(next_nodes[0])
                    fused_ops = self.get_next_fused_op(fused_ops)
        return fused_ops                    
    # starting from current node position, collect as many operations as possible            
    def get_fused_ops(self):
        fused_ops = []
        if (self.current_node == None):
            print("ERROR: found zero operations to fuse")
            exit(-1)
        # when we see ResAdd, backtrack to the last split and follow the next leg in list
        if (self.current_node.data['layer_type'] == "ResAdd" and self.last_split_next_nodes != []):
            if (args.debug > 0): print("DBG: found ResAdd, back-track to last split and follow next leg")
            self.current_node = self.last_split_next_nodes[0] 
            self.last_split_next_nodes = self.last_split_next_nodes[1:]
        fused_ops.append(self.current_node)
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
            print("DBG: fused_ops collected: ",)
            for i in fused_ops:
                print("    ", i.data["layer_type"],":",i.data["layer_name"], )
            print("")
        return fused_ops                    
    def walk_ended(self):
        return self.current_node == None

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
        self.instr_stream = []

    # Compute matrix multiply loops
    # Inputs:
    #   N: batch size
    #   C: number of IFMAPs
    #   H: height of IFMAP
    #   W: width of IFMAP
    #   M: number of OFMAPs
    #   E: height of OFMAP
    #   F: width of OFMAP
    #   pad_*: padding of IFMAP
    #   stride: striding of filter
    # Outputs:
    #   Tn: number of C IFMAPs to roll into a wave
    #   n: number of Tn*C IFMAPs chunks to process
    #   c: number of IFMAPs folds
    #   h: number of single IFMAP/OFMAP folds along y dimension
    #   w: number of single IFMAP/OFMAP folds along x dimension
    #   m: number of OFMAPs folds
    def compute_mm_loops(self, N, C, H, W, M, E, F, pad_north, pad_south, pad_west, pad_east, stride):
        self.N, self.C, self.H, self.W = N, C, H, W 
        self.M, self.E, self.F = M, E, F
        self.pad_north, self.pad_south = pad_north, pad_south
        self.pad_west, self.pad_east = pad_west, pad_east
        self.stride = stride
        # compute the IFMAP folds
        self.c = ceildiv(C, self.pearray.NUM_ROWS)
        # compute the OFMAP folds
        self.m = ceildiv(M, self.pearray.NUM_COLS)
        # computing the input map tiling       
        #self.num_tiles = 1
        self.HW = H*W
        self.EF = E*F
        self.h = 1
        self.w = 1
        self.e = 1
        self.f = 1
        self.n = 1
        self.Tn = 1
        # TODO: determine tile size and aspect ratio taking into account pooling size
        #self.ofmap_tiley_sz = min(self.E, int(math.sqrt(self.pearray.MAX_WAVE_SIZE)))
        #self.ofmap_tilex_sz = min(self.F, int(math.sqrt(self.pearray.MAX_WAVE_SIZE)))
        # per kaena-85, use noodle shapes for tiles (TODO: if pooling, then need to make num rows = pooling rows)
        self.ofmap_tilex_sz = min(self.F, self.pearray.MAX_WAVE_SIZE)
        self.ofmap_tiley_sz = self.pearray.MAX_WAVE_SIZE // self.ofmap_tilex_sz
        self.ofmap_tile_sz = self.ofmap_tilex_sz * self.ofmap_tiley_sz
        if (self.EF >= self.pearray.MAX_WAVE_SIZE):
            # for now, send in noodles that span width of IFMAP, which has implications for pooling
            #self.num_tiles = ceildiv(H*W, self.pearray.MAX_WAVE_SIZE)
            # for now, send in 16x16 tiles, which has implications for pooling
            # for now, use h/w for both IFMAP and OFMAP
            self.e = ceildiv(E, self.ofmap_tiley_sz)
            self.f = ceildiv(F, self.ofmap_tilex_sz)
        else:
            self.Tn = self.pearray.MAX_WAVE_SIZE // self.EF
            if (self.Tn > self.N):
                self.Tn = self.N
            self.n = ceildiv(self.N, self.Tn)
        self.h = self.e
        self.w = self.f
        self.ifmap_tiley_sz = self.ofmap_tiley_sz * self.stride
        self.ifmap_tilex_sz = self.ofmap_tilex_sz * self.stride
        print("n=%d, Tn=%d, c=%d, h=%d, w=%d, m=%d, ofmap_tilex_sz=%d, ofmap_tiley_sz=%d"%(self.n, self.Tn, self.c, self.h, self.w, self.m, self.ofmap_tilex_sz, self.ofmap_tiley_sz))
    
    # Pack the IFMAPs in columns to create a PE-Array IFMAPs input for a particular wave number
    #   ifmaps: IFMAPs in NCHW format
    #   wave_id: current wave ID, [n_id, m_id, h_id, w_id, c_id, r_id, s_id]
    #   return: a 256x128 array
    def pack_wave_ifmaps(self, ifmaps, wave_id):
        out_array = np.zeros((self.pearray.MAX_WAVE_SIZE, self.pearray.NUM_ROWS))
        # remember to extract IFMAPs starting at r_id, s_id (which should be zero for non-conv op)
        # also need to add zeros for padding
        self.ifmap_wave_lower_addr = -1
        self.ifmap_wave_upper_addr = -1
        row_start = wave_id.c_id * self.pearray.NUM_ROWS
        row_stop = min(self.C, row_start + self.pearray.NUM_ROWS)
        assert(row_start < row_stop)
        for row in range(row_start, row_stop):
            #out_array[:,row] = self.pack_wave_ifmap(ifmaps[:, wave_id.c_id * self.pearray.NUM_ROWS + row], wave_id)
            ifmap = ifmaps[:, row]
            row_offset = row - row_start
            for i in range(self.Tn):
                for x in range(self.ofmap_tilex_sz):
                    for y in range(self.ofmap_tiley_sz):
                        ifmap_tilex = wave_id.w_id * self.ifmap_tilex_sz + (x * self.stride) + wave_id.s_id - self.pad_west
                        ifmap_tiley = wave_id.h_id * self.ifmap_tiley_sz + (y * self.stride) + wave_id.r_id - self.pad_north
                        ifmap_addr = i * self.ofmap_tile_sz + y * self.ofmap_tilex_sz + x
                        if (ifmap_tilex < 0 or ifmap_tilex >= self.W):
                            out_array[ifmap_addr, row_offset] = 0
                        elif (ifmap_tiley < 0 or ifmap_tiley >= self.H):
                            out_array[ifmap_addr, row_offset] = 0
                        else:
                            out_array[ifmap_addr, row_offset] = ifmap[(wave_id.n_id * self.Tn) + i, ifmap_tiley, ifmap_tilex]
                            # Check bounds of actual pixels within the original ifmaps for the first ifmap (which should reside in first SB partition)
                            # TODO: check how N/C are arrange in memory; batching within waves may cause different atoms to be accessed by same wave
                            if (row == row_start):                                
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
        out_array = np.zeros((self.pearray.NUM_ROWS, self.pearray.NUM_COLS))
        row_start = wave_id.c_id * self.pearray.NUM_ROWS
        row_stop = min(self.C, row_start + self.pearray.NUM_ROWS)
        col_start = wave_id.m_id * self.pearray.NUM_COLS
        col_stop = min(self.M, col_start + self.pearray.NUM_COLS)
        for row in range(row_start, row_stop):
            for col in range(col_start, col_stop):
                out_array[row - row_start, col - col_start] = weights[row, wave_id.r_id, wave_id.s_id, col] # CRSM
        self.weight_wave_lower_addr = int(np.ravel_multi_index(
                                            (row_start, wave_id.r_id, wave_id.s_id, col_start), # CRSM
                                            dims=weights.shape) 
                                            * weights.dtype.itemsize)
        self.weight_wave_upper_addr = int(np.ravel_multi_index(
                                            (row_stop - 1, self.R - 1, self.S - 1, col_stop-1), # CRSM
                                            dims=weights.shape) 
                                            * weights.dtype.itemsize)
        return out_array

    # generate MatMul instruction and add it to instruction stream
    def gen_matmul_instr_inline(self, op, dram_weights_instr, dram_ifmaps_instr, wave_id, weights, ifmaps, psum_bank, psum_add):
        input_list = []
        if (len(self.instr_stream) > 0):
            input_list.append(self.instr_stream[-1]['waveop_name'])
        if (dram_weights_instr != None):
            input_list.append(dram_weights_instr['waveop_name'])
            self.instr_stream.append(dram_weights_instr)
        if (dram_ifmaps_instr != None):
            input_list.append(dram_ifmaps_instr['waveop_name'])
            self.instr_stream.append(dram_ifmaps_instr)
        matmul_instr = {
              'previous_waveops'        : input_list,
              'waveop_type'             : 'MatMul',
              'waveop_name'             : op.data['layer_name']+"/MatMul_"+wave_id.id_string(),
              'layer_name'              : op.data['layer_name'],
              'weights_atom_id'         : self.statebuffer.infbuf_weights.current_atom_id,
              'ifmaps_atom_id'          : self.statebuffer.infbuf_ifmaps.current_atom_id,
              'weights_offset_in_atom'  : self.weight_wave_lower_addr % self.statebuffer.infbuf_weights.atom_sz,
              'ifmaps_offset_in_atom'   : self.ifmap_wave_lower_addr % self.statebuffer.infbuf_ifmaps.atom_sz,
              'wave_id_format'          : wave_id.format,
              'wave_id'                 : wave_id.show(),
              'start'                   : not(psum_add),
              'psum_bank_id'            : psum_bank
            }
        self.instr_stream.append(matmul_instr)

    # generate Relu instruction and add it to instruction stream
    def gen_relu_instr_inline(self, op, tile_id, psum_bank):
        input_list = []
        if (len(self.instr_stream) > 0):
            input_list.append(self.instr_stream[-1]['waveop_name'])
        instr = {
              'previous_waveops'        : input_list,
              'waveop_type'             : 'Relu',
              'waveop_name'             : op.data['layer_name']+"/Relu_"+tile_id.id_string(),
              'layer_name'              : op.data['layer_name'],
              'tile_id_format'          : tile_id.format,
              'tile_id'                 : tile_id.show(),
              'psum_bank_id'            : psum_bank
            }
        self.instr_stream.append(instr)

    # Execute conv and other operations in list: for each op, load parameters and perform op with input
    def execute_conv_ops(self, op_list, inputs):
        self.inputs = inputs
        assert (op_list[0].data['layer_type'] == 'Conv')
        # get weights from file
        weights = self.statebuffer.infbuf_weights.load_data(op_list[0])
        #if (op_list[0].data['kernel_format'] == "MCRS"):
        #    M, C, R, S = weights.shape
        if (op_list[0].data['kernel_format'] == "CRSM"):
            C, R, S, M = weights.shape
        else:
            print("ERROR: don't understand kernel format %s"%op_list[0].data['kernel_format'])
            exit(-1)
        weight_cols_per_wave = min(M, self.pearray.NUM_COLS)
        ifmap_cols_per_wave = min(M, self.pearray.NUM_COLS)
        self.R = R
        self.S = R

        # load bias values
        bias = []
        for i in range(1, len(op_list)):
            layer_type = op_list[i].data['layer_type'] 
            if (layer_type == 'BiasAdd'):
                for j in op_list[i].prev:
                    if (j.data['layer_type'] == "Const"):
                        bias_temp = self.statebuffer.infbuf_bias.load_data(j)
                        bias = bias_temp.flatten()

        # initial values
        psum_bank = 0
        psum_add = False                               
        result = np.zeros((self.N, self.M, self.E, self.F))
        # wave loop ordering scheme: nmhwcRS
        for n_id in range(self.n):
            for m_id in range(self.m):
                for h_id in range(self.h):
                    for w_id in range(self.w):
                        tile_id = TileID(n_id, m_id, h_id, w_id)
                        # loops for constructing a tile
                        for c_id in range(self.c):
                            for r_id in range(R):
                                for s_id in range(S):
                                    wave_id = WaveID(n_id, m_id, h_id, w_id, c_id, r_id, s_id)
                                    pearray_packed_weights = self.pack_wave_conv_weights(weights, wave_id)
                                    pearray_packed_ifmaps = self.pack_wave_ifmaps(inputs, wave_id)
                                    #print("\npearray_packed_ifmaps", wave_id.show(), "\n", pearray_packed_ifmaps)
                                    #print("\npearray_packed_weights", wave_id.show(), "\n", pearray_packed_weights)
                                    dram_weights_instr = self.statebuffer.infbuf_weights.cache_data(wave_id, self.weight_wave_lower_addr, self.weight_wave_upper_addr)
                                    dram_ifmaps_instr = self.statebuffer.infbuf_ifmaps.cache_data(wave_id, self.ifmap_wave_lower_addr, self.ifmap_wave_upper_addr)
                                    self.pearray.wave_fp16_mm(pearray_packed_ifmaps, pearray_packed_weights, psum_bank, psum_add)
                                    self.gen_matmul_instr_inline(op_list[0], dram_weights_instr, dram_ifmaps_instr, wave_id, weights, inputs, psum_bank, psum_add)
                                    # after the first wave, subsequent waves results are added to partial sums in buffer
                                    if (not psum_add):
                                        psum_add = True
                        # tile is done                                    
                        self.pearray.trig_tile_done(tile_id)
                        tile_x_start = wave_id.w_id * self.ofmap_tilex_sz
                        tile_y_start = wave_id.h_id * self.ofmap_tiley_sz
                        tile_height = self.ofmap_tiley_sz
                        tile_width = self.ofmap_tilex_sz
                        if ((wave_id.h_id+1) * self.ofmap_tiley_sz > self.E):
                            tile_height = self.E - tile_y_start
                        if ((wave_id.w_id+1) * self.ofmap_tilex_sz > self.F):
                            tile_width = self.F - tile_x_start
                        tile_size = tile_height * tile_width
                        self.ifmap_tile_lower_addr = int(np.ravel_multi_index((wave_id.n_id * self.Tn, 0, tile_y_start, tile_x_start),
                                                        dims=inputs.shape) * inputs.dtype.itemsize)
                        self.ifmap_tile_upper_addr = int(np.ravel_multi_index(
                                                            (self.N - 1, 
                                                            self.C - 1, 
                                                            tile_y_start + tile_height - 1, 
                                                            tile_x_start + tile_width - 1),
                                                        dims=inputs.shape) * inputs.dtype.itemsize)
                        # release portion of cached data
                        # TODO: make sure that this doesn't free needed data
                        self.statebuffer.infbuf_ifmaps.uncache_data(self.ifmap_tile_lower_addr, self.ifmap_tile_upper_addr)
                        # go through the remaining operations
                        op_result = self.pearray.extract_psum(psum_bank, 0, self.ofmap_tile_sz)
                        for i in range(1, len(op_list)):
                            layer_type = op_list[i].data['layer_type'] 
                            if (layer_type == 'Relu'):
                                self.activate.wait_tile_done(tile_id)
                                op_result = self.activate.relu(op_result)
                                self.gen_relu_instr_inline(op_list[i], tile_id, psum_bank)
                                if (i != len(op_list)-1):
                                    self.pearray.write_psum(psum_bank, 0, op_result)
                            elif (layer_type == 'BiasAdd'):
                                self.activate.wait_tile_done(tile_id)
                                bias_start = m_id*self.pearray.NUM_COLS
                                bias_end = min(bias_start + self.pearray.NUM_COLS, self.M)
                                bias_extracted = np.zeros(self.pearray.NUM_COLS)
                                bias_extracted[0 : bias_end - bias_start] = bias[bias_start : bias_end]
                                op_result = self.activate.biasadd(op_result, bias_extracted)
                                dram_bias_instr = self.statebuffer.infbuf_bias.cache_data(wave_id, bias_start, bias_end)
                                # TODO: generate BiasAdd instruction inline
                                if (i != len(op_list)-1):
                                    self.pearray.write_psum(psum_bank, 0, self.ofmap_tile_sz, op_result)
                            else:
                                print ("ERROR: %s is currently not yet implemented"%layer_type)
                                exit(-1)
                        # if operation is the last one, dump current result into a portion of final result
                        for j in range(self.pearray.NUM_COLS):
                            M_idx = wave_id.m_id * self.pearray.NUM_COLS + j
                            if (M_idx >= self.M):
                                break
                            else:
                                # For now, multiply zeros, and at the ofmap, extract tile with zeros, then clip
                                result_tile = (op_result[0 : self.ofmap_tile_sz, j]).reshape((self.ofmap_tiley_sz, self.ofmap_tilex_sz))
                                result[n_id, j, tile_y_start : tile_y_start + tile_height, tile_x_start : tile_x_start + tile_width]\
                                        = result_tile[0:tile_height, 0:tile_width]
                                #print(wave_id.show(),"\n", result_tile)                                        
                        # Advance to new bank, while the old bank is being processed                                        
                        psum_bank = (psum_bank + 1)%self.pearray.PSUM_NUM_BANKS
                        psum_add = False
        return result                    

# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kgraph", help="K-graph Json file to read")
    parser.add_argument("--wavegraph", help="Wave-graph Json file to write")
    parser.add_argument("--dot", help="Dot file to write")
    parser.add_argument("--debug", default=1, help="Debug level")
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

    input_layer = kgraph.first_node.data
    output_layer = kgraph.last_node.data
    print("Input layer: ", input_layer['layer_name'])
    print("Output layer: ", output_layer['layer_name'])

    # add forward references
    kgraph.add_forward_refs(kgraph.last_node)

    # go through all layers and add the fusable operations
    tpb = TPBSched()
    while (not kgraph.walk_ended()):
        op_list = kgraph.get_fused_ops()

        # Check init op
        if (re.search(r"Input", op_list[0].data['layer_type'])):
            inputs = tpb.statebuffer.infbuf_ifmaps.load_data(op_list[0])
            results = inputs
        # Check conv fused op
        # TODO: add matrix multiply
        elif (re.search(r"Conv", op_list[0].data['layer_type'])):
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
            tpb.compute_mm_loops(iN, iC, iH, iW, convC, convH, convW, pad_north, pad_south, pad_west, pad_east, stride_x)
            # TODO: add selecting among pre-derived looping schemes
            results = tpb.execute_conv_ops(op_list, results)

        elif (re.search(r"MatMult", op_list[0].data['layer_type'])):
            print("ERROR: MatMult operation is unimplemented")
        elif (re.search(r".*Pool", op_list[0].data['layer_type'])):
            print("ERROR: Pool (unfused) operation is unimplemented")
        else:        
            print("ERROR: the first operation should be Conv")
            exit(-1)

    # Check results against pre-computed results            
    outputs = np.load(output_layer['ref_file'])
    diff = results - outputs
    print("\nInput IFMAPS:\n", inputs)
    print("\nComputed OFMAPS:\n", results)
    print("\nExpected OFMAPS:\n", outputs)
    print("\nDiffed   OFMAPS:\n", diff)
    if (not np.allclose(results, outputs, 1/100, 1e-7)):
        print("\nFAILED: computed OFMAPS is not equal to expected OFMAPS!\n")
    else:
        print("\nPASSED\n")

    # write out wavegraph           
    wavegraph_json = kgraph_json
    if (args.wavegraph != None): 
        wavegraph_json['waveops'] = tpb.instr_stream
        try:
            print("\nSaving Wave-Graph %s"%args.wavegraph)
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

    if (args.dot != None):            
        dot = Digraph()
        for i in tpb.instr_stream:
            dot.node(i['waveop_name'], i['waveop_name'])
            for j in i['previous_waveops']:
                dot.edge(j, i['waveop_name'])
        (dotfile_root, dotfile_ext) = os.path.splitext(args.dot)                
        dot.format = dotfile_ext[1:]
        dot.render(dotfile_root)
        print("INFO: Wrote " + args.dot)

    # test by reading it back
    try:
        print("\nTest by loading Wave-graph %s"%args.wavegraph)
        wavegraph_json = json.load(open(args.wavegraph))
    except Exception as e:
        print(e)
        exit(-1)

    # create graph from JSON file        
    wavegraph = KGraph()
    wavegraph.populate_from_kgraph_json(wavegraph_json)


