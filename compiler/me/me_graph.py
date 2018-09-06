"""
Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
"""

""" Graph and graph nodes for KGraph and WaveGraph
"""

import re
import os
import copy
import numpy as np

from me_utils import ceildiv
from me_utils import ShapeDims
from me_utils import FileParams
from me_utils import Coord
from me_utils import Dim2D
from me_utils import Rect
from me_models import PEArray
from me_fused_op import FusedOp

"""Macros for dumping arrays
"""
def DBG_DUMP_ARRAY(msg, a):
    print (msg, "\n" , a)
    return a

def DBG_DUMP_PSUM_COL(msg, psum, col):
    x = psum[:, col]
    print (msg, "\n" , x)
    return x

"""Neural network node, containing data read from JSON
"""
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
        self.result_avail = False
        self.filter = Dim2D(0,0)
        self.pool_window = Dim2D(0,0)
        self.stride = Dim2D(1,1)
        self.padWN = Dim2D(0,0)
        self.padES = Dim2D(0,0)
        self.is_nop = False
        self.is_placeholder = False
        self.is_input = False
        self.is_const = False
        self.is_join = False
        self.is_fork = False
        self.is_id_pool = False
        self.is_conv_transpose = False
        self.is_concat = False
        self.residue_index = -1
        self.src_is_psum = True
        self.dst_is_psum = True
        self.src_circbuf = None
        self.dst_circbuf = None
        self.ifmaps_file_params = None
        self.ifmaps_file_params_concat = []
        self.weights_file_params_concat = []
        self.ofmaps_file_params = None
        self.weights_file_params = None
        self.bias_file_params = None
        self.fused_op = None
        self.repl_multiple_of_C = 1
        self.ifmaps_padded_and_split = False
        self.distance_to_next_join = 1000 

    def add_prev(self, prev_node):
        self.prev.append(prev_node)

    def add_next(self, next_node):
        if (not self.in_next(next_node)):
            self.next.append(next_node)
            return True
        return False
    def in_next(self, node):
        for i in self.next:
            if (i == node):
                return True
        return False   
    # Returns number of missing input results
    def count_missing_input_results(self):
        count = 0
        for i in self.prev:
            if (not i.is_const) and (not i.result_avail):
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
                                    file_name   = file_name,
                                    file_dims   = ofmaps_shape_dims, 
                                    data_type   = self.parent.data_type, 
                                    op_params   = self,
                                    args        = self.parent.args)
        self.ofmaps_file_params.layer_name =  layer_info['layer_name']
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
                                            file_name       = prev_node.data['ref_file'], 
                                            file_dims       = bias_shape_dims, 
                                            data_type       = self.parent.data_type, 
                                            op_params       = self,
                                            args            = self.parent.args,
                                            contain_weights = True)
                self.bias_file_params.layer_name =  prev_node.data['layer_name']
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
            # taemk :
            # Create a separate container for ifmap file params of Concat
            # operation
            self.ifmaps_file_params_concat = []
            for i in range(len(self.prev)):
                self.prev[i].ofmaps_file_params.consumers.append(self)
                self.ifmaps_file_params_concat.append(\
                    self.prev[i].ofmaps_file_params)
        else:
            print("Layer % has no input"%(self.data['layer_name']))
        # since M is set to C above, but for Softmax2, we just need one output channel (sum across input channels)
        layer_info = self.data
        if (layer_info['layer_type'] == 'Softmax2'): self.M = 1
        # get padding and stride information
        if ('padding' in layer_info):            
            self.padWN = Dim2D(layer_info['padding'][3][0], layer_info['padding'][2][0])
            self.padES = Dim2D(layer_info['padding'][3][1], layer_info['padding'][2][1])
        if ('stride' in layer_info):            
            self.stride = Dim2D(layer_info['stride'][3], layer_info['stride'][2])
        # OFMAP total areas
        self.EF = self.E * self.F
        # Construct rectangles for later use            
        self.ifmap_full_rect = Dim2D(self.W, self.H).make_rect()
        self.ofmap_full_rect = Dim2D(self.F, self.E).make_rect()
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
        if (adjust_for_pool and self.ofmap_full_tiley_sz < self.pool_window.y):
            self.ofmap_full_tiley_sz = min(self.E, self.pool_window.y)
            self.ofmap_full_tilex_sz = min(self.F, PEArray.MAX_WAVE_SIZE // self.ofmap_full_tiley_sz)
        # Construct rectangle for later use            
        self.ofmap_full_tile_dim2d = Dim2D(self.ofmap_full_tilex_sz, self.ofmap_full_tiley_sz)
        self.ofmap_full_tile_rect   = self.ofmap_full_tile_dim2d.make_rect()
        self.ofmap_full_tile_sz     = self.ofmap_full_tile_rect.get_tot_size()
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
            self.e = ceildiv(self.E, self.ofmap_full_tile_dim2d.y)
            self.f = ceildiv(self.F, self.ofmap_full_tile_dim2d.x)
        # heigh/width folding is the same for IFMAP and OFMAP            
        self.h = self.e
        self.w = self.f
        print("Common params1 for layer %s:  N=%d, M=%d, H=%d, W=%d, C=%d, E=%d, F=%d"
                %(self.data['layer_name'], self.N, self.M, self.H, self.W, self.C, self.E, self.F))
        print("Common params2 for layer %s:  n=%d, m=%d, h=%d, w=%d, c=%d, Tn=%d"
                %(self.data['layer_name'], self.n, self.m, self.h, self.w, self.c, self.Tn))
        print("Common params3 for layer %s:  stride_x=%d, stride_y=%d, ofmap_full_tilex_sz=%d, ofmap_full_tiley_sz=%d, ofmap_full_tile_sz=%d"
                %(self.data['layer_name'], self.stride.x, self.stride.y, self.ofmap_full_tile_dim2d.x, self.ofmap_full_tile_dim2d.y, self.ofmap_full_tile_rect.get_tot_size()))

    # Compute Conv looping params
    def populate_conv_params(self):
        # convolution kernel shape
        layer_info = self.data
        if (layer_info['layer_type'] == 'Softmax2'):
            weights_shape_dims = ShapeDims(layer_info['ofmap_format'], layer_info['ofmap_shape'])            
            weights_file = self.data['ref_file'].replace(".npy", "-ones.npy")
            if (not self.parent.args.inference):
                ones_tensor = np.ones(weights_shape_dims.shape_tuple, dtype=self.parent.data_type)
                np.save(weights_file, ones_tensor)
        else:
            weights_shape_dims = ShapeDims(layer_info['kernel_format'], layer_info['kernel_shape'])            
            weights_file = self.data['kernel_file']
        self.weights_file_params = FileParams(
                                        file_name       = weights_file, 
                                        file_dims       = weights_shape_dims, 
                                        data_type       = self.data_type, 
                                        op_params       = self, 
                                        args            = self.parent.args, 
                                        contain_weights = True)
        self.weights_file_params.layer_name =  self.data['layer_name']
        self.weights_file_params.load_file()
        self.weights_file_params.consumers.append(self)
        self.R = weights_shape_dims.R
        self.S = weights_shape_dims.S
        self.RS = self.R * self.S
        self.filter = Dim2D(self.S, self.R)
        # kaena-141: replicate IFMAP a number of times.
        # The number is determined by S, multiplied by a portion of R to match r*S*C <= 128
        # In the case of 1st layer ResNet50, R=7, S=7, C=3 so R can be broken a number of ways. 
        # For now, split evenly among two waves.
        self.repl_multiple_of_C = 1
        if self.parent.args.enable_replication and self.is_input:
            num_replicated_waves = ceildiv(self.RS * weights_shape_dims.C,  PEArray.NUM_ROWS)
            self.repl_multiple_of_C = ceildiv(self.R, num_replicated_waves) * self.S
        print("Conv params for layer %s: R=%d, S=%d, repl_multiple_of_C=%d"%(self.data['layer_name'], self.weights_file_params.file_dims.R, self.weights_file_params.file_dims.S, self.repl_multiple_of_C))

    # Compute pooling params
    def populate_pooling_params(self):
        # are the dimensions from layer info correct?
        layer_info = self.data
        self.pool_window = Dim2D(layer_info['kernel_shape'][3], layer_info['kernel_shape'][2])
        self.stride      = Dim2D(layer_info['stride'][3], layer_info['stride'][2])
        print("Pooling params for layer %s: pool_window_x=%d, pool_window_y=%d, stride_x=%d, stride_y=%d"
                %(self.data['layer_name'], self.pool_window.x, self.pool_window.y, self.stride.x, self.stride.y))

    # Recompute conv tile params due to fused pooling
    def recompute_conv_params(self):
        # For pooling using PSUM (fused), max tile size must be a multiple of pooling window
        self.ofmap_full_tile_dim2d = (self.ofmap_full_tile_dim2d // pool_window) * pool_window
        self.ofmap_full_tile_rect = self.ofmap_full_tile_dim2d.make_rect()
        self.ofmap_full_tile_sz   = self.ofmap_full_tile_rect.get_tot_size()
        print("Recomputed Conv params due to fused pooling: pool_window_x=%d, pool_window_y=%d, ofmap_full_tilex_sz=%d, ofmap_full_tiley_sz=%d"
                %(self.pool_window.x, self.pool_window.y, self.ofmap_full_tile_dim2d.x, self.ofmap_full_tile_dim2d.y))

    """ compute output tile info
        Parameters:
            - ifmap_tile, ofmap_tile: IFMAP/OFMAP tile objects being processed
        Updates:
            - ofmap_count: Number of output FMAP channels of the tile being processed
            - ofmap_cropped_tile_rect: OFMAP tile rectangle, cropped to actual OFMAP rectangle (ie edge cases)
            - ifmap_cropped_tile_rect: IFMAP tile rectangle (strided, with padding), cropped to actual IFMAP rectangle (ie edge cases)
            - ofmap_tile_lower_addr/ofmap_tile_upper_addr: Addresses within file of OFMAP tile (for mappingt to SB)
            - ifmap_tile_lower_addr/ifmap_tile_upper_addr: Addresses within file of IFMAP tile (for mappingt to SB)
    """
    def compute_ifmap_ofmap_tile_info(self, ifmap_tile, ofmap_tile, conv_transpose=False):        
        # number of OFMAPs for this tile 
        self.ofmap_count = ofmap_tile.get_ofmap_count()

        assert(ifmap_tile.is_ifmap == True) 
        assert(ofmap_tile.is_ifmap == False) 

        # compute the bounds for OFMAP tile within OFMAPs tensor (adjusted for boundary conditions)
        ofmap_tile_start_coord       = ofmap_tile.get_fmap_coord(self.ofmap_full_tile_rect.get_width_height())
        ofmap_tile_start_coord       = ofmap_tile_start_coord + Coord(0, self.unstack_h_offset)
        ofmap_tile.padded_tile_rect  = self.ofmap_full_tile_rect + ofmap_tile_start_coord
        ofmap_tile.tile_rect         = ofmap_tile.padded_tile_rect.get_overlap(self.ofmap_full_rect)
        # compute the bounds for IFMAP tile within IFMAPs tensor (adjusted for boundary conditions)
        # projecting backwards from OFMAP tile
        if conv_transpose:
            if self.filter > self.stride:
                ofmap_tile.padded_tile_rect = ofmap_tile.padded_tile_rect.increase_size(self.filter - self.stride)
            ofmap_tile.padded_tile_rect  = ofmap_tile.padded_tile_rect - self.padWN
            ofmap_tile.tile_rect         = ofmap_tile.padded_tile_rect.get_overlap(self.ofmap_full_rect)
            ifmap_tile.padded_tile_rect  = ofmap_tile.tile_rect // self.stride
        else:
            ifmap_tile.padded_tile_rect  = ofmap_tile.tile_rect * self.stride
            # TODO: handle the case of conv is followed by fused pool, so both pool_window and filter exists (two possible different OFMAPs)
            if self.pool_window > self.stride:
                ifmap_tile.padded_tile_rect = ifmap_tile.padded_tile_rect.increase_size(self.pool_window - self.stride)
            if self.filter > self.stride:
                ifmap_tile.padded_tile_rect = ifmap_tile.padded_tile_rect.increase_size(self.filter - self.stride)
            ifmap_tile.padded_tile_rect  = ifmap_tile.padded_tile_rect - self.padWN
        ifmap_tile.tile_rect         = ifmap_tile.padded_tile_rect.get_overlap(self.ifmap_full_rect)
        #print(ofmap_tile.padded_tile_rect, ofmap_tile.tile_rect, ifmap_tile.padded_tile_rect, ifmap_tile.tile_rect)

        # obtain file address bounds of the OFMAP tile
        (ofmap_tile.lower_addr, ofmap_tile.upper_addr) = ofmap_tile.make_pewave().get_file_addrs()
        ofmap_tile.lower_to_upper_len_bytes = []
        for i in range(len(ofmap_tile.lower_addr)):
            ofmap_tile.lower_to_upper_len_bytes.append(ofmap_tile.upper_addr[i] - ofmap_tile.lower_addr[i] + self.item_sz)

        # obtain file address bounds of the IFMAP tile
        if not self.src_is_psum:
            (ifmap_tile.lower_addr, ifmap_tile.upper_addr) = ifmap_tile.make_pewave().get_file_addrs()
        else:
            (ifmap_tile.lower_addr, ifmap_tile.upper_addr) = (-1, -1)
        ifmap_tile.lower_to_upper_len_bytes = []
        for i in range(len(ifmap_tile.lower_addr)):
            ifmap_tile.lower_to_upper_len_bytes.append(ifmap_tile.upper_addr[i] - ifmap_tile.lower_addr[i] + self.item_sz)

    """ compute input/output PE-Wave info
        Parameters:
            - ifmap_pewave, ofmap_pewave: IFMAP/OFMAP pewave objects being processed
        Updates:
            - ofmap_cropped_tile_rect: OFMAP tile rectangle, cropped to actual OFMAP rectangle (ie edge cases)
            - ifmap_cropped_tile_rect: IFMAP tile rectangle (strided, with padding), cropped to actual IFMAP rectangle (ie edge cases)
            - ofmap_tile_lower_addr/ofmap_tile_upper_addr: Addresses within file of OFMAP tile (for mappingt to SB)
            - ifmap_tile_lower_addr/ifmap_tile_upper_addr: Addresses within file of IFMAP tile (for mappingt to SB)
    """
    def compute_ifmap_ofmap_pewave_info(self, ifmap_pewave, ofmap_pewave, conv_transpose=True):        
        # compute the bounds for OFMAP PE-Wave within OFMAP tile (adjusted for boundary conditions)a
        if conv_transpose:
            # Compute padded PE-Wave rectangle
            padded_ofmap_pewave_rect   = ifmap_pewave.tile.tile_rect * self.stride
            padded_ofmap_pewave_rect   = padded_ofmap_pewave_rect + Coord(ofmap_pewave.s_id, ofmap_pewave.r_id)
            padded_ofmap_pewave_rect   = padded_ofmap_pewave_rect - self.padWN
            ofmap_pewave.subtile_rect  = padded_ofmap_pewave_rect.get_overlap(ofmap_pewave.tile.tile_rect)
            # Snap rectangle to the striding grid
            ofmap_pewave.subtile_rect.snap_rect_to_stride_grid(padded_ofmap_pewave_rect.lower, self.stride)
            # Offset from input tile to the IFMAP PE-Wave (to avoid computing explicit padding pixels)
            ofmap_pewave_offset              = ofmap_pewave.subtile_rect.get_offset_from(padded_ofmap_pewave_rect)
            ifmap_pewave_rect_dim2d          = ceildiv(ofmap_pewave.subtile_rect.dim2d, self.stride)
            ifmap_pewave.subtile_rect        = ifmap_pewave_rect_dim2d.make_rect()
            ifmap_pewave_offset              = ofmap_pewave_offset // self.stride
            ifmap_pewave.subtile_psum_offset = ifmap_pewave_offset.y * ifmap_pewave.tile.tile_rect.dim2d.x + ifmap_pewave_offset.x
            ifmap_pewave.subtile_rect.set_lower(ifmap_pewave.tile.tile_rect.lower + ifmap_pewave_offset)
        else:
            # Compute padded PE-Wave rectangle
            padded_ifmap_pewave_rect   = ofmap_pewave.tile.tile_rect * self.stride
            padded_ifmap_pewave_rect   = padded_ifmap_pewave_rect + Coord(ifmap_pewave.s_id, ifmap_pewave.r_id)
            padded_ifmap_pewave_rect   = padded_ifmap_pewave_rect - self.padWN
            ifmap_pewave.subtile_rect  = padded_ifmap_pewave_rect.get_overlap(ifmap_pewave.tile.tile_rect)
            # Snap rectangle to the striding grid
            ifmap_pewave.subtile_rect.snap_rect_to_stride_grid(padded_ifmap_pewave_rect.lower, self.stride)
            #print("compute_ifmap_ofmap_pewave_info: tile_rect ", ifmap_pewave.tile.tile_rect, " padded pewave_rect ", padded_ifmap_pewave_rect, " subtile_rect ", ifmap_pewave.subtile_rect)
            # Offset from input tile to the IFMAP PE-Wave (to avoid computing explicit padding pixels)
            ifmap_pewave_offset              = ifmap_pewave.subtile_rect.get_offset_from(padded_ifmap_pewave_rect)
            ofmap_pewave_rect_dim2d          = ceildiv(ifmap_pewave.subtile_rect.dim2d, self.stride)
            ofmap_pewave.subtile_rect        = ofmap_pewave_rect_dim2d.make_rect()
            ofmap_pewave_offset              = ifmap_pewave_offset // self.stride
            ofmap_pewave.subtile_psum_offset = ofmap_pewave_offset.y * ofmap_pewave.tile.tile_rect.dim2d.x + ofmap_pewave_offset.x
            ofmap_pewave.subtile_rect.set_lower(ofmap_pewave.tile.tile_rect.lower + ofmap_pewave_offset)

        # obtain file address bounds of the PE-Wave
        if not self.src_is_psum:
            (ifmap_pewave.lower_addr, ifmap_pewave.upper_addr) = ifmap_pewave.get_subtile_file_addrs()
        else:
            (ifmap_pewave.lower_addr, ifmap_pewave.upper_addr) = (-1, -1)
        ifmap_pewave.lower_to_upper_len_bytes = []
        for i in range(len(ifmap_pewave.lower_addr)):
            ifmap_pewave.lower_to_upper_len_bytes.append(ifmap_pewave.upper_addr[i] - ifmap_pewave.lower_addr[i] + self.item_sz)


    # Pack the IFMAPs in columns to create a PE-Array IFMAPs input for a particular wave number
    #   ifmaps: IFMAPs in NCHW format
    #   pewave: current tile wave, IDed by [n_id, m_id, h_id, w_id, c_id, r_id, s_id]
    #   layer_type: 'conv' or 'MaxPool'; can be extended to handle other layer types
    #   return: a 256x128 array
    def pack_wave_ifmaps(self, ifmaps, ifmap_pewave, ofmap_pewave, repl_multiple_of_C, for_softmax):
        # If we are not doing convolution (aka pooling), set out_array_dim_y to be PEArray.NUM_COLS to match pooling/activation engines dimension
        if (for_softmax):
            out_array_dim_y = ofmap_pewave.tile.channel_count
        else:            
            out_array_dim_y = ifmap_pewave.ifmap_channel_count * repl_multiple_of_C
        fmap_folding_idx = ifmap_pewave.c_id
        fmap_total_count = self.C
        out_array = np.zeros((PEArray.MAX_WAVE_SIZE, out_array_dim_y))
        #out_array = np.zeros((ofmap_pewave.tile.tile_rect.get_tot_size(), out_array_dim_y))
        # remember to extract IFMAPs starting at r_id, s_id (which should be zero for non-conv op)
        # also need to add zeros for padding
        self.ifmap_wave_lower_addr = [-1 for i in range(self.Tn)]
        self.ifmap_wave_upper_addr = [-1 for i in range(self.Tn)]
        self.ifmap_wave_lower_coordx = [0 for i in range(self.Tn)]
        self.ifmap_wave_lower_coordy = [0 for i in range(self.Tn)]
        self.ifmap_wave_upper_coordx = [0 for i in range(self.Tn)]
        self.ifmap_wave_upper_coordy = [0 for i in range(self.Tn)]
        self.ofmap_wave_lower_coordx = [0 for i in range(self.Tn)]
        self.ofmap_wave_lower_coordy = [0 for i in range(self.Tn)]
        self.ofmap_wave_upper_coordx = [0 for i in range(self.Tn)]
        self.ofmap_wave_upper_coordy = [0 for i in range(self.Tn)]
        self.psum_bank_offset = 0
        # for pooling, the "row" below actually means output columns
        pe_row_start = ifmap_pewave.ifmap_channel_start
        pe_row_stop = ifmap_pewave.ifmap_channel_stop
        assert(pe_row_start < pe_row_stop)
        r_id = ofmap_pewave.r_id
        s_id = ofmap_pewave.s_id
        num_rows = pe_row_stop - pe_row_start
        for repl in range(repl_multiple_of_C):
            pe_row_repl_start = num_rows * repl
            s_id_temp = s_id
            r_id_temp = r_id
            if self.is_conv_transpose:
                s_id_temp = self.S - 1 - s_id
                r_id_temp = self.R - 1 - r_id
            for row in range(pe_row_start, pe_row_stop):
                pe_row_offset = pe_row_repl_start + row - pe_row_start
                for z in range(self.Tn):
                    batch_id = (ofmap_pewave.tile.n_id * self.Tn) + z
                    for x in range(self.ofmap_full_tile_dim2d.x):
                        for y in range(self.ofmap_full_tile_dim2d.y):
                            if self.is_conv_transpose:
                                ifmap_tilex = ofmap_pewave.tile.w_id * self.ofmap_full_tile_dim2d.x + x
                                ifmap_tiley = ofmap_pewave.tile.h_id * self.ofmap_full_tile_dim2d.y + y + self.unstack_h_offset
                            else:
                                ifmap_tilex = (ofmap_pewave.tile.w_id * self.ofmap_full_tile_dim2d.x + x) * self.stride.x + s_id - self.padWN.x
                                ifmap_tiley = ((ofmap_pewave.tile.h_id + self.unstack_h_offset) * self.ofmap_full_tile_dim2d.y + y) * self.stride.y + r_id - self.padWN.y
                            ifmap_addr = z * self.ofmap_full_tile_rect.get_tot_size() + y * self.ofmap_full_tile_dim2d.x + x
                            if (ifmap_tilex < 0 or ifmap_tilex >= self.W):
                                out_array[ifmap_addr, pe_row_offset] = 0
                            elif (ifmap_tiley < 0 or ifmap_tiley >= self.H):
                                out_array[ifmap_addr, pe_row_offset] = 0
                            else:
                                if (self.parent.args.nname == "lm"):
                                    out_array[ifmap_addr, pe_row_offset] = self.ifmaps_file_params.elem_nchw(batch_id, row, ifmap_tiley, ifmap_tilex)
                                else:                                   
                                    if repl_multiple_of_C > 1:
                                        out_array[ifmap_addr, pe_row_offset] = ifmaps[batch_id, row, (ifmap_tiley%2)*ceildiv(self.H,2) + ifmap_tiley//2 + (ifmap_tilex%2)*self.H, ifmap_tilex//2]
                                        #out_array[ifmap_addr, pe_row_offset] = ifmaps[batch_id, row, ifmap_tiley, ifmap_tilex]
                                    else:                                        
                                        out_array[ifmap_addr, pe_row_offset] = ifmaps[batch_id, row, ifmap_tiley, ifmap_tilex]
                                # Check bounds of actual pixels within the original ifmaps for the first ifmap (which should reside in first SB partition)
                                # TODO: check how N/C are arrange in memory; batching within waves may cause different atoms to be accessed by same wave
                                # TODO: for Tn>1, need to have multiple bounds for each batch item
                                if (repl == 0 and row == pe_row_start):                                
                                    # NCHW
                                    self.ifmap_wave_upper_addr[z] = self.ifmaps_file_params.ravel_nchw(batch_id, row, ifmap_tiley, ifmap_tilex)
                                    self.ifmap_wave_upper_coordx[z] = ifmap_tilex
                                    self.ifmap_wave_upper_coordy[z] = ifmap_tiley
                                    self.ofmap_wave_upper_coordx[z] = x
                                    self.ofmap_wave_upper_coordy[z] = y
                                    if (self.ifmap_wave_lower_addr[z] < 0):
                                        self.ifmap_wave_lower_addr[z] = self.ifmap_wave_upper_addr[z]
                                        self.ifmap_wave_lower_coordx[z] = ifmap_tilex
                                        self.ifmap_wave_lower_coordy[z] = ifmap_tiley
                                        self.ofmap_wave_lower_coordx[z] = x
                                        self.ofmap_wave_lower_coordy[z] = y
                                        self.psum_bank_offset = (y * self.ofmap_full_tile_dim2d.x + x)
                            #print("x %d y %d ifmap_tilex %d ifmap_tiley %d wave_lower_coordx %d wave_upper_coordy %d wave_upper_coordx %d wave_upper_coordy %d"%(x, y, ifmap_tilex, ifmap_tiley, self.ofmap_wave_lower_coordx[0], self.ofmap_wave_lower_coordy[0], self.ofmap_wave_upper_coordx[0], self.ofmap_wave_upper_coordy[0]))                                    
                            #if (self.parent.args.debug > 3): print("DBG: pack_wave_ifmaps for wave %s batch_id %d x %d y %d r_id %d s_id %d padN %d padW %d ifmap_tilex %d ifmap_tiley %d wave_lower_coordx %d wave_upper_coordy %d wave_upper_coordx %d wave_upper_coordy %d"%(ofmap_pewave.tile.id_array, batch_id, x, y, r_id, s_id, self.padWN.y, self.padWN.x, ifmap_tilex, ifmap_tiley, self.ofmap_wave_lower_coordx[0], self.ofmap_wave_lower_coordy[0], self.ofmap_wave_upper_coordx[0], self.ofmap_wave_upper_coordy[0]))                                    
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

    def pack_wave_ifmaps_deconv(self, ifmap_pewave, ofmap_pewave):
        assert(self.is_conv_transpose)
        ifmap_tile_data = ifmap_pewave.get_subtile_data_from_file(flatten=True)
        return ifmap_tile_data

    """ Pack input data for unfused pooling (like a DMA operation)
        Parameters:
            - ifmaps: IFMAPs data in NCHW format
            - pewave: current pewave, with ID=[n_id, m_id, h_id, w_id, c_id, r_id, s_id]
        Returns:
            - an array of IFMAPs arranged as NUM_COLS of flattened FMAPs
    """            
    def pack_wave_ifmaps_unfused_pooling (self, ifmaps, pewave):
        # If we are not doing convolution (aka pooling), set out_array_dim_y to be PEArray.NUM_COLS to match pooling/activation engines dimension
        out_array_dim_y = PEArray.NUM_COLS
        fmap_folding_idx = pewave.tile.m_id
        fmap_total_count = self.M
        out_array = np.zeros((PEArray.MAX_WAVE_SIZE * self.stride.x * self.stride.y * 2, out_array_dim_y))
        # remember to extract IFMAPs starting at r_id, s_id (which should be zero for non-conv op)
        # also need to add zeros for padding
        self.psum_bank_offset = 0
        # for pooling, the "row" below actually means output columns
        pe_row_start = fmap_folding_idx * out_array_dim_y + self.stridedslice_chan_offset
        pe_row_stop = min(fmap_total_count + self.stridedslice_chan_offset, pe_row_start + out_array_dim_y)
        assert(pe_row_start < pe_row_stop)
        # make use of the upper 64 partitions by resetting address back to 128-channels boundary
        row_adjust = (pewave.tile.m_id%2)*PEArray.NUM_COLS
        for row in range(pe_row_start, pe_row_stop):
            ifmap = ifmaps[:, row]  # NCHW
            pe_row_offset = row - pe_row_start
            for z in range(self.Tn):
                batch_id = (pewave.tile.n_id * self.Tn) + z
                row_temp = ifmap[batch_id,
                                 pewave.tile.tile_rect.lower.y : pewave.tile.tile_rect.upper.y + 1,
                                 pewave.tile.tile_rect.lower.x : pewave.tile.tile_rect.upper.x + 1].flatten()
                out_array[z * len(row_temp) : (z+1) * len(row_temp), pe_row_offset] = row_temp
        return out_array

    """ Pack the conv weights in columns to create a PE-Array weights array for a particular wave number
        Parameters:
            - weights: conv weights in CRSM format
            - pewave: current pewave, with ID=[n_id, m_id, h_id, w_id, c_id, r_id, s_id]
        Returns: 
            - a 128x64 array of weights values
        Updates: 
            - this KNode object's ifmap_count/ofmap_count (TODO: consolidate this elsewhere, maybe compute_ifmap_ofmap_tile_info)
            - weight_wave_lower/upper_addr (TODO: consolidate this elsewhere, maybe compute within Wave object when needed to emit waveop)
    """
    def pack_wave_conv_weights(self, weights, ifmap_pewave, ofmap_pewave, repl_multiple_of_C):
        out_array = np.zeros((ifmap_pewave.ifmap_channel_count * repl_multiple_of_C, ofmap_pewave.tile.channel_count))
        r_id = ofmap_pewave.r_id
        s_id = ofmap_pewave.s_id
        last_r_id = r_id
        last_s_id = s_id
        for repl in range(repl_multiple_of_C):
            pe_row_repl_start = ifmap_pewave.ifmap_channel_count * repl
            s_id_temp = s_id
            r_id_temp = r_id
            if self.is_conv_transpose:
                s_id_temp = self.S - 1 - s_id
                r_id_temp = self.R - 1 - r_id
            last_r_id = r_id_temp
            last_s_id = s_id_temp
            for row in range(ifmap_pewave.ifmap_channel_start, ifmap_pewave.ifmap_channel_stop):
                pe_row_offset = pe_row_repl_start + row - ifmap_pewave.ifmap_channel_start 
                for col in range(ofmap_pewave.tile.channel_start, ofmap_pewave.tile.channel_stop):
                    out_array[pe_row_offset, col - ofmap_pewave.tile.channel_start] = weights[row, r_id_temp, s_id_temp, col] # CRSM
            if (self.parent.args.debug > 2): print("DBG: pack_wave_conv_weights for wave %s conv_transpose %d r_id %d s_id %d (repl_multiple_of_C %d)"%(ofmap_pewave.tile.id_array, self.is_conv_transpose, r_id_temp, s_id_temp, repl_multiple_of_C))
            s_id += 1
            if (s_id >= self.S): 
                r_id += 1
                s_id = 0
                if (r_id >= self.R): break

        self.ofmap_count = ofmap_pewave.tile.channel_count
        self.ifmap_count = ifmap_pewave.ifmap_channel_count * repl_multiple_of_C

        if self.is_conv_transpose:
            self.weight_wave_lower_addr = self.weights_file_params.ravel_crsm(
                                            ifmap_pewave.ifmap_channel_start, 
                                            last_r_id, 
                                            last_s_id, 
                                            ofmap_pewave.tile.channel_start)
            self.weight_wave_upper_addr = self.weights_file_params.ravel_crsm(
                                            ifmap_pewave.ifmap_channel_start, 
                                            self.R - 1 - ofmap_pewave.r_id, 
                                            self.S - 1 - ofmap_pewave.s_id, 
                                            ofmap_pewave.tile.channel_stop - 1)
        else:            
            self.weight_wave_lower_addr = self.weights_file_params.ravel_crsm(
                                            ifmap_pewave.ifmap_channel_start, 
                                            ofmap_pewave.r_id, 
                                            ofmap_pewave.s_id, 
                                            ofmap_pewave.tile.channel_start)
            self.weight_wave_upper_addr = self.weights_file_params.ravel_crsm(
                                            ifmap_pewave.ifmap_channel_start, 
                                            last_r_id,
                                            last_s_id,
                                            ofmap_pewave.tile.channel_stop - 1)
        return out_array


"""Graph class for KGraph or WaveGraph
"""
class KGraph:

    def __init__(self, args):
        # Node dictionary contains name -> Node pairs for quick reference
        self.node_dict = {}
        self.final_nodes = []
        self.data_type = 'float16'
        self.item_sz = 2
        self.current_node = None
        self.last_split_next_nodes = []
        self.args = args

    # add forward edges for forward traversals        
    def add_forward_refs(self, final_nodes):
        if final_nodes != []:
            for node in final_nodes:
                #print (node.data['layer_name'], len(node.prev))
                if len(node.prev) > 0:
                    non_const_prev_count = 0
                    for i in node.prev:
                        if not i.is_const:
                            non_const_prev_count += 1
                    node.is_join = (non_const_prev_count > 1)                    
                    if node.is_join:
                        node.distance_to_next_join = 0
                    for i in node.prev:
                        i.distance_to_next_join = node.distance_to_next_join + 1
                        if i.add_next(node):
                            self.add_forward_refs([i])

    # add a copy of layer, and change it to a new type
    def add_copy_with_new_type(self, node_to_copy, new_type, node_number):
        layer = node_to_copy.data
        new_layer = copy.deepcopy(layer)
        new_layer['layer_type'] = new_type
        new_layer['layer_name'] = layer['layer_name'] + "_" + new_type
        new_layer['ref_file'] = layer['ref_file'].replace(".npy", "_" + new_type + ".npy")
        new_node = KNode(self, new_layer, self.item_sz, self.data_type, node_number)
        new_node.add_prev(node_to_copy)
        self.node_dict[ new_layer['layer_name'] ] = new_node
        node_top_copy = new_node

    # populate graph using layer info from JSON                    
    def populate_from_kgraph_json(self, kgraph_json):                    
        # get the lowest significant bit
        if ("data_type" in kgraph_json):
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
        node_number = 0
        if ("layers" in kgraph_json):
            layers = kgraph_json["layers"]
            num_layers = len(layers)
            assert(num_layers >= 1)
            for l in layers:
                new_node = KNode(self, l, self.item_sz, self.data_type, node_number)
                node_number += 1 
                prev_layers = l['previous_layers']
                if (len(prev_layers) > 0):
                    for i in prev_layers:
                        if i in self.node_dict:
                            #if (self.args.debug>0): print("Previous layer for ", new_node.data['layer_name'], " is ", i)
                            prev_node = self.node_dict[i]
                            # dissolve StridedSlice into the next operation
                            if (prev_node.data['layer_type'] == "Pad"):
                                # NCHW
                                new_node.data['padding'] = prev_node.data['padding']
                                prev_node.data['padding'] = [[0, 0], [0, 0], [0, 0], [0, 0]]
                                prev_node.data['ofmap_shape'] = prev_node.prev[0].data['ofmap_shape']
                                assert(new_node.data['padding'][0] == [0, 0])
                                assert(new_node.data['padding'][1] == [0, 0])
                                assert(new_node.data['layer_type'] == "Conv" or new_node.data['layer_type'] == "ConvTranspose")
                                new_node.add_prev(prev_node.prev[0])
                                prev_node.prev = []
                            elif (prev_node.data['layer_type'] == "StridedSlice"):
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
                        if (self.last_split_next_nodes == []):
                            self.last_split_next_nodes.append([])
                        self.last_split_next_nodes[0].append(new_node)
                # assume the last node is the last one processed (JSON graph is in order), at least for the last one
                self.final_nodes.append(new_node)
                self.node_dict[ l['layer_name'] ] = new_node
                # if softmax, expand to multiple subnodes
                if (l['layer_type'] == "Softmax"):
                    self.final_nodes[-1].data['layer_type'] = "Exp"
                    self.add_copy_with_new_type(self.final_nodes[-1], "Softmax2", node_number)
                    node_number += 1 
                    # move ref file attribute to the last operation for final comparisons
                    self.final_nodes[-1].data['ref_file'] = new_node.data['ref_file']
                    new_node.data['ref_file'] = new_node.data['ref_file'].replace(".npy", "_Exp.npy")
                elif (l['layer_type'] == "Const"):
                    new_node.is_const = True
                elif (l['layer_type'] == "Reshape"):
                    new_node.is_nop = True
                elif (l['layer_type'] == "ConvTranspose"):
                    new_node.is_conv_transpose = True
                elif (l['layer_type'] == "Concat"):
                    new_node.is_concat = True
            if (len(self.last_split_next_nodes) > 0 and len(self.last_split_next_nodes[0]) > 0) :                    
                self.current_node = self.last_split_next_nodes[0].pop()
                if self.last_split_next_nodes[0] == []:
                    self.last_split_next_nodes.pop()
            else:
                print("ERROR: can't find any Input layer!")
                exit(-1)

        # process waveops 
        if ("waveops" in kgraph_json):
            layers = kgraph_json["waveops"]
            num_layers = len(layers)
            if (num_layers >= 1):
                for l in layers:
                    new_node = KNode(self, l, self.item_sz, self.data_type, node_number)
                    node_number += 1 
                    try:
                        new_node.order = l['order']
                    except:
                        new_node.order = -1

                    prev_layers = l['previous_waveops']
                    if (len(prev_layers) > 0):
                        for i in prev_layers:
                            if i in self.node_dict:
                                #if (args.debug > 0): print("Previous waveop for ", new_node.data['waveop_name'], " is ", i)
                                new_node.add_prev(self.node_dict[i])
                            else:
                                print("ERROR: node %s isn't declared before %s"%(i, l['waveop_name']))
                                exit(-1)
                    # assume the last node is the last one processed (JSON graph is in order), at least for the last one
                    self.last_node = new_node                
                    self.node_dict[ l['waveop_name'] ] = new_node
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
            if last_node_type in FusedOp.next_is_fusable:
                if next_nodes[0].count_missing_input_results() <= 1:
                    regex = FusedOp.next_is_fusable[last_node_type]
                    if re.search(regex, next_nodes[0].data['layer_type']):               
                        if fused_ops.add(next_nodes[0]):
                            fused_ops = self.get_next_fused_op(fused_ops)
                elif next_nodes[0].is_join:
                    fused_ops.ofmap_is_for_join = True
        return fused_ops                    

    # starting from current node position, collect as many operations as possible            
    def get_fused_ops(self, fused_op_id):
        fused_ops = FusedOp(self.data_type, fused_op_id, self.args)
        if (self.current_node == None):
            print("ERROR: found zero operations to fuse")
            exit(-1)
        # when we see ResAdd/Multiply, backtrack to the last split and follow the next branch in list
        if (self.current_node.is_join and self.current_node.count_missing_input_results() > 0):
            if (self.args.debug > 0): print("DBG: found join (ResAdd, Multiply, etc), back-track to last split and follow next branch")
            if self.last_split_next_nodes != [] and self.last_split_next_nodes[-1] != []:
                self.current_node = self.last_split_next_nodes[-1].pop()
                if self.last_split_next_nodes[-1] == []: 
                    self.last_split_next_nodes.pop()
            else:
                print("ERROR: back-track from a join %s, but can't find fork!"%(self.current_node.data['layer_name']))
                exit(-1)
        fused_ops.add(self.current_node)
        #for i in self.current_node.next:
        #    print(i.data['layer_type'], ":", i.data['layer_name'])
        fused_ops = self.get_next_fused_op(fused_ops)
        # if there are multiple next nodes
        next_nodes = [i for i in fused_ops[-1].next]
        last_node_type = fused_ops[-1].data['layer_type']
        num_next_nodes = len(next_nodes)
        if (num_next_nodes == 1):
            self.current_node = next_nodes[0]   
        elif (num_next_nodes > 1):
            fused_ops[-1].is_fork = True
            # Delete the branch that goes to ResAdd directly first, if it exists.
            for i in range(num_next_nodes):
                if (next_nodes[i].is_join):
                    resadd_node = next_nodes[i]
                    del next_nodes[i]
                    #next_nodes.insert(0, resadd_node)
                    break
            # sort next nodes list based on distance to next ResAdd    
            if len(next_nodes) > 1:
                next_nodes.sort(key=lambda x: x.distance_to_next_join, reverse=True)
            # pick the first branch as current_node                        
            self.current_node = next_nodes.pop()
            # save the remaining branches in a list
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
        if (last_node_type == "Conv"
                or last_node_type == "ConvTranspose" 
                or last_node_type == "MatMul" 
                or last_node_type == "Concat"):
            fused_ops.add(self.gen_id_pool_op(fused_ops[-1]))
        # set the first op source is not PSUM, and last op dest is not PSUM
        fused_ops.first_op = fused_ops[0]
        fused_ops.first_op.src_is_psum = False
        fused_ops.last_op = fused_ops[-1]
        fused_ops.last_op.dst_is_psum = False
        # Optimization: share FMAP space for join operation that has same input shape as output shape
        if fused_ops.has_join \
                and fused_ops.join_op.data["layer_type"] != "Concat" \
                and not fused_ops.join_op.is_input:
            if fused_ops.last_op != fused_ops.join_op:
                # If join is followed by BiasAdd or Activate, use the same ofmaps_file_params as last op
                assert(fused_ops.join_op.ofmaps_file_params.file_dims.shape_tuple == fused_ops.last_op.ofmaps_file_params.file_dims.shape_tuple)
                fused_ops.join_op.ofmaps_file_params = fused_ops.last_op.ofmaps_file_params
                fused_ops.join_op.ofmaps_file_params.writers_of_shared_fmap.append(fused_ops.join_op)
            # If join, capture ofmaps_file_params from the other branch (residue branch).
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
        for i in fused_ops.first_op.ifmaps_file_params_concat:
            if i is not None:
                i.load_file()
        if fused_ops.first_op.ifmaps_file_params is not None:
            fused_ops.first_op.ifmaps_file_params.load_file()
        if fused_ops.last_op.ofmaps_file_params is not None:
            fused_ops.last_op.ofmaps_file_params.zero_file()
            fused_ops.last_op.ofmaps_file_params.writers_of_shared_fmap.append(fused_ops.last_op)
        # transfer replication info from Conv KNode to ifmaps_file_params
        if fused_ops.has_conv:
            fused_ops.conv_op.ifmaps_file_params.weights_S_dim = fused_ops.conv_op.weights_file_params.file_dims.S
            fused_ops.conv_op.ifmaps_file_params.stride = fused_ops.conv_op.stride
            #print("copied ifmaps_file_params.repl_multiple_of_C = weights_file_params.repl_multiple_of_C %d"%(fused_ops.conv_op.weights_file_params.repl_multiple_of_C))
        if (self.args.debug > 0):
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
        id_pool_op.is_id_pool = True
        id_pool_op.prev.append(last_op)
        id_pool_op.next = []
        for next_op in last_op.next:
            id_pool_op.next.append(next_op)
            for j in range(len(next_op.prev)):
                if next_op.prev[j] == last_op:
                    next_op.prev[j] = id_pool_op
        last_op.next = [id_pool_op]                    
        return id_pool_op

    def walk_ended(self):
        return (self.current_node == None and self.last_split_next_nodes == [])

