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
from me_models import PEArray
from me_fused_op import FusedOp

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
        self.is_const = False
        self.is_join = False
        self.is_fork = False
        self.is_id_pool = False
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
                                    file_name,
                                    ofmaps_shape_dims, 
                                    self.parent.data_type, 
                                    2048, 
                                    PEArray, 
                                    self,
                                    self.parent.args)
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
                                            prev_node.data['ref_file'], 
                                            bias_shape_dims, 
                                            self.parent.data_type, 
                                            2048, 
                                            PEArray, 
                                            self,
                                            self.parent.args,
                                            contain_weights=True)
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
        if (self.parent.args.eigenlib_stride):
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
            if (not self.parent.args.inference):
                ones_tensor = np.ones(weights_shape_dims.shape_tuple, dtype=self.parent.data_type)
                np.save(weights_file, ones_tensor)
        else:
            weights_shape_dims = ShapeDims(layer_info['kernel_format'], layer_info['kernel_shape'])            
            weights_file = self.data['kernel_file']
        self.weights_file_params = FileParams(weights_file, weights_shape_dims, self.data_type, 2048, PEArray, self, self.parent.args, contain_weights=True)
        self.weights_file_params.layer_name =  self.data['layer_name']
        self.weights_file_params.load_file()
        self.R = weights_shape_dims.R
        self.S = weights_shape_dims.S
        self.RS = self.R * self.S
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
        self.pool_window_y = layer_info['kernel_shape'][2]
        self.pool_window_x = layer_info['kernel_shape'][3]
        self.stride_y = layer_info['stride'][2]
        self.stride_x = layer_info['stride'][3]
        if (self.parent.args.eigenlib_stride):
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
    def pack_wave_ifmaps(self, ifmaps, wave_id, repl_multiple_of_C, for_softmax):
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
        for repl in range(repl_multiple_of_C):
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
                                if (self.parent.args.nname == "lm"):
                                    out_array[ifmap_addr, pe_row_offset] = self.ifmaps_file_params.elem_nchw(batch_id, row, ifmap_tiley, ifmap_tilex)
                                else:                                   
                                    if repl_multiple_of_C > 1:
                                        out_array[ifmap_addr, pe_row_offset] = ifmaps[batch_id, row, ifmap_tiley + (ifmap_tilex%2)*self.H, ifmap_tilex//2]
                                        #out_array[ifmap_addr, pe_row_offset] = ifmaps[batch_id, row, ifmap_tiley, ifmap_tilex]
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
                            #if (self.parent.args.debug > 3): print("DBG: pack_wave_ifmaps for wave %s batch_id %d x %d y %d r_id %d s_id %d ifmap_tilex %d ifmap_tiley %d wave_lower_coordx %d wave_upper_coordy %d wave_upper_coordx %d wave_upper_coordy %d"%(wave_id.id_array, batch_id, x, y, r_id, s_id, ifmap_tilex, ifmap_tiley, self.ofmap_wave_lower_coordx[0], self.ofmap_wave_lower_coordy[0], self.ofmap_wave_upper_coordx[0], self.ofmap_wave_upper_coordy[0]))                                    
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
    def pack_wave_conv_weights(self, weights, wave_id, repl_multiple_of_C):
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
        for repl in range(repl_multiple_of_C):
            pe_row_repl_start = num_rows * repl
            for row in range(pe_row_start, pe_row_stop):
                pe_row_offset = pe_row_repl_start + row - pe_row_start
                for col in range(pe_col_start, pe_col_stop):
                    out_array[pe_row_offset, col - pe_col_start] = weights[row, r_id, s_id, col] # CRSM
                    last_r_id = r_id
                    last_s_id = s_id
            if (self.parent.args.debug > 2): print("DBG: pack_wave_conv_weights for wave %s r_id %d s_id %d (repl_multiple_of_C %d)"%(wave_id.id_array, r_id, s_id, repl_multiple_of_C))
            s_id += 1
            if (s_id >= self.S): 
                r_id += 1
                s_id = 0
                if (r_id >= self.R): break

        self.ifmap_count = self.ifmap_count * repl_multiple_of_C

        self.weight_wave_lower_addr = self.weights_file_params.ravel_crsm(
                                            pe_row_start, wave_id.r_id, wave_id.s_id, pe_col_start)
        self.weight_wave_upper_addr = self.weights_file_params.ravel_crsm(
                                            pe_row_start, last_r_id, last_s_id, pe_col_stop-1)
        return out_array

"""RegExs to determine whether next node is fusable or not
"""
next_is_fusable = {
        'Conv'     : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        'MatMul'   : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        'BiasAdd'  : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        'Add'      : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        'ResAdd'   : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        'Multiply' : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        'Relu'     : "BiasAdd|Relu|Sigmoid|Tanh|Exp|Identity|Lrelu|Prelu|.*Pool|Add|Multiply|ResAdd",
        }

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
        # transfer replication info from Conv KNode to ifmaps_file_params
        if fused_ops.has_conv:
            fused_ops.conv_op.ifmaps_file_params.weights_S_dim = fused_ops.conv_op.weights_file_params.file_dims.S
            fused_ops.conv_op.ifmaps_file_params.stride_x = fused_ops.conv_op.stride_x
            fused_ops.conv_op.ifmaps_file_params.stride_y = fused_ops.conv_op.stride_y
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
                    del next_op.prev[j]
                    next_op.prev.append(id_pool_op)
        last_op.next = [id_pool_op]                    
        return id_pool_op

    def walk_ended(self):
        return (self.current_node == None and self.last_split_next_nodes == [])

