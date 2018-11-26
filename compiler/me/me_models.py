"""
Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
"""

"""Properties and methods for each of the engines within TPB
"""

import numpy as np
import copy
from skimage.util.shape import view_as_windows

"""Macros for dumping arrays
"""
def DBG_DUMP_ARRAY(msg, a):
    print (msg, "\n" , a)
    return a

def DBG_DUMP_PSUM_COL(msg, psum, col):
    x = psum[:, col]
    print (msg, "\n" , x)
    return x

""" PE Array properties and methods
"""
class PEArray:

    NUM_ROWS = 128
    NUM_COLS = 64
    PSUM_NUM_BANKS = 4
    MAX_WAVE_SIZE = 256
    num_wave_fp16_mm = 0
    num_wave_uint8_mm = 0
    num_of_ops_executed = 0
    total_pearray_latency_cycles = 0
    total_pearray_wave_elems = 0
    weights_file_params = None
    weights_sb_addr = 0
    weights_for_batch_item = 0

    def __init__(self):
        self.psum_buf = np.zeros((self.PSUM_NUM_BANKS, self.MAX_WAVE_SIZE, self.NUM_COLS), dtype=np.float32)
        self.Tn = 0
        self.next_psum_bank_id = 0

    # Check if weights already loaded. 
    # Returns true if weights are already loaded
    # kaena-421: During execution for batch item other than the first one, 
    # need to check if there's any SBAtomLoad due to eviction when fused-op 
    # is reexecuted (since these states arespreserved across batch item calls).
    # So just return false if batch item is different to force a read of SB 
    def check_loaded_weights(self, file_params, sb_addr, batch_item):
        if file_params == self.weights_file_params\
                and sb_addr == self.weights_sb_addr \
                and batch_item == self.weights_for_batch_item:
            #print("check_loaded_weights: old ", self.weights_file_params, self.weights_sb_addr) 
            return True
        else: 
            self.weights_file_params = file_params
            self.weights_sb_addr = sb_addr
            self.weights_for_batch_item = batch_item
            #print("check_loaded_weights: new ", self.weights_file_params, self.weights_sb_addr) 
            return False

    """Pick psum bank to use, and advance pointer to new bank (cycle through 4 banks), 
    while the old bank is being processed by other engines.
        Args: 
            - dst_is_psum: destination of operation is PSUM; if false, return -1 for psum_bank_id
        Returns:
            - psum_bank_id: 0 to PSUM_NUM_BANKS-1; -1 if PSUM is unused
    """
    def use_psum_bank_and_adv_ptr(self, dst_is_psum):
        psum_bank_id = -1
        if dst_is_psum:
            psum_bank_id = self.next_psum_bank_id
            self.next_psum_bank_id = (self.next_psum_bank_id + 1) % self.PSUM_NUM_BANKS
        return psum_bank_id

    def extract_psum (self, psum_bank, start_entry, num_entries):
        assert(start_entry < self.MAX_WAVE_SIZE)
        assert(psum_bank >= 0)
        assert(psum_bank < self.PSUM_NUM_BANKS)
        #assert((start_entry+num_entries) < self.MAX_WAVE_SIZE)
        return self.psum_buf[psum_bank, start_entry:start_entry+num_entries, :]

    def write_psum (self, psum_bank, start_entry, num_entries, psum_temp):
        assert(psum_bank >= 0)
        assert(psum_bank < self.PSUM_NUM_BANKS)
        assert(start_entry < self.MAX_WAVE_SIZE)
        #assert((start_entry+num_entries) < self.MAX_WAVE_SIZE)
        self.psum_buf[psum_bank, start_entry:start_entry+num_entries, :] = psum_temp

    # Do wave fp16->fp32 matrix-multiply        
    #   packed_ifmaps: must be 256x128 matrix, float16
    #   packet_weights: must be 128x64 matrix, float16
    #   psum_bank: the PSUM bank number to write result to
    #   psum_add: if True, add to PSUM value in buffer; if False, replace with new value
    def wave_fp16_mm(self, packed_ifmaps, packet_weights, psum_bank, psum_add):
        #assert (packed_ifmaps.shape == (self.MAX_WAVE_SIZE, self.NUM_ROWS))
        #assert (packet_weights.shape == (self.NUM_ROWS, self.NUM_COLS))
        assert(psum_bank >= 0)
        assert(psum_bank < self.PSUM_NUM_BANKS)
        a = packed_ifmaps.astype(np.float32)
        b = packet_weights.astype(np.float32)
        #print(a.dtype, b.dtype, self.psum_buf.dtype)
        result = np.matmul(a, b)
        shape = result.shape
        if (psum_add):
            self.psum_buf[psum_bank][0:shape[0], 0:shape[1]] += result
        else:            
            self.psum_buf[psum_bank] = np.zeros((self.MAX_WAVE_SIZE, self.NUM_COLS), dtype=np.float32)
            self.psum_buf[psum_bank][0:shape[0], 0:shape[1]] = result
        self.num_wave_fp16_mm += 1

    # Do wave uint8->int32 matrix-multiply
    #   packed_ifmaps: must be 256x128 matrix, float16
    #   packet_weights: must be 128x64 matrix, float16
    #   psum_bank: the PSUM bank number to write result to
    #   psum_add: if True, add to PSUM value in buffer; if False, replace with new value
    def wave_uint8_mm(self, packed_ifmaps, packet_weights, psum_bank, psum_add, offset_ifmaps, offset_weights):
        #assert (packed_ifmaps.shape == (self.MAX_WAVE_SIZE, self.NUM_ROWS))
        #assert (packet_weights.shape == (self.NUM_ROWS, self.NUM_COLS))
        assert(psum_bank >= 0)
        assert(psum_bank < self.PSUM_NUM_BANKS)
        a = packed_ifmaps.astype(np.int32)
        b = packet_weights.astype(np.int32)
        #print(a.dtype, b.dtype, self.psum_buf.dtype)
        result = np.matmul(a - offset_ifmaps, b - offset_weights)
        shape = result.shape
        if psum_add:
            self.psum_buf[psum_bank][0:shape[0], 0:shape[1]] += result
        else:
            self.psum_buf[psum_bank] = np.zeros((self.MAX_WAVE_SIZE, self.NUM_COLS), dtype=np.int32)
            self.psum_buf[psum_bank][0:shape[0], 0:shape[1]] = result
        self.num_wave_uint8_mm += 1

    # Unpack the OFMAPs in columns to add to psum in strided pattern (for conv_tranpose/deconv)
    #   packed_ofmaps: Packed OFMAPs in columns 
    #   pewave: current output tile wave, IDed by [n_id, m_id, h_id, w_id, c_id, r_id, s_id]
    #   psum_data: PSUM data to accumulate unpacked OFMAPs (this will be modified and returned)
    #   start_tensor_calc: True: replace PSUM contents; False: add to PSUM content
    #   return: a 256x128 array
    def unpack_wave_ofmaps_deconv(self, packed_ofmaps, ifmap_pewave, ofmap_pewave, psum_data, start_tensor_calc, stride):
        out_array = psum_data
        # remember to extract IFMAPs starting at r_id, s_id (which should be zero for non-conv op)
        self.psum_bank_offset = 0
        # for pooling, the "row" below actually means output columns
        #DBG_DUMP_PSUM_COL("packed_ofmaps:", packed_ofmaps, 0)
        #DBG_DUMP_PSUM_COL("out_array:", out_array, 0)
        ifmap_subtile_dim2d = ifmap_pewave.subtile_rect.dim2d
        ofmap_subtile_dim2d = ofmap_pewave.subtile_rect.dim2d
        ofmap_tile_dim2d = ofmap_pewave.tile.tile_rect.dim2d
        ofmap_subtile_offset = ofmap_pewave.subtile_rect.get_offset_from(ofmap_pewave.tile.tile_rect)
        #print("Deconv PSUM offset: ", ofmap_subtile_offset)
        #print("Deconv PSUM offset: ", ofmap_pewave.subtile_psum_offset)
        for z in range(ofmap_pewave.tile.Tn):
            for j in range(ofmap_pewave.tile.channel_start, ofmap_pewave.tile.channel_stop):
                result_subtile_data_tmp = (packed_ofmaps[      z * ifmap_subtile_dim2d.y * ifmap_subtile_dim2d.x 
                                                        : (z+1) * ifmap_subtile_dim2d.y * ifmap_subtile_dim2d.x, j - ofmap_pewave.tile.channel_start])
                result_subtile_data = result_subtile_data_tmp.reshape((ifmap_subtile_dim2d.y, ifmap_subtile_dim2d.x))
                # NCHW
                ofmap_tile_tmp = psum_data[  z * ofmap_tile_dim2d.y * ofmap_tile_dim2d.x 
                                       : (z+1) * ofmap_tile_dim2d.y * ofmap_tile_dim2d.x, j - ofmap_pewave.tile.channel_start]
                ofmap_tile = ofmap_tile_tmp.reshape((ofmap_tile_dim2d.y, ofmap_tile_dim2d.x))
                if start_tensor_calc:
                    ofmap_tile[0 : ofmap_tile_dim2d.y, 
                               0 : ofmap_tile_dim2d.x] = np.zeros((ofmap_tile_dim2d.y, ofmap_tile_dim2d.x), dtype=np.float32)
                    #print(ofmap_tile, result_subtile_data)
                    ofmap_tile[ofmap_subtile_offset.y : ofmap_subtile_offset.y + ofmap_subtile_dim2d.y : stride.y, 
                               ofmap_subtile_offset.x : ofmap_subtile_offset.x + ofmap_subtile_dim2d.x : stride.x] = result_subtile_data 
                else:
                    #print(ofmap_tile, result_subtile_data)
                    ofmap_tile[ofmap_subtile_offset.y : ofmap_subtile_offset.y + ofmap_subtile_dim2d.y : stride.y, 
                               ofmap_subtile_offset.x : ofmap_subtile_offset.x + ofmap_subtile_dim2d.x : stride.x] += result_subtile_data 
                out_array[      z * ofmap_tile_dim2d.y * ofmap_tile_dim2d.x
                          : (z+1) * ofmap_tile_dim2d.y * ofmap_tile_dim2d.x, j - ofmap_pewave.tile.channel_start] = ofmap_tile.flatten()
        return out_array

"""Pooling properties and methods
"""
class Pool:

    def resadd(self, array_a, array_b):
        result = copy.copy(array_a)
        num_chan = array_b.shape[1]
        result[:, 0:num_chan] += array_b
        return result

    def sub(self, array_a, array_b):
        result = copy.copy(array_a)
        num_chan = array_b.shape[1]
        result[:, 0:num_chan] -= array_b
        return result

    def multiply(self, array_a, array_b):
        result = copy.copy(array_a)
        num_chan = array_b.shape[1]
        result[:, 0:num_chan] *= array_b
        return result

    def maximum(self, array_a, array_b):
        result = copy.copy(array_a)
        num_chan = array_b.shape[1]
        result[:, 0:num_chan] = np.maximum(array_a[:, 0:num_chan], array_b)
        return result

    def minimum(self, array_a, array_b):
        result = copy.copy(array_a)
        num_chan = array_b.shape[1]
        result[:, 0:num_chan] = np.minimum(array_a[:, 0:num_chan], array_b)
        return result

    def clipbyvalue(self, array_a, min_val, max_val):
        return np.clip(array_a, min_val, max_val)

    """Dequantize uint8 or int32 into float32
    quantization equation: real_value = scale * (quantized_value - zero_point)
    """
    def dequantize(self, quantized_value, dequant_scale, zero_point):
        return (dequant_scale * (quantized_value - zero_point)).astype(np.float32)

    """Quantize float32 into uint8
    quantization equation: real_value = scale * (quantized_value - zero_point)
    """
    def quantize_uint8(self, real_value, quant_scale, zero_point):
        return (quant_scale * real_value + zero_point).astype(np.uint8)

    def pool(self, type, in_array, stride, pool_window, Tn, ifmap_tilex_sz, ifmap_tiley_sz, ofmap_tilex_sz, ofmap_tiley_sz):
        num_cols = in_array.shape[1]
        # view_as_windows needs in_array to be in the same dimension as window_shape
        # need to make sure the third dimension of stride_shape to be '1' since that is the column direction
        #print("ifmap_tilex_sz ", ifmap_tilex_sz, " ifmap_tiley_sz ", ifmap_tiley_sz)
        input_tilex_with_pad = ofmap_tilex_sz * stride.x + pool_window.x - stride.x
        input_tiley_with_pad = ofmap_tiley_sz * stride.y + pool_window.y - stride.y
        input_tile_with_pad_sz = input_tilex_with_pad*input_tiley_with_pad
        tile_array = np.empty((input_tiley_with_pad, input_tilex_with_pad))
        tile_array[:] = -np.inf  # set all padding values to -inf to allow only actual tile values to be analyzed
        ifmap_tile_sz = ifmap_tilex_sz*ifmap_tiley_sz
        ofmap_tile_sz = ofmap_tilex_sz*ofmap_tiley_sz
        pool_result = np.zeros((ofmap_tile_sz * Tn, num_cols))
        for j in range(Tn):
            for i in range(num_cols):
                tile_array[0:ifmap_tiley_sz, 0:ifmap_tilex_sz] = in_array[j*ifmap_tile_sz : (j+1)*ifmap_tile_sz, i].reshape(ifmap_tiley_sz, ifmap_tilex_sz) # ignoring Tn for now
                window_shape = (pool_window.y, pool_window.x)
                stride_shape = (stride.y, stride.x)
                pool_result_temp = view_as_windows(tile_array, window_shape, stride_shape)
                if (type == "MaxPool"):
                    pool_result[j*ofmap_tile_sz : (j+1)*ofmap_tile_sz, i] = pool_result_temp.max(axis=(2,3)).reshape(-1)
                elif (type == "AvgPool"):                    
                    pool_result[j*ofmap_tile_sz : (j+1)*ofmap_tile_sz, i] = pool_result_temp.mean(axis=(2,3)).reshape(-1)
                else:                    
                    print("ERROR: unknown type %s Pool.pool"%type)
                    exit(-1)
        return pool_result

    def pool2(self, type, in_array, stride, pool_window, Tn, ifmap_tilex_sz, ifmap_tiley_sz, ofmap_tilex_sz, ofmap_tiley_sz):
        num_cols = in_array.shape[1]
        ifmap_tile_sz = ifmap_tilex_sz*ifmap_tiley_sz
        ofmap_tile_sz = ofmap_tilex_sz*ofmap_tiley_sz
        pool_result = np.zeros((Tn, num_cols, ofmap_tiley_sz, ofmap_tilex_sz))
        for j in range(Tn):
            for i in range(num_cols):
                tile_array = in_array[j, i]
                window_shape = (pool_window.y, pool_window.x)
                stride_shape = (stride.y, stride.x)
                pool_result_temp = view_as_windows(tile_array, window_shape, stride_shape)
                if (type == "MaxPool"):
                    pool_result[j, i] = pool_result_temp.max(axis=(2,3))
                elif (type == "AvgPool"):                    
                    pool_result[j, i] = pool_result_temp.mean(axis=(2,3))
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

    def tensor_scalar_op(self, op_type, in_array, scalar):
        if op_type == 'Add':
            return in_array + scalar
        elif op_type == 'Multiply':
            return in_array * scalar
        elif op_type == 'Maximum':
            return np.maximum(in_array, scalar)
        elif op_type == 'Minimum':
            return np.minimum(in_array, scalar)


"""Bias-Add and Activate properties and methods
"""
class BiasAddAct:

    def biasadd(self, in_array, bias_array, scale = 1.0):
        in_array = scale * in_array
        result = in_array
        num_chan = bias_array.shape[0]
        result[:, 0:num_chan] += bias_array
        return result

    def act(self, type, in_array, alpha = 1.0, scale = 1.0):
        in_array = scale * in_array
        if (type == 'Relu'):
            return self.__relu(in_array)
        elif (type == 'Lrelu'):
            return np.maximum(in_array, alpha*in_array)
        elif (type == 'Sigmoid'):
            return 1/(1 + np.exp(-in_array))
        elif (type == 'Exp'):
            return np.exp(in_array)
        elif (type == 'Tanh'):
            return np.tanh(in_array)
        elif (type == 'Softplus'):
            return np.log(1 + np.exp(in_array))
        elif (type == 'Sqrt'):
            return np.sqrt(in_array)
        else:
            print("ERROR BiasAddAct.act: unrecognized activation type %s"%type)
            exit(-1)

    def __relu(self, in_array):
        return np.maximum(np.zeros(in_array.shape, dtype=in_array.dtype), in_array)

