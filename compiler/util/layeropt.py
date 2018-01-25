import json
import os
import math
import re
import numpy as np
import argparse

np.set_printoptions(threshold=np.nan)

#kgraph_file = os.environ['KAENA_PATH'] + "/compiler/tffe/rundir/0-1conv0/trivnet_compiler.json"

def ceildiv(a,b):
    return (a//b) + (a%b != 0)

# Wave ID
class WaveID:
    def __init__(self, n_id, m_id, h_id, w_id, c_id, r_id, s_id):
        self.n_id, self.m_id, self.h_id, self.w_id = n_id, m_id, h_id, w_id
        self.c_id, self.r_id, self.s_id = c_id, r_id, s_id
    def show(self):
        return [self.n_id, self.m_id, self.h_id, self.w_id, self.c_id, self.r_id, self.s_id]

# PE Array properties and methods
class PEArray:
    NUM_ROWS = 128
    NUM_COLS = 64
    PSUM_BUFFER_DEPTH = 1024
    PSUM_NUM_BANKS = 4
    PSUM_NUM_PARTITIONS = 64
    MAX_WAVE_SIZE = 256
    def __init__(self):
        self.psum_buf = np.zeros((self.PSUM_NUM_BANKS, self.MAX_WAVE_SIZE, self.NUM_COLS))
    def trig_tile_done(self, fullwave_id):
        pass
    def extract_psum (self, psum_bank, start_entry, num_entries):
        assert(start_entry < self.MAX_WAVE_SIZE)
        #assert((start_entry+num_entries) < self.MAX_WAVE_SIZE)
        return self.psum_buf[psum_bank, start_entry:start_entry+num_entries, :]
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
    def wait_tile_done(self, fullwave_id):
        pass
    def avg(self, in_array):
        return in_array
    def max(self, in_array):
        return in_array

# Bias-Add and Activate properties and methods
class BiasAddAct:
    def wait_tile_done(self, fullwave_id):
        pass
    def biasadd(self, in_array, bias_array):
        return in_array
    def relu(self, in_array):
        return np.maximum(np.zeros(in_array.shape), in_array)

# The TPB scheduler has access to:
#   PEArray 
#   Pool 
#   BiasAddAct 
class TPBSched:
    def __init__(self):
        self.pearray = PEArray()
        self.pool = Pool()
        self.activate = BiasAddAct()

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
        self.h = 1
        self.w = 1
        self.n = 1
        self.Tn = 1
        # TODO: determine tile size and aspect ratio taking into account pooling size
        self.ofmap_tilex_sz = min(self.W, int(math.sqrt(self.pearray.MAX_WAVE_SIZE)))
        self.ofmap_tiley_sz = min(self.H, int(math.sqrt(self.pearray.MAX_WAVE_SIZE)))
        if (self.HW >= self.pearray.MAX_WAVE_SIZE):
            # for now, send in noodles that span width of IFMAP, which has implications for pooling
            #self.num_tiles = ceildiv(H*W, self.pearray.MAX_WAVE_SIZE)
            # for now, send in 16x16 tiles, which has implications for pooling
            # for now, use h/w for both IFMAP and OFMAP
            self.h = ceildiv(E, self.ofmap_tiley_sz)
            self.w = ceildiv(F, self.ofmap_tilex_sz)
        elif (self.HW < self.pearray.MAX_WAVE_SIZE):
            self.Tn = self.pearray.MAX_WAVE_SIZE // self.HW
            if (self.Tn > self.N):
                self.Tn = self.N
            self.n = ceildiv(self.N, self.Tn)
        print("n=%d, Tn=%d, c=%d, h=%d, w=%d, m=%d, ofmap_tilex_sz=%d, ofmap_tiley_sz=%d"%(self.n, self.Tn, self.c, self.h, self.w, self.m, self.ofmap_tilex_sz, self.ofmap_tiley_sz))
    
    # Pack a single flattened tiled IFMAP for a particular wave ID
    #   ifmap: IFMAP in NCHW format
    #   wave_id: current wave ID, [n_id, m_id, h_id, w_id, c_id, r_id, s_id]
    #   return: a 256 array
    def pack_wave_ifmap(self, ifmap, wave_id):
        out_array = np.zeros(self.pearray.MAX_WAVE_SIZE)
        # pack ifmap to 0 index, note the ifmap length for general case (not boundary, where ifmap length can be shorter)
        for i in range(self.Tn):
            for x in range(self.ofmap_tilex_sz):
                for y in range(self.ofmap_tiley_sz):
                    ifmap_tilex = (x * self.stride) + wave_id.s_id - self.pad_west
                    ifmap_tiley = (y * self.stride) + wave_id.r_id - self.pad_north
                    ifmap_addr = i * self.ofmap_tilex_sz * self.ofmap_tiley_sz + y * self.ofmap_tilex_sz + x
                    if (ifmap_tilex < 0 or ifmap_tilex >= self.W):
                        out_array[ifmap_addr] = 0
                    elif (ifmap_tiley < 0 or ifmap_tiley >= self.H):
                        out_array[ifmap_addr] = 0
                    else:
                        out_array[ifmap_addr] = ifmap[(wave_id.n_id * self.Tn) + i, ifmap_tiley, ifmap_tilex]
                    # TODO: for optimization, don't send padding zeros by using the following clipped version
                    #ifmap_tilex_clip = max(0, min(ifmap_tilex, self.W-1))
                    #ifmap_tiley_clip = max(0, min(ifmap_tiley, self.H-1))
        #print(out_array)                    
        return out_array

    # Pack the IFMAPs in columns to create a PE-Array IFMAPs input for a particular wave number
    #   ifmaps: IFMAPs in NCHW format
    #   wave_id: current wave ID, [n_id, m_id, h_id, w_id, c_id, r_id, s_id]
    #   return: a 256x128 array
    def pack_wave_ifmaps(self, ifmaps, wave_id):
        out_array = np.zeros((self.pearray.MAX_WAVE_SIZE, self.pearray.NUM_ROWS))
        # remember to extract IFMAPs starting at r_id, s_id (which should be zero for non-conv op)
        # also need to add zeros for padding
        for row in range(self.pearray.NUM_ROWS):
            if (wave_id.c_id * self.pearray.NUM_ROWS + row < self.C):
                out_array[:,row] = self.pack_wave_ifmap(ifmaps[:, wave_id.c_id * self.pearray.NUM_ROWS + row], wave_id)
                #print(out_array[:,row])
            else:
                break
        return out_array

    # Pack the conv weights in columns to create a PE-Array weights array for a particular wave number
    #   weights: conv weights in MCRS format
    #   wave_id: current wave ID, [n_id, m_id, h_id, w_id, c_id, r_id, s_id]
    #   return: a 128x64 array
    def pack_wave_conv_weights(self, weights, wave_id):
        out_array = np.zeros((self.pearray.NUM_ROWS, self.pearray.NUM_COLS))
        for row in range(self.pearray.NUM_ROWS):
            if (wave_id.c_id * self.pearray.NUM_ROWS + row >= self.C):
                break
            else :
                for col in range(self.pearray.NUM_COLS):
                    if (wave_id.m_id * self.pearray.NUM_COLS + col >= self.M):
                        break
                    else:
                        out_array[row, col] = weights[wave_id.m_id * self.pearray.NUM_COLS + col,
                                                      wave_id.c_id * self.pearray.NUM_ROWS + row, 
                                                      wave_id.r_id, 
                                                      wave_id.s_id]
        return out_array
    
    # Execute conv and other operations in list: for each op, load parameters and perform op with input
    def execute_conv_ops(self, op_list, inputs):
        self.inputs = inputs
        assert (op_list[0]['layer_type'] == 'Conv')
        # get weights from file
        weights = np.load(op_list[0]['kernel_file'])
        M, C, R, S = weights.shape
        # initial values
        psum_bank = 0
        psum_add = False                               
        wave_size = max(self.HW, self.pearray.MAX_WAVE_SIZE)
        result = np.zeros((self.N, self.M, self.H, self.W))
        # wave loop ordering scheme: nmtcRS
        for n_id in range(self.n):
            for m_id in range(self.m):
                for h_id in range(self.h):
                    for w_id in range(self.w):
                        fullwave_id = [n_id, m_id, h_id, w_id]
                        # loops for constructing a tile
                        for c_id in range(self.c):
                            for r_id in range(R):
                                for s_id in range(S):
                                    wave_id = WaveID(n_id, m_id, h_id, w_id, c_id, r_id, s_id)
                                    pearray_packed_ifmaps = self.pack_wave_ifmaps(inputs, wave_id)
                                    pearray_packed_weights = self.pack_wave_conv_weights(weights, wave_id)
                                    #print("\npearray_packed_ifmaps", wave_id.show(), "\n", pearray_packed_ifmaps)
                                    #print("\npearray_packed_weights", wave_id.show(), "\n", pearray_packed_weights)
                                    self.pearray.wave_fp16_mm(pearray_packed_ifmaps, pearray_packed_weights, psum_bank, psum_add)
                                    # after the first wave, subsequent waves results are added to partial sums in buffer
                                    if (not psum_add):
                                        psum_add = True
                        # tile is done                                    
                        self.pearray.trig_tile_done(fullwave_id)
                        # go through the remaining operations
                        if (len(op_list)>1):
                            for i in range(1, len(op_list)):
                                if (op_list[i]['layer_type'] == 'Relu'):
                                    self.activate.wait_tile_done(fullwave_id)
                                    relu_result = self.activate.relu(self.pearray.extract_psum(psum_bank, 0, wave_size))
                                    tile_x_start = wave_id.w_id * self.ofmap_tilex_sz
                                    tile_y_start = wave_id.h_id * self.ofmap_tiley_sz
                                    tile_height = self.ofmap_tiley_sz
                                    tile_width = self.ofmap_tilex_sz
                                    if ((wave_id.h_id+1) * self.ofmap_tiley_sz - 1 >= self.H):
                                        tile_height = self.H - tile_y_start
                                    if ((wave_id.w_id+1) * self.ofmap_tilex_sz - 1 >= self.W):
                                        tile_width = self.W - tile_x_start
                                    tile_pixels = tile_height * tile_width
                                    for j in range(M):
                                        result[n_id, j, tile_y_start:tile_y_start+tile_height, tile_x_start:tile_x_start+tile_width]\
                                            = (relu_result[0:tile_pixels, j]).reshape((tile_height, tile_width))
                                else:
                                    print ("%s is currently not yet implemented"%op_list[i]['layer_type'])
                        # Advance to new bank, while the old bank is being processed                                        
                        psum_bank += 1
                        psum_add = False                                
        return result                    

# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kgraph", help="K-graph Json file to read")
    args = parser.parse_args()

    try:
        print("\nLoading K-graph %s"%args.kgraph)
        kgraph = json.load(open(args.kgraph))
    except Exception as e:
        print(e)
        exit(-1)

    # get the lowest significant bit
    data_type = kgraph["data_type"]
    #if (data_type == "float16"):

    # collect some information
    layers = kgraph["layers"]
    num_layers = len(layers)
    if (num_layers >= 1):
        input_layer = layers[0]
        output_layer = layers[num_layers-1]
    else:
        print("ERROR: there are no layers!")
        exit(-1)

    # take first layer as input
    if (layers[0]['ofmap_format'] == 'NCHW'):
        iN,iC,iH,iW = layers[0]['ofmap_shape']
        print("Input shape",iN,iC,iH,iW)
    else:
        print("ERROR: don't understand input ofmap_format %s"%layers[0]['ofmap_format'])
        exit(-1)

    # take last layer as output
    if (num_layers >= 1):
        if (layers[num_layers-1]['ofmap_format'] == 'NCHW'):
            oN,oC,oH,oW = layers[num_layers-1]['ofmap_shape']
            print("Output shape",oN,oC,oH,oW)
        else:
            print("ERROR: don't understand output ofmap_format %s"%layers[num_layers-1]['ofmap_format'])
            exit(-1)

    # go through all layers and add the fusable operations
    # TODO: this assumes that the fusable layers are next to each other, and in order in JSON file
    op_list = []    
    for l in layers:
        if (re.search(r"Conv|Add|BiasAdd|Relu|.*Pool", l['layer_type'])) :
            op_list.append(l)
    if (len(op_list) == 0):
        print("ERROR: found zero operations to fuse")
        exit(-1)

    # The first one should be conv
    # TODO: add matrix multiply
    if (re.search(r"Conv", op_list[0]['layer_type'])):
        tpb = TPBSched()
        convN, convC, convH, convW = op_list[0]['ofmap_shape']
        pad_north, pad_south = op_list[0]['padding'][2]
        pad_west, pad_east = op_list[0]['padding'][3]
        # stride: ignore stride_batch and stride_depth for now; also assume stride_x==stride_y
        stride_x = op_list[0]['stride'][1]
        stride_y = op_list[0]['stride'][2]
        assert(stride_x == stride_y)
        tpb.compute_mm_loops(iN, iC, iH, iW, convC, convH, convW, pad_north, pad_south, pad_west, pad_east, stride_x)
        # TODO: add selecting among pre-derived looping schemes
        inputs = np.load(input_layer['ref_file'])
        weights = np.load(layers[1]['kernel_file'])
        results = tpb.execute_conv_ops(op_list, inputs)
        outputs = np.load(output_layer['ref_file'])
        np.allclose(results, outputs, 1/100, 1e-9)

        print("\nInput IFMAPS:\n", inputs)
        print("\nWeights:\n", weights)
        print("\nComputed OFMAPS:\n", results)
        print("\nExpected OFMAPS:\n", outputs)
        if (not np.allclose(results, outputs, 1/100, 1e-9)):
            print("\nFAILED: computed OFMAPS is not equal to expected OFMAPS!\n")
        else:
            print("\nPASSED\n")
    else:        
        print("ERROR: the first operation should be Conv")
        exit(-1)

