import json
import os
import re
import numpy as np
import argparse

#kgraph_file = os.environ['KAENA_PATH'] + "/compiler/tffe/rundir/0-1conv0/trivnet_compiler.json"

def ceildiv(a,b):
    return (a//b) + (a%b != 0)

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
    # do striding here?
    def extract_psum (self, psum_bank, start_entry, num_entries, stride_x, stride_y):
        assert(start_entry < self.MAX_WAVE_SIZE)
        #assert((start_entry+num_entries) < self.MAX_WAVE_SIZE)
        return self.psum_buf[psum_bank, start_entry:start_entry+num_entries, :]
    # Do wave fp16->fp32 matrix-multiply        
    #   packed_ifmaps: must be 256x128 matrix, float16
    #   packet_weights: must be 128x64 matrix, float16
    #   psum_bank: the PSUM bank number to write result to
    #   psum_add: if True, add to PSUM value in buffer; if False, replace with new value
    def wave_fp16_mm(self, packed_ifmaps, packet_weights, psum_bank, psum_add):
        assert (packed_ifmaps.shape == (256,128))
        assert (packet_weights.shape == (128,64))
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
        return np.abs(in_array)

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
    # Outputs:
    #   Tn: number of C IFMAPs to roll into a wave
    #   n: number of Tn*C IFMAPs chunks to process
    #   c: number of IFMAP folds
    #   num_tiles: number of OFMAP tiles
    #   m: number of OFMAP folds
    def compute_mm_loops(self, N, C, H, W, M, pad_north, pad_south, pad_west, pad_east):
        self.N, self.C, self.H, self.W, self.M = N, C, H, W, M
        self.pad_north, self.pad_south = pad_north, pad_south
        self.pad_west, self.pad_east = pad_west, pad_east
        # compute the IFMAP folds
        self.c = ceildiv(C, self.pearray.NUM_ROWS)
        # compute the OFMAP folds
        self.m = ceildiv(M, self.pearray.NUM_COLS)
        # computing the input map tiling       
        self.num_tiles = 1
        self.HW = H*W
        self.n = 1
        self.Tn = 1
        if (self.HW >= self.pearray.MAX_WAVE_SIZE):
            # for now, send in noodles that span width of IFMAP, which has implications for pooling
            self.num_tiles = ceildiv(H*W, self.pearray.MAX_WAVE_SIZE)
        elif (self.HW < self.pearray.MAX_WAVE_SIZE):
            self.Tn = self.pearray.MAX_WAVE_SIZE // self.HW
            if (self.Tn > self.N):
                self.Tn = self.N
            self.n = ceildiv(self.N, self.Tn)
        print("n=%d, Tn=%d, c=%d, num_tiles=%d, m=%d"%(self.n, self.Tn, self.c, self.num_tiles, self.m))
        #return [self.n, self.Tn, self.c, self.num_tiles, self.m]
    
    # Pack the IFMAPs in columns to create a PE-Array IFMAPs input for a particular wave number
    #   ifmaps: IFMAPs in NCHW format
    #   wave_id: current wave ID, [n_id, m_id, tile_id, c_id, r_id, s_id]
    #   return: a 256x128 array
    def pack_wave_ifmaps(self, ifmaps, wave_id):
        out_array = np.zeros((256, 128))
        # remember to extract IFMAPs starting at r_id, s_id (which should be zero for non-conv op)
        # also need to add zeros for padding
        return out_array

    # Pack the conv weights in columns to create a PE-Array weights array for a particular wave number
    #   weights: conv weights in MCRS format
    #   wave_id: current wave ID, [n_id, m_id, tile_id, c_id, r_id, s_id]
    #   return: a 128x64 array
    def pack_wave_conv_weights(self, weights, wave_id):
        out_array = np.zeros((128, 64))
        return out_array
    
    # Execute conv and other operations in list: for each op, load parameters and perform op with input
    def execute_conv_ops(self, op_list, inputs):
        self.inputs = inputs
        assert (op_list[0]['layer_type'] == 'Conv')
        # get weights from file
        weights = np.load(op_list[0]['kernel_file'])
        M, C, R, S = weights.shape
        # stride: ignore stride_batch and stride_depth for now
        stride_x = op_list[0]['stride'][1]
        stride_y = op_list[0]['stride'][2]
        # initial values
        psum_bank = 0
        psum_add = False                               
        wave_size = max(self.HW, self.pearray.MAX_WAVE_SIZE)
        result = np.zeros((self.N, self.M, self.H, self.W))
        # wave loop ordering scheme: nmtcRS
        for n_id in range(self.n):
            for m_id in range(self.m):
                for tile_id in range(self.num_tiles):
                    fullwave_id = [n_id, m_id, tile_id]
                    # loops for constructing a tile
                    for c_id in range(self.c):
                        for r_id in range(R):
                            for s_id in range(S):
                                wave_id = [n_id, m_id, tile_id, c_id, r_id, s_id]
                                pearray_packed_ifmaps = self.pack_wave_ifmaps(inputs, wave_id)
                                pearray_packed_weights = self.pack_wave_conv_weights(weights, wave_id)
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
                                relu_result = self.activate.relu(self.pearray.extract_psum(psum_bank, 0, wave_size, stride_x, stride_y))
                                for j in range(M):
                                    result[n_id, j] = (relu_result[0:self.HW, j]).reshape((self.H, self.W))
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
        kgraph = json.load(open(args.kgraph))
    except Exception as e:
        print(e)
        exit(-1)

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
        tpb.compute_mm_loops(iN, iC, iH, iW, convC, pad_north, pad_south, pad_west, pad_east)

        inputs = np.load(input_layer['ref_file'])
        #weights = np.load(layers[1]['kernel_file'])
        results = tpb.execute_conv_ops(op_list, inputs)
        outputs = np.load(output_layer['ref_file'])
        compared = results == outputs

        print("Input IFMAPS:\n", inputs)
        print("Computed OFMAPS:\n", results)
        print("Expected OFMAPS:\n", outputs)
        print("Compared OFMAPS:\n", compared)
        if (not compared.all()):
            print("FAILED: computed IFMAPS is not equal to expected IFMAPS!")
        else:
            print("PASSED")
    else:        
        print("ERROR: the first operation should be Conv")
        exit(-1)

