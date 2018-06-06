import unittest
import numpy as np
import os.path
from enum import Enum

def ceildiv(x, y):
    return -(-x//y)

def data_type_to_item_sz(data_type):
    return np.dtype(data_type).itemsize

# Class to manage head and tail pointers for circular buffer with two modes:
#   - endzone_sz = 0: Normal mode: advance pointer until capacity, where it wraps to 0
#   - endzone_sz > 0: End-sink mode: advance pointer until endzone, where it wraps to (capacity - endzone_sz)
#   - endzone_sz < 0: No wrap mode: advance pointer until capacity, where it wraps to 0, but there's no overflow error if tail hits head, only warning
class CircbufPtrs():
    TAIL = 0
    HEAD = 1

    def __init__(self, capacity, endzone_sz):
        self.reset_full(capacity, endzone_sz)

    def reset_full(self, capacity, endzone_sz):
        assert(capacity > 0)
        assert(endzone_sz <= capacity)
        self.ptr_in_endzone = [False for i in range(2)]
        self.ptr            = [0     for i in range(2)]
        self.capacity       = capacity
        self.endsink_mode   = endzone_sz > 0
        self.no_wrap_mode   = endzone_sz < 0
        self.endzone_sz     = endzone_sz

    def reset_ptrs(self):
        for i in range(2):
            if self.ptr_in_endzone[i]:
                self.ptr[i]    = self.capacity - self.endzone_sz 
    
    def reset_ptrs_clear_endsink_mode(self):
        self.reset_ptrs()
        self.endsink_mode   = 0

    def get_next(self, ptr_type):
        next_ptr  = self.ptr[ptr_type] + 1
        if (next_ptr == self.capacity):
            next_ptr = 0
            # endsink mode: head pointer wraps in endzone if tail is also in endzone
            if self.endsink_mode:
                if self.ptr_in_endzone[ptr_type]:
                    if self.ptr_in_endzone[1-ptr_type]:
                        next_ptr = self.capacity - self.endzone_sz
                    else:
                        raise RuntimeError("CircbufPtrs end-sink mode: Pointer %d (value %d) is in endzone but pointer %d (value %d) is not, so can't wrap pointer %d (capacity %d, endzone_sz %d)"%(ptr_type, self.ptr[ptr_type], 1-ptr_type, self.ptr[1-ptr_type], ptr_type, self.capacity, self.endzone_sz))  
        return next_ptr

    def advance(self, ptr_type):      
        try:
            self.ptr[ptr_type] = self.get_next(ptr_type)
        except RuntimeError as e: raise e
        if self.endsink_mode and (self.ptr[ptr_type] >= self.capacity - self.endzone_sz):
            self.ptr_in_endzone[ptr_type] = True
        if ptr_type == self.TAIL and self.ptr[ptr_type] == self.ptr[1-ptr_type]:
            if self.no_wrap_mode:
                print("WARNING: CircbufPtrs collision/full in no-wrap mode: tail pointer value %d is hitting head pointer value %d"%(self.ptr[ptr_type], self.ptr[1-ptr_type]))
            else:            
                raise RuntimeError("CircbufPtrs collision/full: tail pointer value %d is hitting head pointer value %d"%(self.ptr[ptr_type], self.ptr[1-ptr_type]))
        return self.ptr[ptr_type]                

    def get(self, ptr_type):        
        return self.ptr[ptr_type]

    def print(self, tag):
        print("DBG %s: capacity %d endsink_mode %d endzone_sz %d head %d head_in_endzone %d tail %d tail_in_endzone %d"
                %(tag, self.capacity, self.endsink_mode, self.endzone_sz, self.ptr[self.HEAD],self.ptr_in_endzone[self.HEAD], self.ptr[self.TAIL], self.ptr_in_endzone[self.TAIL]))

class TestCircbufPtrsMethods(unittest.TestCase):
    def test_advancing(self):
        for capacity in range(200):
            for endzone_sz in range(10):
                #print("capacity %d endzone_sz %d"%(capacity, endzone_sz))
                if (capacity < 1):
                    with self.assertRaises(AssertionError):
                        CircbufPtrs(capacity, endzone_sz)
                elif (endzone_sz > capacity): 
                    with self.assertRaises(AssertionError):
                        CircbufPtrs(capacity, endzone_sz)
                else:
                    test_obj = CircbufPtrs(capacity, endzone_sz)
                    for i in range(250):
                        if (test_obj.get_next(CircbufPtrs.TAIL) == test_obj.get(CircbufPtrs.HEAD)):
                            with self.assertRaises(RuntimeError):
                                self.assertLess (test_obj.advance(CircbufPtrs.TAIL), capacity)
                        else:                                
                            self.assertLess (test_obj.advance(CircbufPtrs.TAIL), capacity)
                        self.assertLess (test_obj.advance(CircbufPtrs.HEAD), capacity)
                        if (endzone_sz > 0):
                            if (test_obj.ptr_in_endzone[CircbufPtrs.TAIL]):
                                self.assertGreaterEqual (test_obj.get(CircbufPtrs.TAIL), capacity - endzone_sz)
                            if (test_obj.ptr_in_endzone[CircbufPtrs.HEAD]):
                                self.assertGreaterEqual (test_obj.get(CircbufPtrs.HEAD), capacity - endzone_sz)
                        self.assertEqual(test_obj.get(CircbufPtrs.HEAD), test_obj.get(CircbufPtrs.TAIL))
                    if (endzone_sz > 0):
                        self.assertEqual(test_obj.ptr_in_endzone[CircbufPtrs.TAIL], True)
                        self.assertEqual(test_obj.ptr_in_endzone[CircbufPtrs.HEAD], True)
                        test_obj.reset_ptrs()
                        self.assertEqual(test_obj.get(CircbufPtrs.TAIL), capacity - endzone_sz)
                        self.assertEqual(test_obj.get(CircbufPtrs.HEAD), capacity - endzone_sz)

    #def test_endsink_exception(self):
    #    for capacity in range(10,100):
    #        for endzone_sz in range(1,10):
    #            #print("capacity %d endzone_sz %d"%(capacity, endzone_sz))
    #            test_obj = CircbufPtrs(capacity, endzone_sz)
    #            for i in range(150):
    #                if ((i%capacity) == capacity-1):
    #                    with self.assertRaises(RuntimeError):
    #                        self.assertLess (test_obj.advance(CircbufPtrs.TAIL), capacity)
    #                else:                            
    #                    self.assertLess (test_obj.advance(CircbufPtrs.TAIL), capacity)


# Class to manage N-dimensional shapes

class ShapeDims():
    supported_dims = set(["N", "H", "W", "C", "M", "R", "S"])

    def __init__(self, format_str, shape_tuple):
        var_list = vars(self)            
        self.dim = {}
        self.axis = {}
        self.format_str = format_str
        self.shape_tuple = shape_tuple
        self.tot_elems = 1
        self.has_M = False
        if (len(format_str) != len(shape_tuple)):
            raise RuntimeError("ERROR ShapeDims: format_str %s doesn't have the same length as shape_tuple %s"%(format_str, str(shape_tuple)))
        for d in self.supported_dims:
            var_list[d] = 1
            var_list[d+"_axis"] = -1
            self.dim[d] = 1
            self.axis[d] = -1
        for i in range(len(format_str)):
            if format_str[i] not in self.supported_dims:
                raise RuntimeError("ERROR ShapeDims: format_str char %s is not supported (supported list %s)"%(format_str[i], ",".join(self.supported_dims)))
            #if format_str[i] in self.dim:
            #    raise RuntimeError("ERROR ShapeDims: duplicate format_str char %s (format_st %s)"%(format_str[i], format_str))
            var_list[format_str[i]] = shape_tuple[i]
            var_list[format_str[i]+"_axis"] = i
            self.dim[format_str[i]] = shape_tuple[i]
            self.axis[format_str[i]] = i
            self.tot_elems *= shape_tuple[i]
            if format_str[i] == 'M': 
                self.has_M = True

    def check_format_str(self, format_str):
        if (format_str != self.format_str):
            raise RuntimeError("ERROR ShapeDims: format_str %s doesn't match initialized format %s"%(format_str, self.format_str))

    def check_shape(self, shape_tuple):
        if (shape_tuple != self.shape_tuple):
            raise RuntimeError("ERROR ShapeDims: shape_tuple %s doesn't match %s"%(str(shape_tuple), str(self.shape_tuple)))

    # from a window (subshape), extract start/end offsets
    def get_startend_for_subshape(self, subshape_tuple):        
        pass

# Class to manage file-related parameters
class FileParams():
    def __init__(self, file_id, file_name, file_dims, data_type, chunk_sz_limit, pearray_params, op_params, args=None):
        self.layer_name = "DEPRECATED"
        self.final_layer_ofmap = False
        self.file_id = file_id
        self.file_name = file_name
        self.file_loaded = False
        self.file_dims = file_dims
        self.dram_data = None
        self.file_sz = 0
        self.file_addr_skip_per_batch_item = 0
        self.item_sz = data_type_to_item_sz(data_type)
        self.data_type = data_type
        self.chunk_sz_limit = chunk_sz_limit
        self.chunk_sz = -1
        self.tot_partition_usage_sz = -1
        self.tot_num_chunks = -1
        self.fmap_data_len = -1
        self.fmap_num_chunks = -1
        self.fmap_channels_folds = 1
        self.fmap_last_chunk_sz = -1
        self.batch_item_partition_usage_sz = -1
        self.batch_item_partition_usage_sz_rounded = -1
        self.batch_item_num_chunks = -1
        self.mapped_params = None
        self.file_addr_skip_per_fmap_fold = -1
        # Keeping a list of ops that writes to the same OFMAP, so that all can be transfered if OFMAP file is combined with another
        self.writers_of_shared_fmap = []
        self.readers_of_shared_fmap = []
        self.compute_params(pearray_params, op_params, args)

    def load_file(self):
        if (self.file_name != None):
            if not self.file_loaded:
                # initialize file if it doesn't exist
                if not os.path.isfile(self.file_name):
                    self.dram_data = np.zeros(self.file_dims.shape_tuple, dtype=self.data_type)
                    try:
                        np.save(self.file_name, self.dram_data)
                    except:
                        raise RuntimeError("Cannot save numpy file %s"%(self.file_name))
                else:
                    try:
                        self.dram_data = np.load(self.file_name)
                    except:
                        raise RuntimeError("Cannot load numpy file %s"%(self.file_name))
                self.file_loaded = True                    
                assert(self.dram_data.flags.c_contiguous == True)
                assert(self.dram_data.itemsize == self.item_sz)
                assert(self.dram_data.size == self.file_dims.tot_elems)
                self.file_sz = self.dram_data.size * self.dram_data.itemsize
                self.file_addr_skip_per_batch_item  = self.file_sz // self.file_dims.N
                #print("dram_data.size %d file_dims.tot_elems %d"%(self.dram_data.size, self.file_dims.tot_elems))
            else:
                print("INFO: file %s is already loaded"%(self.file_name))
        else:
            raise RuntimeError("File name is empty")
        return self.dram_data            

    def zero_file(self):
        if (self.file_name != None):
            # clear file even if it exists
            self.dram_data = np.zeros(self.file_dims.shape_tuple, dtype=self.data_type)
            try:
                np.save(self.file_name, self.dram_data)
            except:
                raise RuntimeError("Cannot save numpy file %s"%(self.file_name))
            self.file_loaded = True                    
            assert(self.dram_data.flags.c_contiguous == True)
            assert(self.dram_data.itemsize == self.item_sz)
            assert(self.dram_data.size == self.file_dims.tot_elems)
            self.file_sz = self.dram_data.size * self.dram_data.itemsize
            self.file_addr_skip_per_batch_item  = self.file_sz // self.file_dims.N
            #print("dram_data.size %d file_dims.tot_elems %d"%(self.dram_data.size, self.file_dims.tot_elems))
        else:
            raise RuntimeError("File name is empty")
        return self.dram_data            

    def save_file(self):
        # TODO: reshape if result shape is different from file shape
        try:
            np.save(self.file_name, self.dram_data)
        except:
            raise RuntimeError("Cannot save numpy file %s"%(self.file_name))

    def get_nchw_shape(self):
        return (self.file_dims.N, self.file_dims.C, self.file_dims.H, self.file_dims.W)

    # obtain element address within numpy array
    def ravel_nchw(self, N, C, H, W):
        coord = [0, 0, 0, 0]
        coord[self.file_dims.N_axis] = N
        coord[self.file_dims.C_axis] = C
        coord[self.file_dims.H_axis] = H
        coord[self.file_dims.W_axis] = W
        return int(np.ravel_multi_index(coord, dims=self.file_dims.shape_tuple) * self.item_sz)

    def ravel_crsm(self, C, R, S, M):
        coord = [0, 0, 0, 0]
        coord[self.file_dims.C_axis] = C
        coord[self.file_dims.R_axis] = R
        coord[self.file_dims.S_axis] = S
        coord[self.file_dims.M_axis] = M
        return int(np.ravel_multi_index(coord, dims=self.file_dims.shape_tuple) * self.item_sz)

    # obtain element data within numpy array
    def elem_nchw(self, N, C, H, W):
        coord = [0, 0, 0, 0]
        coord[self.file_dims.N_axis] = N
        coord[self.file_dims.C_axis] = C
        coord[self.file_dims.H_axis] = H
        coord[self.file_dims.W_axis] = W
        return self.dram_data[tuple(coord)]

    def compute_params(self, pearray_params, op_params, args):
        # Single FMAP elem count (unified formula for weights and FMAP)
        fmap_elem_count = self.file_dims.R * self.file_dims.S * self.file_dims.M * self.file_dims.H * self.file_dims.W
        self.fmap_data_len = fmap_elem_count * self.item_sz
        self.stride_x = op_params.stride_x
        self.stride_y = op_params.stride_y
        self.replicate_multiple = op_params.replicate_multiple
        self.weights_S_dim = self.file_dims.S
        # per kaena-85, use noodle shapes for tiles
        # need to guard against small EF and build noodle tile to enable higher state buffer efficiency
        self.fmap_full_tilex_sz = min(self.file_dims.W, pearray_params.MAX_WAVE_SIZE)
        self.fmap_full_tiley_sz = min(self.file_dims.H, pearray_params.MAX_WAVE_SIZE // self.fmap_full_tilex_sz)
        # Chunk (aka atom) size computation for weights
        if self.file_dims.has_M:
            m_data_len = self.file_dims.M * self.item_sz
            sm_data_len = self.file_dims.S * m_data_len
            folding_multiple = (self.file_dims.C // pearray_params.NUM_ROWS) * (self.file_dims.M // pearray_params.NUM_COLS)
            atom_sz_for_computation = self.chunk_sz_limit
            # TODO: simplify to just limiting to 64 output channels
            if (folding_multiple > 16):
                atom_sz_for_computation = self.chunk_sz_limit//4
            if (self.fmap_data_len <= atom_sz_for_computation):
                self.chunk_sz = self.fmap_data_len
            # Map to M or SM to fit into circular-buffer regions exactly
            elif (sm_data_len <= atom_sz_for_computation):
                #multiple = atom_sz_for_computation // sm_data_len
                self.chunk_sz = sm_data_len # * min(self.file_dims.R, multiple)
            elif (m_data_len <= atom_sz_for_computation):
                #multiple = atom_sz_for_computation // m_data_len
                self.chunk_sz = m_data_len  #* min(self.file_dims.S, multiple)
            else:
                self.chunk_sz = atom_sz_for_computation
        else:                
            ifmap_width_data_len = self.file_dims.W * self.item_sz
            # make atom size multiple of IFMAP if IFMAP is smaller than default atom size (CNHW)
            # For NCHW, just use ifmap size as atom size (see rule above: "different FMAPs folds will be in different atoms")
            if (self.fmap_data_len <= self.chunk_sz_limit):
                self.chunk_sz = self.fmap_data_len
            # make atom size multiple of width data length if it is smaller than default atom size
            # For FP32, use initial atom of 2KB to guarantee gapless spaces for 28x28 (without using skip-atoms), when folding is involved
            elif (ifmap_width_data_len <= self.chunk_sz_limit):
                input_fmap_full_tiley_sz = self.fmap_full_tiley_sz * op_params.stride_y
                if (args is not None and args.abstract_mem):
                    self.chunk_sz = ifmap_width_data_len * input_fmap_full_tiley_sz
                else:
                    multiple = self.chunk_sz_limit // ifmap_width_data_len
                    multiple = min(self.file_dims.H, multiple)
                    # eliminate skip atoms by requiring atom size is multiple of tile size 
                    if (input_fmap_full_tiley_sz < multiple):
                        multiple = (multiple//input_fmap_full_tiley_sz) * input_fmap_full_tiley_sz
                    elif (self.fmap_full_tiley_sz < multiple):
                        multiple = (multiple//self.fmap_full_tiley_sz) * self.fmap_full_tiley_sz
                    self.chunk_sz = ifmap_width_data_len * min(self.file_dims.H, multiple)
                    # warn if FMAP size is not multiple of chunk size (i.e. 55x55) where c>=1 addresses don't align to chunks
                    #if (self.fmap_data_len % self.chunk_sz) != 0:
                    #    print("WARNING: FMAP size %d is not a multiple of chunk size %d for shape %s, so c>=1 addresses don't align to chunks!"%(self.fmap_data_len, self.chunk_sz, str(self.file_dims.shape_tuple)))
            else:
                self.chunk_sz = self.chunk_sz_limit
        self.fmap_channels_folds            = ceildiv(self.file_dims.C, pearray_params.NUM_ROWS)  # num of folds (same as lowercase "c" computed elsewhere):
        self.batch_item_partition_usage_sz  = self.fmap_data_len * self.fmap_channels_folds
        self.tot_partition_usage_sz         = self.batch_item_partition_usage_sz * self.file_dims.N
        self.fmap_num_chunks                = ceildiv(self.fmap_data_len, self.chunk_sz)
        self.fmap_count                     = min(self.file_dims.C, pearray_params.NUM_ROWS)
        self.fmap_last_fold_channels        = self.file_dims.C % pearray_params.NUM_ROWS
        if self.fmap_last_fold_channels == 0: 
            self.fmap_last_fold_channels = pearray_params.NUM_ROWS
        self.batch_item_num_chunks          = self.fmap_num_chunks * self.fmap_channels_folds
        self.tot_num_chunks                 = self.batch_item_num_chunks * self.file_dims.N
        self.fmap_last_chunk_sz             = self.fmap_data_len % self.chunk_sz
        if self.fmap_last_chunk_sz == 0:    
            self.fmap_last_chunk_sz         = self.chunk_sz
        self.file_addr_skip_per_fmap_fold   = self.fmap_data_len * min(self.file_dims.C, pearray_params.NUM_ROWS)
        # rounding up usage to make 55x55 takes up 56x56 space for more regular chunk/region divisions
        H_rounded = ((self.file_dims.H + 1) // 2) * 2
        W_rounded = ((self.file_dims.W + 1) // 2) * 2
        self.batch_item_partition_usage_sz_rounded = self.batch_item_partition_usage_sz // self.file_dims.H // self.file_dims.W
        self.batch_item_partition_usage_sz_rounded = self.batch_item_partition_usage_sz_rounded * H_rounded * W_rounded
        print("INFO: file %s shape %s tot_partition_usage_sz %d batch_item_partition_usage_sz %d"%(self.file_name, str(self.file_dims.shape_tuple), self.tot_partition_usage_sz, self.batch_item_partition_usage_sz))

# Class to hold map information related to a file
class MappedParams():
    def __init__(self, N, start_addr, region_sz, num_region_chunks, num_file_chunks_per_batch_item, end_addr, modify_in_place):
        self.start_addr = start_addr
        self.region_sz  = region_sz
        self.num_region_chunks = num_region_chunks
        self.num_file_chunks_per_batch_item = num_file_chunks_per_batch_item
        self.chunk2waveop_map = {}
        #self.chunk_is_mapped = []
        #for i in range(N):
        #    self.chunk_is_mapped.append([False for i in range(num_file_chunks_per_batch_item)])
        self.chunk_is_mapped = [False for i in range(N*num_file_chunks_per_batch_item)]
        self.end_addr = end_addr
        self.modify_in_place = modify_in_place

# Class to represent a morsel of data
class SbMorsel():
    def __init__(self):
        self.file_id = -1
        self.writer_id = -1
        self.reader_id = -1
        self.chunk_id = -1
        self.batch_item = -1

# Class to manage mapping file to SB regions
class FileMapper():
    def __init__(self, sb_partition_sz, data_type):
        self.item_sz         = data_type_to_item_sz(data_type)
        self.sb_partition_sz = sb_partition_sz
        self.data_type       = data_type
        self.file_params_list = {}
        self.morsels = [SbMorsel() for i in range(sb_partition_sz)]

    def check_overlap(self, region_start0, region_sz0, region_start1, region_sz1):
        #print("DBG: checking overlap: region 0 start %d sz %d, region 1 start %d sz %d"%(region_start0, region_sz0, region_start1, region_sz1))
        if (region_start0 <= region_start1):
            if (region_start0 + region_sz0 > region_start1):
                return True
        else:    
            if (region_start1 + region_sz1 > region_start0):
                return True
        return False

    def check_overlap100(self, region_start0, region_sz0, region_start1, region_sz1):
        return (region_start0 == region_start1) and (region_sz0 == region_sz1)

    # File_params contains information about file
    # Start_addr is start address in SB
    # Region_sz is size of region used
    # If region size is less than file size, then wrap-around if wrap_around is True
    # If region size is 0, allow it to expand to file size
    def map_file(self, file_params, start_addr, wrap_around=True, region_sz=0, modify_in_place=False):       
        if file_params is None:
            raise RuntimeError("File information file_params is None")
        if file_params.dram_data is None:
            raise RuntimeError("File data is not loaded")
        if file_params.file_id in self.file_params_list:
            if file_params.mapped_params.start_addr != start_addr:
                print("WARNING: file %s already mapped to start_addr %d; cannot map to new region start_addr %d"%(file_params.file_name, file_params.mapped_params.start_addr, start_addr))
            return file_params.mapped_params.end_addr + file_params.item_sz
        # validity checks
        assert(start_addr >= 0)
        if start_addr < 0 or start_addr >= self.sb_partition_sz:
            raise RuntimeError("Start address %d is less than 0 or exceeds partition size %d"%(start_addr, self.sb_partition_sz))
        # If region size is 0, allow it to expand to file size
        adj_region_sz = file_params.tot_partition_usage_sz if region_sz == 0 else region_sz
        if not wrap_around:
            # If not wrapping around, check that file can fit into alloted region
            if file_params.tot_partition_usage_sz > adj_region_sz:
                raise RuntimeError("File %s size %d is larger than region size %d, and wrap around is not enabled"%(file_params.file_name, file_params.tot_partition_usage_sz, adj_region_sz))
            # Compute number of chunks, including the last odd chunk
            num_region_chunks = file_params.tot_num_chunks
            adj_region_sz     = file_params.tot_partition_usage_sz
        else:
            if adj_region_sz >= file_params.fmap_data_len:
                num_region_fmaps  = adj_region_sz // file_params.fmap_data_len
                num_region_chunks = num_region_fmaps * file_params.fmap_num_chunks
                adj_region_sz     = num_region_fmaps * file_params.fmap_data_len
            else:                
                # If wrapping around and FMAP too big, waste the last odd chunk and just make sure to have chunk_sz pieces
                num_region_chunks = adj_region_sz // file_params.chunk_sz
                adj_region_sz     = num_region_chunks * file_params.chunk_sz

        # Check number of chunks is reasonable            
        if num_region_chunks == 0:                
            raise RuntimeError("Region size %d cannot accomodate chunk size of %d"%(adj_region_sz, file_params.chunk_sz))
        # check end address            
        end_addr = start_addr + adj_region_sz - file_params.item_sz
        if end_addr >= self.sb_partition_sz:
            raise RuntimeError("End address %d falls outside partition size %d"%(end_addr, self.sb_partition_sz))
        # Save mapped information            
        file_params.mapped_params = MappedParams(file_params.file_dims.N, start_addr, adj_region_sz, num_region_chunks, file_params.batch_item_num_chunks, end_addr, modify_in_place)
        # Save file params in a list
        self.file_params_list[file_params.file_id] = file_params
        return end_addr + file_params.item_sz

    def get_chunk_id_from_file_addr(self, file_params, batch_item, addr):
        assert(addr >= 0)
        assert(addr >= batch_item * file_params.file_addr_skip_per_batch_item)
        addr_adj         = addr - batch_item * file_params.file_addr_skip_per_batch_item
        fold_idx         = addr_adj // file_params.file_addr_skip_per_fmap_fold 
        fold_offset      = addr_adj % file_params.file_addr_skip_per_fmap_fold 
        chunk_id_in_fold = fold_offset // file_params.chunk_sz
        chunk_id         = fold_idx * file_params.fmap_num_chunks + chunk_id_in_fold
        chunk_id        += batch_item * file_params.batch_item_num_chunks 
        return chunk_id

    def get_file_addr_from_chunk_id(self, file_params, batch_item, chunk_id):
        assert(chunk_id >= 0)
        assert(chunk_id >= batch_item * file_params.batch_item_num_chunks)
        assert(chunk_id < file_params.tot_num_chunks)
        chunk_id_adj     = chunk_id - batch_item * file_params.batch_item_num_chunks
        fold_idx         = chunk_id_adj // file_params.fmap_num_chunks
        chunk_id_in_fold = chunk_id_adj % file_params.fmap_num_chunks
        fold_offset      = chunk_id_in_fold * file_params.chunk_sz
        addr_adj         = fold_idx * file_params.file_addr_skip_per_fmap_fold + fold_offset
        addr             = addr_adj + batch_item * file_params.file_addr_skip_per_batch_item
        return addr 

    def get_fmap_count_from_chunk_id(self, file_params, batch_item, chunk_id):
        assert(chunk_id >= 0)
        assert(chunk_id >= batch_item * file_params.batch_item_num_chunks)
        assert(chunk_id < file_params.tot_num_chunks)
        chunk_id_adj    = chunk_id - batch_item * file_params.batch_item_num_chunks
        fmap_count      = file_params.fmap_count
        fold_idx        = chunk_id_adj // file_params.fmap_num_chunks
        if file_params.fmap_channels_folds > 1:
            if (fold_idx == file_params.fmap_channels_folds - 1):
                fmap_count = file_params.fmap_last_fold_channels
        return fmap_count

    def get_chunk_offset_from_file_addr(self, file_params, batch_item, addr):
        assert(addr >= 0)
        assert(addr >= batch_item * file_params.file_addr_skip_per_batch_item)
        addr_adj    = addr - batch_item * file_params.file_addr_skip_per_batch_item
        fold_idx    = addr_adj // file_params.file_addr_skip_per_fmap_fold 
        fold_offset = addr_adj % file_params.file_addr_skip_per_fmap_fold 
        chunk_offset = fold_offset % file_params.chunk_sz
        return chunk_offset

    def get_chunk_len_from_chunk_id (self, file_params, batch_item, chunk_id):
        assert(chunk_id >= 0)
        assert(chunk_id >= batch_item * file_params.batch_item_num_chunks)
        assert(chunk_id < file_params.tot_num_chunks)
        chunk_id_adj = chunk_id - batch_item * file_params.batch_item_num_chunks
        chunk_id_in_fold = chunk_id_adj % file_params.fmap_num_chunks
        if chunk_id_in_fold == (file_params.fmap_num_chunks - 1):            
            chunk_len = file_params.fmap_last_chunk_sz
        else:
            chunk_len = file_params.chunk_sz
        return chunk_len

    def get_atom_id_from_file_addr (self, file_params, batch_item, addr):
        assert(addr >= 0)
        #assert(addr < file_params.batch_item_partition_usage_sz)
        chunk_id        = self.get_chunk_id_from_file_addr(file_params, batch_item, addr)
        atom_id         = chunk_id % file_params.mapped_params.num_region_chunks
        return atom_id

    def get_sb_addr_from_chunk_id (self, file_params, batch_item, chunk_id):
        assert(chunk_id >= 0)
        assert(chunk_id >= batch_item * file_params.batch_item_num_chunks)
        assert(chunk_id < file_params.tot_num_chunks)
        chunk_id_offset = chunk_id % file_params.mapped_params.num_region_chunks
        if file_params.mapped_params.region_sz >= file_params.fmap_data_len:
            # if region_sz >= fmap_data_len, earlier computation guarantees that region_sz is multiple of fmap_data_len
            fold_idx         = chunk_id_offset // file_params.fmap_num_chunks
            chunk_id_in_fold = chunk_id_offset % file_params.fmap_num_chunks
            sb_addr          = fold_idx * file_params.fmap_data_len + chunk_id_in_fold * file_params.chunk_sz 
        else:            
            sb_addr         = chunk_id_offset * file_params.chunk_sz 
        sb_addr     += file_params.mapped_params.start_addr            
        return sb_addr

    def get_sb_addr_from_file_addr(self, file_params, batch_item, addr):
        assert(addr >= 0)
        #assert(addr < file_params.batch_item_partition_usage_sz)
        chunk_id        = self.get_chunk_id_from_file_addr(file_params, batch_item, addr)
        chunk_offset    = self.get_chunk_offset_from_file_addr(file_params, batch_item, addr)
        sb_addr         = self.get_sb_addr_from_chunk_id(file_params, batch_item, chunk_id) + chunk_offset
        return sb_addr

    def get_dram_waveop_names(self, file_params, batch_item, lower_addr, upper_addr):
        dram_waveop_names = []
        lower_addr_chunked = self.get_chunk_id_from_file_addr(file_params, batch_item, lower_addr)
        upper_addr_chunked = self.get_chunk_id_from_file_addr(file_params, batch_item, upper_addr)
        for i in range(lower_addr_chunked, upper_addr_chunked+1):
            if (i in file_params.mapped_params.chunk2waveop_map):
                dram_waveop_names.append(file_params.mapped_params.chunk2waveop_map[i]["waveop_name"])
                #if not args.abstract_mem:
                # Only need to load data once (transitive dependency for later matmuls)
                del file_params.mapped_params.chunk2waveop_map[i]
        return dram_waveop_names            

    def write_file_data_region(self, nonload_waveop_id, nonload_waveop_list, file_params, batch_item, start_addr, length, start_at_mid_part):
        assert(batch_item < file_params.file_dims.N)
        assert(length > 0)
        assert(length <= file_params.mapped_params.region_sz)
        assert(start_addr >= 0)
        end_file_addr       = start_addr + length - self.item_sz
        start_sb_addr       = self.get_sb_addr_from_file_addr(file_params, batch_item, start_addr)
        end_sb_addr         = self.get_sb_addr_from_file_addr(file_params, batch_item, end_file_addr)
        start_chunk_id      = self.get_chunk_id_from_file_addr(file_params, batch_item, start_addr)
        end_chunk_id        = self.get_chunk_id_from_file_addr(file_params, batch_item, end_file_addr)
        num_chunks          = end_chunk_id - start_chunk_id + 1
        #print("Writing batch item %d starting at %d for length %d (chunks %d to %d)"%(batch_item, start_addr, length, start_chunk_id, end_chunk_id))
        if num_chunks > file_params.mapped_params.num_region_chunks:
            raise RuntimeError("Number of chunks written %d for start %d length %d is larger than mapped number of chunks %d"%(num_chunks, start_addr, length, file_params.mapped_params.num_region_chunks))
        list_of_writers = []
        list_of_readers = []
        list_of_waveops = []
        for i in range(start_chunk_id, end_chunk_id + 1):
            list_of_writers_per_chunk = []
            list_of_readers_per_chunk = []
            # TODO: fix start_fmap_addr to match start_addr
            start_fmap_addr = self.get_sb_addr_from_chunk_id(file_params, batch_item, i)
            end_fmap_addr = start_fmap_addr + self.get_chunk_len_from_chunk_id(file_params, batch_item, i)
            for j in range(start_fmap_addr, end_fmap_addr, file_params.item_sz):
                sb_addr = j
                if sb_addr >= start_sb_addr and sb_addr <= end_sb_addr:
                    # return list of writers/readers for dependency
                    if self.morsels[sb_addr].writer_id not in list_of_writers_per_chunk:
                        list_of_writers_per_chunk.append(self.morsels[sb_addr].writer_id)
                    if self.morsels[sb_addr].reader_id not in list_of_readers_per_chunk:
                        list_of_readers_per_chunk.append(self.morsels[sb_addr].reader_id)
                    # Evict old owner                    
                    # TODO: remap atom_id back to file chunk ID
                    file_id = self.morsels[sb_addr].file_id
                    chunk_id = self.morsels[sb_addr].chunk_id
                    owner_batch_item = self.morsels[sb_addr].batch_item
                    if file_id in self.file_params_list:
                        if (file_id != file_params.file_id) or (chunk_id != i):
                            #print("INFO: batch item %d: Writer waveop (non-load) ID %d is writing chunk_id %d (addr %d) of file %s (file ID %d), clearing previous owner which is chunk_id %d of file %s (file ID %d)"%(batch_item, nonload_waveop_id, i, sb_addr, file_params.file_name, file_params.file_id, chunk_id, self.file_params_list[file_id].file_name, file_id))
                            self.file_params_list[file_id].mapped_params.chunk_is_mapped[chunk_id] = False
                    elif file_id != -1:
                        raise RuntimeError("File ID %d not found in list of file_params"%(file_id))
                    #print("INFO: batch item %d: Writer waveop (non-load) ID %d is writing chunk_id %d (addr %d) of file %s (file ID %d), clearing previous writer_id %s or reader_id %d, and replacing with writer_id %d"%(batch_item, nonload_waveop_id, i, sb_addr, file_params.file_name, file_params.file_id, self.morsels[sb_addr].writer_id, self.morsels[sb_addr].reader_id, nonload_waveop_id))
                    self.morsels[sb_addr].file_id = file_params.file_id
                    self.morsels[sb_addr].writer_id = nonload_waveop_id
                    self.morsels[sb_addr].reader_id = -1
                    self.morsels[sb_addr].chunk_id = i
                    self.morsels[sb_addr].batch_item = batch_item
                # if data is not in SB, map region
                #if not file_params.mapped_params.chunk_is_mapped[batch_item][i]:
                #    file_params.mapped_params.chunk_is_mapped[batch_item][i] = True
                if not file_params.mapped_params.chunk_is_mapped[i]:
                    file_params.mapped_params.chunk_is_mapped[i] = True
                    #print("INFO: batch item %d: Writer waveop (non-load) ID %d is writing chunk_id %d (start %d, end %d) of file %s"%(batch_item, nonload_waveop_id, i, start_fmap_addr, end_fmap_addr, file_params.file_name))
                #if file_params.dump_to_file:
                #    list_of_accessors = list_of_writers + list_of_readers
                #    prev_waveops = []
                #    if list_of_accessors != []:
                #        list_of_accessors_sorted = sorted(list_of_accessors)
                #        assert(list_of_accessors_sorted[-1] < len(nonload_waveop_list))
                #        if list_of_accessors_sorted[-1] >= 0:
                #            prev_waveops.append(nonload_waveop_list[list_of_accessors_sorted[-1]]['waveop_name'])
                #    sb_addr = file_params.mapped_params.start_addr + atom_id*file_params.chunk_sz
                #    list_of_waveops.append(self.gen_dram_save_waveop(file_params, batch_item, i, sb_addr, prev_waveops)) 
                list_of_writers += list_of_writers_per_chunk                
                list_of_readers += list_of_readers_per_chunk                
        return (list_of_writers, list_of_readers, list_of_waveops)

    # Save data to file 
    def flush_file (self, nonload_waveop_id, nonload_waveop_list, file_params, batch_item):
        nonload_waveop_id_tmp = nonload_waveop_id
        start_chunk_id      = batch_item * file_params.batch_item_num_chunks
        end_chunk_id        = start_chunk_id + file_params.batch_item_num_chunks - 1
        num_chunks          = file_params.batch_item_num_chunks
        if num_chunks > file_params.mapped_params.num_region_chunks:
            raise RuntimeError("Number of chunks written %d is larger than mapped number of chunks %d"%(num_chunks, file_params.mapped_params.num_region_chunks))
        list_of_waveops = []
        for i in range(start_chunk_id, end_chunk_id + 1):
            list_of_writers_per_chunk = []
            list_of_readers_per_chunk = []
            start_fmap_addr = self.get_sb_addr_from_chunk_id(file_params, batch_item, i)
            end_fmap_addr = start_fmap_addr + self.get_chunk_len_from_chunk_id(file_params, batch_item, i)
            for j in range(start_fmap_addr, end_fmap_addr, file_params.item_sz):
                sb_addr = j
                # return list of writers/readers for dependency
                if self.morsels[sb_addr].writer_id not in list_of_writers_per_chunk:
                    list_of_writers_per_chunk.append(self.morsels[sb_addr].writer_id)
                if self.morsels[sb_addr].reader_id not in list_of_readers_per_chunk:
                    list_of_readers_per_chunk.append(self.morsels[sb_addr].reader_id)
                # Evict old owner                    
                file_id = self.morsels[sb_addr].file_id
                chunk_id = self.morsels[sb_addr].chunk_id
                owner_batch_item = self.morsels[sb_addr].batch_item
                if file_id in self.file_params_list:
                    self.file_params_list[file_id].mapped_params.chunk_is_mapped[chunk_id] = False
                    #print("INFO: batch item %d: Reader waveop (non-load) ID %d is reading chunk_id %d (addr %d) of file %s (file ID %d), clearing previous owner which is chunk_id %d of file %s (file ID %d)"%(batch_item, nonload_waveop_id_tmp, i, sb_addr, file_params.file_name, file_params.file_id, chunk_id, self.file_params_list[file_id].file_name, file_id))
                elif file_id != -1:
                    raise RuntimeError("File ID %d not found in list of file_params"%(file_id))
                self.morsels[sb_addr].file_id = file_params.file_id
                self.morsels[sb_addr].writer_id = -1
                self.morsels[sb_addr].reader_id = nonload_waveop_id_tmp
                self.morsels[sb_addr].chunk_id = i
                self.morsels[sb_addr].batch_item = batch_item
            if not file_params.mapped_params.chunk_is_mapped[i]:
                file_params.mapped_params.chunk_is_mapped[i] = True
            # generate DRAM save waveops (only need to depend on writers, when saving data to DRAM)                   
            list_of_accessors = list_of_writers_per_chunk + list_of_readers_per_chunk
            prev_waveops = []
            if list_of_accessors != []:
                # include all accessors for saving to DRAM, instead of just the latest accessor
                for accessor in list_of_accessors:
                    # allow for the fact that when generating matmul waveops, there could be read to the same space before waveop is added to nonload_waveop_list
                    if accessor >= 0 and accessor < len(nonload_waveop_list):
                        accessor_name = nonload_waveop_list[accessor]['waveop_name']
                        if accessor_name not in prev_waveops:
                            prev_waveops.append(accessor_name)
            new_dram_waveop = self.gen_dram_save_waveop(file_params, batch_item, i, prev_waveops)
            list_of_waveops.append(new_dram_waveop)
            #print("INFO: batch item %d: DRAM saver (SBAtomSave) waveop (non-load) ID %d is reading chunk_id %d (start %d, end %d) of file %s"%(batch_item, nonload_waveop_id_tmp, i, start_fmap_addr, end_fmap_addr, file_params.file_name))
            # trace waveop_id for newly created SBAtomSaves (to trace dependency so that new writer to same space need to wait for this save to complete)
            nonload_waveop_id_tmp += 1
        return list_of_waveops

    # Always read the maximum number of channels (min(C, 128))
    # TODO: add case for replication
    def read_file_data_region(self, nonload_waveop_id, nonload_waveop_list, file_params, batch_item, start_addr, length):
        assert(batch_item < file_params.file_dims.N)
        assert(length > 0)
        assert(length <= file_params.mapped_params.region_sz)
        assert(start_addr >= 0)
        start_chunk_id      = self.get_chunk_id_from_file_addr(file_params, batch_item, start_addr)
        end_chunk_id        = self.get_chunk_id_from_file_addr(file_params, batch_item, start_addr + length - self.item_sz)
        num_chunks          = end_chunk_id - start_chunk_id + 1
        #print("Reading batch item %d starting at %d for length %d (chunks %d to %d)"%(batch_item, start_addr, length, start_chunk_id, end_chunk_id))
        if num_chunks > file_params.mapped_params.num_region_chunks:
            raise RuntimeError("Number of chunks read %d for start %d length %d is larger than mapped number of chunks %d"%(num_chunks, start_addr, length, file_params.mapped_params.num_region_chunks))
        list_of_waveops = []
        list_of_writers = []
        list_of_readers = []
        for i in range(start_chunk_id, end_chunk_id + 1):
            list_of_writers_per_chunk = []
            list_of_readers_per_chunk = []
            start_fmap_addr = self.get_sb_addr_from_chunk_id(file_params, batch_item, i)
            end_fmap_addr = start_fmap_addr + self.get_chunk_len_from_chunk_id(file_params, batch_item, i)
            for j in range(start_fmap_addr, end_fmap_addr, file_params.item_sz):
                sb_addr = j
                # return list of writers/readers for dependency
                if self.morsels[sb_addr].writer_id not in list_of_writers_per_chunk:
                    list_of_writers_per_chunk.append(self.morsels[sb_addr].writer_id)
                if self.morsels[sb_addr].reader_id not in list_of_readers_per_chunk:
                    list_of_readers_per_chunk.append(self.morsels[sb_addr].reader_id)
                # Evict old owner                    
                file_id = self.morsels[sb_addr].file_id
                chunk_id = self.morsels[sb_addr].chunk_id
                owner_batch_item = self.morsels[sb_addr].batch_item
                if file_id in self.file_params_list:
                    if (file_id != file_params.file_id) or (chunk_id != i):
                        #print("INFO: batch item %d: Reader waveop (non-load) ID %d is reading chunk_id %d (addr %d) of file %s (file ID %d), clearing previous owner which is chunk_id %d of file %s (file ID %d)"%(batch_item, nonload_waveop_id, i, sb_addr, file_params.file_name, file_params.file_id, chunk_id, self.file_params_list[file_id].file_name, file_id))
                        self.file_params_list[file_id].mapped_params.chunk_is_mapped[chunk_id] = False
                elif file_id != -1:
                    raise RuntimeError("File ID %d not found in list of file_params"%(file_id))
                self.morsels[sb_addr].file_id = file_params.file_id
                self.morsels[sb_addr].writer_id = -1
                self.morsels[sb_addr].reader_id = nonload_waveop_id
                self.morsels[sb_addr].chunk_id = i
                self.morsels[sb_addr].batch_item = batch_item
            # if data is not in SB, issue DRAM loads
            #if not file_params.mapped_params.chunk_is_mapped[batch_item][i]:
            if not file_params.mapped_params.chunk_is_mapped[i]:
                file_params.mapped_params.chunk_is_mapped[i] = True
                # If modifying in place, don't create DRAM waveops for region
                if not file_params.mapped_params.modify_in_place:
                    list_of_accessors = list_of_writers_per_chunk + list_of_readers_per_chunk
                    prev_waveops = []
                    if list_of_accessors != []:
                        latest_accessor = max(list_of_accessors)
                        # allow for the fact that when generating matmul waveops, there could be read to the same space before waveop is added to nonload_waveop_list
                        if latest_accessor >= 0 and latest_accessor < len(nonload_waveop_list):
                            latest_accessor_name = nonload_waveop_list[latest_accessor]['waveop_name']
                            if latest_accessor_name not in prev_waveops:
                                prev_waveops.append(latest_accessor_name)
                    new_dram_waveop = self.gen_dram_read_waveop(file_params, batch_item, i, prev_waveops)
                    list_of_waveops.append(new_dram_waveop)
                    file_params.mapped_params.chunk2waveop_map[i] = new_dram_waveop
                    #file_params.mapped_params.chunk_is_mapped[batch_item][i] = True
                    #print("INFO: batch item %d: Reader ID %d is reading chunk_id %d (start %d, end %d) of file %s, creating DRAM load waveops"%(batch_item, nonload_waveop_id, i, start_fmap_addr, end_fmap_addr, file_params.file_name))
                #else:
                #    print("INFO: batch item %d: Reader ID %d is reading chunk_id %d (start %d, end %d) of file %s, which is being modified in place, so not creating DRAM load waveops"%(batch_item, nonload_waveop_id, i, start_fmap_addr, end_fmap_addr, file_params.file_name))
            list_of_writers += list_of_writers_per_chunk                
            list_of_readers += list_of_readers_per_chunk                
        return (list_of_writers, list_of_readers, list_of_waveops)

    def gen_dram_read_waveop(self, file_params, batch_item, chunk_id, previous_waveops):
        length          = self.get_chunk_len_from_chunk_id(file_params, batch_item, chunk_id)
        offset_in_file  = self.get_file_addr_from_chunk_id(file_params, batch_item, chunk_id)
        sb_addr         = self.get_sb_addr_from_chunk_id(file_params, batch_item, chunk_id)
        fmap_count      = self.get_fmap_count_from_chunk_id(file_params, batch_item, chunk_id)
        assert (length > 0)           
        # IFMAP replication parameters
        src_step_elem = 1
        ifmap_replication_num_rows = 0
        ifmap_replication_resolution = 0
        ifmap_replication_step_bytes = 0
        if file_params.replicate_multiple > 1:
            src_step_elem = file_params.stride_x
            fmap_count = fmap_count * file_params.replicate_multiple
            if file_params.file_dims.has_M:
                ifmap_replication_num_rows = file_params.file_dims.C
                ifmap_replication_resolution = file_params.file_dims.C
                ifmap_replication_step_bytes = file_params.file_dims.M * file_params.item_sz
            else:
                ifmap_replication_num_rows = file_params.file_dims.C * file_params.weights_S_dim
                ifmap_replication_resolution = file_params.file_dims.C * file_params.stride_x
                ifmap_replication_step_bytes = file_params.file_dims.W * file_params.stride_x * file_params.item_sz

        # collect stats
        #if (args.debug > 1):
        #    self.DRAM_elem_read += length * fmap_count / self.item_sz
        #    self.DRAM_atoms_read += 1
        #    self.circbuf_stats.sb_all_channels_memcpys_in += fmap_count
        #    if (length < self.atom_data_sz):
        #        self.DRAM_atoms_read_short += 1
        #print("gen_dram_read_waveop - DRAM_elem_read: ", self.DRAM_elem_read, "length: ", length, "fmap_count: ",fmap_count)
        #print("fmap_data_len",fmap_data_len, "atom_data_sz",self.atom_data_sz)
        #print("chunk_id", chunk_id, "offset", offset)
        #if (args.golden_inputs):            
        #    simout_file = self.dram_data_in_file.replace("-midout.", ".")
        #else:            
        #    simout_file = self.dram_data_in_file.replace("-midout.", "-simout.")
        simout_file = file_params.file_name.replace("-midout.", "-simout.")
        waveop_name = simout_file.replace(":", "__") + "_%d"%(chunk_id)
        return {
              'previous_waveops' : previous_waveops,
              'waveop_type'      : "SBAtomFile",
              'waveop_name'      : waveop_name,
              'layer_name'       : file_params.layer_name,
              'sb_address'       : sb_addr,
              'data_type'        : file_params.data_type,
              'contain_weights'  : file_params.file_dims.has_M,
              'ref_file'         : simout_file,
              'ref_file_format'  : file_params.file_dims.format_str,
              'ref_file_shape'   : file_params.file_dims.shape_tuple,
              'offset_in_file'   : offset_in_file,
              'length'           : length,
              'start_at_mid_part' : False,  # TODO: is this always false for loads?
              'ifmaps_replicate' : ifmap_replication_resolution > 0,  # TODO: is this still needed?
              'ifmaps_fold_idx'  : 0,    # TODO: is this still needed?
              'batch_fold_idx'   : 0, #wave_id.n_id,
              'ifmap_count'      : fmap_count,  # if this is larger than C, replicate fmap_count/C times
              'partition_step_bytes': file_params.fmap_data_len,
              'src_step_elem'     : src_step_elem,
              'ifmap_replication_resolution' : ifmap_replication_resolution, 
              'ifmap_replication_num_rows' : ifmap_replication_num_rows,
              'ifmap_replication_step_bytes' : ifmap_replication_step_bytes,
            }

    def gen_dram_save_waveop(self, file_params, batch_item, chunk_id, previous_waveops):
        length          = self.get_chunk_len_from_chunk_id(file_params, batch_item, chunk_id)
        offset_in_file  = self.get_file_addr_from_chunk_id(file_params, batch_item, chunk_id)
        sb_addr         = self.get_sb_addr_from_chunk_id(file_params, batch_item, chunk_id)
        fmap_count      = self.get_fmap_count_from_chunk_id(file_params, batch_item, chunk_id)
        # collect stats
        #if (args.debug > 1):
        #    self.DRAM_elem_written += length * ofmap_count / self.item_sz
        #    self.DRAM_atoms_written += 1
        #    self.circbuf_stats.sb_all_channels_memcpys_out += ofmap_count*((tile_id.m_id%2)+1)
        # if this is last chunk in OFMAP, mark it as last
        last_atom_of_file = chunk_id == (file_params.tot_num_chunks - 1)
        #last_atom_of_file = (tile_id.m_id+1 == tile_id.m) and (ceildiv(offset_in_fold+length, self.atom_data_sz) == ceildiv(self.ofmap_data_len, self.atom_data_sz))
        #print("m_id %d m %d offset_in_fold %d length %d ofmap_data_len %d last %d"%(tile_id.m_id, tile_id.m, offset_in_fold, length, self.ofmap_data_len, last_atom_of_file))
        # use "simout" tag for Back-end/Inkling result file
        simout_file = file_params.file_name.replace("-midout.", "-simout.")
        waveop_name = simout_file.replace(":", "__") + "_%d"%(chunk_id)
        return {
              'previous_waveops' : previous_waveops,
              'waveop_type'      : "SBAtomSave",
              'waveop_name'      : waveop_name,
              'layer_name'       : file_params.layer_name,
              'sb_address'       : sb_addr,
              'data_type'        : self.data_type,
              'ref_file'         : simout_file,
              'ref_file_format'  : file_params.file_dims.format_str,
              'ref_file_shape'   : file_params.file_dims.shape_tuple,
              'offset_in_file'   : offset_in_file,
              'length'           : length,
              'start_at_mid_part' : False, #(tile_id.m_id%2) == 1,
              'ofmaps_fold_idx'  : 0,   # TODO: is this still needed?
              'batch_fold_idx'   : 0,   # TODO: is this still needed?
              'ofmap_count'      : fmap_count,
              'partition_step_bytes': file_params.fmap_data_len,
              'last'             : last_atom_of_file,
              'final_layer_ofmap' : file_params.final_layer_ofmap,
            }

#######################################################################
# Unit tests
#######################################################################
class TestShapeDims(unittest.TestCase):
    def test_instantiation_and_retrieval(self):
        test_obj = ShapeDims("NHWC", [10,20,30,40]) 
        self.assertEqual(test_obj.N, 10)
        self.assertEqual(test_obj.H, 20)
        self.assertEqual(test_obj.W, 30)
        self.assertEqual(test_obj.C, 40)
        self.assertEqual(test_obj.N_axis, 0)
        self.assertEqual(test_obj.H_axis, 1)
        self.assertEqual(test_obj.W_axis, 2)
        self.assertEqual(test_obj.C_axis, 3)
        test_obj = ShapeDims("WHCN", [100,200,300,400]) 
        self.assertEqual(test_obj.dim["N"], 400)
        self.assertEqual(test_obj.dim["H"], 200)
        self.assertEqual(test_obj.dim["W"], 100)
        self.assertEqual(test_obj.dim["C"], 300)
        self.assertEqual(test_obj.dim["M"], 1)
        self.assertEqual(test_obj.dim["R"], 1)
        self.assertEqual(test_obj.dim["S"], 1)
        self.assertEqual(test_obj.axis["N"], 3)
        self.assertEqual(test_obj.axis["H"], 1)
        self.assertEqual(test_obj.axis["W"], 0)
        self.assertEqual(test_obj.axis["C"], 2)
        self.assertEqual(test_obj.axis["M"], -1)
        self.assertEqual(test_obj.axis["R"], -1)
        self.assertEqual(test_obj.axis["S"], -1)
        test_obj = ShapeDims("CRSM", [10,20,30,40]) 
        self.assertEqual(test_obj.C, 10)
        self.assertEqual(test_obj.R, 20)
        self.assertEqual(test_obj.S, 30)
        self.assertEqual(test_obj.M, 40)
        self.assertEqual(test_obj.C_axis, 0)
        self.assertEqual(test_obj.R_axis, 1)
        self.assertEqual(test_obj.S_axis, 2)
        self.assertEqual(test_obj.M_axis, 3)
        with self.assertRaises(RuntimeError):
            test_obj = ShapeDims("XHWC", [10,20,30,40]) 
        #with self.assertRaises(RuntimeError):
        #    test_obj = ShapeDims("CHWC", [10,20,30,40], 1) 
        with self.assertRaises(RuntimeError):
            test_obj = ShapeDims("CHWNX", [10,20,30,40]) 
        with self.assertRaises(RuntimeError):
            test_obj = ShapeDims("CHWC", [1,10,20,30,40]) 

class TestFileParams(unittest.TestCase):
    class pearray_params():
        MAX_WAVE_SIZE=256
        NUM_ROWS=128
        NUM_COLS=64

    class op_params_stride2():
        stride_x = 2
        stride_y = 2
        replicate_multiple = 1

    class op_params_stride1():
        stride_x = 1
        stride_y = 1
        replicate_multiple = 1

    def test_file_params_instantiation(self):
        shape_dims = ShapeDims("CRSM", [1,7,7,64]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 896)
        shape_dims = ShapeDims("CRSM", [256,1,1,128]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 256)
        self.assertEqual(test_obj.ravel_crsm(0,0,0,0), 0)
        self.assertEqual(test_obj.ravel_crsm(1,0,0,0), 128*test_obj.item_sz)
        shape_dims = ShapeDims("NHWC", [1,224,224,3]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride2)
        self.assertEqual(test_obj.chunk_sz, 1792)
        shape_dims = ShapeDims("NHWC", [1,112,112,64]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 1792)
        shape_dims = ShapeDims("NHWC", [1,55,55,128]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride2)
        self.assertEqual(test_obj.chunk_sz, 1760)
        shape_dims = ShapeDims("NHWC", [1,55,55,128]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 1760)
        shape_dims = ShapeDims("NHWC", [1,55,55,128]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 1210, self.pearray_params, self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 880)
        shape_dims = ShapeDims("NHWC", [1,28,28,256]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 1568)
        shape_dims = ShapeDims("NHWC", [1,14,14,256]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 392)
        shape_dims = ShapeDims("NHWC", [1,7,7,256]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 98)
        self.assertEqual(test_obj.ravel_nchw(0,0,0,0), 0)
        self.assertEqual(test_obj.ravel_nchw(0,0,0,1), 256*test_obj.item_sz)
        self.assertEqual(test_obj.ravel_nchw(0,0,1,0), 7*256*test_obj.item_sz)
        shape_dims = ShapeDims("NHWC", [4,1,1,2048]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        self.assertEqual(test_obj.tot_partition_usage_sz, 2048*4//128*test_obj.item_sz)
        self.assertEqual(test_obj.batch_item_partition_usage_sz, 2048//128*test_obj.item_sz)
        shape_dims = ShapeDims("NHWC", [4,55,55,512]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        self.assertEqual(test_obj.tot_partition_usage_sz, 55*55*512*4//128*test_obj.item_sz)
        self.assertEqual(test_obj.batch_item_partition_usage_sz, 55*55*512//128*test_obj.item_sz)
        shape_dims = ShapeDims("NHWC", [4,1,1,1000]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        self.assertEqual(test_obj.tot_partition_usage_sz, 64)
        self.assertEqual(test_obj.batch_item_partition_usage_sz, 16)
        shape_dims = ShapeDims("NHWC", [16,1,1,1000]) 
        test_obj = FileParams(0, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        self.assertEqual(test_obj.tot_partition_usage_sz, 4*64)
        self.assertEqual(test_obj.batch_item_partition_usage_sz, 16)


class TestFileMapper(unittest.TestCase):
    class pearray_params():
        MAX_WAVE_SIZE=256
        NUM_ROWS=128
        NUM_COLS=64

    class op_params_stride2():
        stride_x = 2
        stride_y = 2
        replicate_multiple = 1

    class op_params_stride1():
        stride_x = 1
        stride_y = 1
        replicate_multiple = 1

    def test_map_file(self):
        current_file_id = 0
        shape_dims = ShapeDims("CRSM", [256,7,7,64]) 
        file_params = FileParams(current_file_id, "testfile.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        self.assertEqual(file_params.chunk_sz, 896)
        test_obj = FileMapper(96*1024, "float16")
        self.assertEqual(test_obj.item_sz, 2)
        self.assertEqual(test_obj.data_type, "float16")
        self.assertEqual(test_obj.sb_partition_sz, 96*1024)
        with self.assertRaises(RuntimeError):   
            test_obj.map_file(None, 0)          # no file info
        with self.assertRaises(RuntimeError):   
            test_obj.map_file(file_params, 0)   # file not loaded
        file_params.load_file()            
        #with self.assertRaises(RuntimeError):
        #    test_obj.map_file(file_params, 0, True, region_sz=10)   # wrap-around but region size cannot accomodate chunk size
        with self.assertRaises(RuntimeError):
            test_obj.map_file(file_params, 0, False, region_sz=10)  # file size larger than region size, but wrap-around not enabled
        with self.assertRaises(RuntimeError):
            test_obj.map_file(file_params, 96*1024, False, region_sz=10)   # error start
        with self.assertRaises(RuntimeError):
            test_obj.map_file(file_params, 95*1024, False, region_sz=1024)   # error end
        end = test_obj.map_file(file_params, 40*1024, False, region_sz=2*7*7*64*file_params.item_sz)            
        self.assertEqual(end, 40*1024 + 2*7*7*64*file_params.item_sz)
        #with self.assertRaises(RuntimeError):
        #    test_obj.map_file(file_params, 80*1024, False, region_sz=7*7*64*file_params.item_sz)  # duplicate file
        self.assertEqual(file_params.chunk_sz, 896)
        self.assertEqual(file_params.tot_partition_usage_sz, 2*7*7*64*file_params.item_sz)
        self.assertEqual(file_params.batch_item_partition_usage_sz, 2*7*7*64*file_params.item_sz)
        self.assertEqual(file_params.batch_item_num_chunks, 2*7*7*64*file_params.item_sz//896)
        self.assertEqual(file_params.tot_num_chunks, 2*7*7*64*file_params.item_sz//896)
        self.assertEqual(file_params.file_dims.H, 1)
        self.assertEqual(file_params.file_dims.W, 1)
        self.assertEqual(file_params.fmap_data_len, 7*7*64*file_params.item_sz)
        self.assertEqual(file_params.file_addr_skip_per_fmap_fold, 128*7*7*64*file_params.item_sz * 1)
        self.assertEqual(test_obj.get_chunk_id_from_file_addr(file_params, 0, 0), 0)
        self.assertEqual(test_obj.get_chunk_id_from_file_addr(file_params, 0, 896), 1)
        self.assertEqual(test_obj.get_chunk_id_from_file_addr(file_params, 0, 896*3), 3)
        #with self.assertRaises(RuntimeError):
        #    test_obj.get_chunk_id_from_file_addr(file_params, file_params.dram_data.size)
        self.assertEqual(test_obj.get_sb_addr_from_file_addr(file_params, 0, 0), 40*1024)
        self.assertEqual(test_obj.get_chunk_offset_from_file_addr(file_params, 0, 100), 100)
        list_of_accessors = [{'waveop_name' : "waveop_%d"%i} for i in range(100)]
        (writers, readers, waveops) = test_obj.read_file_data_region(10, list_of_accessors, file_params, 0, 0, 100)
        self.assertEqual(len(waveops), 1)
        self.assertEqual(waveops[0]['previous_waveops'], [])
        self.assertEqual(writers, [-1])
        self.assertEqual(readers, [-1])
        (writers, readers, waveops) = test_obj.write_file_data_region(20, list_of_accessors, file_params, 0, 0, 10, False)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(writers, [-1])
        self.assertEqual(readers, [10])
        (writers, readers, waveops) = test_obj.write_file_data_region(30, list_of_accessors, file_params, 0, 0, 100, False)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(writers.sort(), [10, 20].sort())
        self.assertEqual(readers, [-1])
        (writers, readers, waveops) = test_obj.write_file_data_region(40, list_of_accessors, file_params, 0, 100, 200, False)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(writers, [30])
        self.assertEqual(readers, [-1])
        (writers, readers, waveops) = test_obj.read_file_data_region(40, list_of_accessors, file_params, 0, 100, 200)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(writers, [40])
        self.assertEqual(readers, [-1])
        self.assertEqual(test_obj.get_chunk_id_from_file_addr(file_params, 0, 100), 0)
        #self.assertEqual(file_params.mapped_params.chunk_is_mapped[0][test_obj.get_chunk_id_from_file_addr(file_params, 0, 100)], True)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[test_obj.get_chunk_id_from_file_addr(file_params, 0, 100)], True)
        (writers, readers, waveops) = test_obj.read_file_data_region(40, list_of_accessors, file_params, 0, 100, 200)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(writers, [-1])
        self.assertEqual(readers, [40])
        (writers, readers, waveops) = test_obj.read_file_data_region(50, list_of_accessors, file_params, 0, 50, 150)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(writers, [-1])
        self.assertEqual(readers, [40])
        # test reading from file
        (writers, readers, waveops) = test_obj.read_file_data_region(10, list_of_accessors, file_params, 0, 7*7*64*file_params.item_sz, file_params.chunk_sz)
        self.assertEqual(len(waveops), 1)
        self.assertEqual(waveops[0]['previous_waveops'], [])
        self.assertEqual(waveops[0]["sb_address"], 40*1024 + 7*7*64*file_params.item_sz)
        self.assertEqual(waveops[0]["offset_in_file"], 128*7*7*64*file_params.item_sz)
        # test writing to file
        (writers, readers, waveops) = test_obj.write_file_data_region(40, list_of_accessors, file_params, 0, 100, 200, False)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(writers, [-1])
        self.assertEqual(readers, [50])
        (writers, readers, waveops) = test_obj.write_file_data_region(40, list_of_accessors, file_params, 0, 100, 200, False)
        self.assertEqual(len(waveops), 0)
        #self.assertEqual(waveops[0]['previous_waveops'], ["waveop_40"])
        self.assertEqual(writers, [40])
        self.assertEqual(readers, [-1])
        waveops = test_obj.flush_file(list_of_accessors, file_params, 0)
        self.assertEqual(len(waveops), file_params.tot_num_chunks)
        self.assertEqual(waveops[0]['previous_waveops'], ["waveop_40"])
        self.assertEqual(waveops[1]['previous_waveops'], [])

    def test_map_file2(self):
        shape_dims = ShapeDims("NCHW", [16,3,224,224]) 
        file_params = FileParams(0, "testfile2.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride2)
        file_params.load_file()
        test_obj = FileMapper(96*1024, "float16")
        test_obj.map_file(file_params, 50240, True, region_sz=55*55*file_params.item_sz)
        self.assertEqual(file_params.mapped_params.start_addr, 50240)
        self.assertEqual(file_params.mapped_params.region_sz, 3*file_params.chunk_sz)
        self.assertEqual(file_params.mapped_params.num_region_chunks, 3)
        self.assertEqual(file_params.mapped_params.num_file_chunks_per_batch_item, 56)
        self.assertEqual(file_params.mapped_params.end_addr, 50240 + 3*file_params.chunk_sz - file_params.item_sz)
        list_of_accessors = [{'waveop_name' : "waveop_%d"%i} for i in range(100)]
        (writers, readers, waveops) = test_obj.write_file_data_region(10, list_of_accessors, file_params, 15, 15*3*224*224*file_params.item_sz + 0, 100, False)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[15*file_params.batch_item_num_chunks + 0], True)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[15*file_params.batch_item_num_chunks + 1], False)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[0*file_params.batch_item_num_chunks + 0], False)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[0*file_params.batch_item_num_chunks + 1], False)
        self.assertEqual(writers, [-1])
        self.assertEqual(readers, [-1])
        (writers, readers, waveops) = test_obj.write_file_data_region(20, list_of_accessors, file_params, 0, 0, 10, False)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[15*file_params.batch_item_num_chunks + 0], False)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[15*file_params.batch_item_num_chunks + 1], False)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[0*file_params.batch_item_num_chunks + 0], True)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[0*file_params.batch_item_num_chunks + 1], False)
        self.assertEqual(writers, [10])
        self.assertEqual(readers, [-1])

    def test_map_file_55x55(self):
        shape_dims = ShapeDims("NCHW", [1,256,55,55]) 
        file_params = FileParams(0, "testfile_55x55.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride2)
        file_params.load_file()
        test_obj = FileMapper(96*1024, "float16")
        #test_obj.map_file(file_params, 0, True, region_sz=55*55*file_params.item_sz)
        test_obj.map_file(file_params, 0, True, region_sz=12544)
        self.assertEqual(file_params.chunk_sz, 1760)
        self.assertEqual(file_params.mapped_params.start_addr, 0)
        #self.assertEqual(file_params.mapped_params.region_sz, 6*file_params.chunk_sz)
        self.assertEqual(file_params.mapped_params.num_region_chunks, 8)
        self.assertEqual(file_params.mapped_params.num_file_chunks_per_batch_item, 8)
        self.assertEqual(file_params.mapped_params.end_addr, (55*55*2 - 1)*file_params.item_sz) 
        list_of_accessors = [{'waveop_name' : "waveop_%d"%i} for i in range(100)]
        (writers, readers, waveops) = test_obj.read_file_data_region(10, list_of_accessors, file_params, 0, 0, 100)
        self.assertEqual(waveops[0]["sb_address"], 0)
        self.assertEqual(waveops[0]["offset_in_file"], 0)
        (writers, readers, waveops) = test_obj.read_file_data_region(10, list_of_accessors, file_params, 0, 128*55*55*file_params.item_sz, 100)
        self.assertEqual(test_obj.get_chunk_id_from_file_addr(file_params, 0, 0), 0)
        self.assertEqual(test_obj.get_file_addr_from_chunk_id(file_params, 0, 0), 0)
        self.assertEqual(test_obj.get_chunk_id_from_file_addr(file_params, 0, 128*55*55*file_params.item_sz), 4)
        self.assertEqual(test_obj.get_sb_addr_from_file_addr(file_params, 0, 128*55*55*file_params.item_sz), 55*55*file_params.item_sz)
        self.assertEqual(test_obj.get_file_addr_from_chunk_id(file_params, 0, 4), 128*55*55*file_params.item_sz)
        self.assertEqual(waveops[0]["sb_address"], 55*55*file_params.item_sz)
        self.assertEqual(waveops[0]["offset_in_file"], 128*55*55*file_params.item_sz)

    def test_zero_file(self):
        shape_dims = ShapeDims("NCHW", [16,3,224,224]) 
        file_params = FileParams(0, "testfile2.npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride2)
        x = file_params.load_file()
        y = np.random.rand(x.size)
        z = y.astype(x.dtype)
        w = z.reshape(x.shape)
        file_params.dram_data = w
        file_params.save_file()
        file_params.load_file()
        self.assertEqual(np.allclose(w, file_params.dram_data), True)
        w = np.zeros([16,3,224,224], dtype=file_params.data_type)
        file_params.zero_file()
        self.assertEqual(np.allclose(w, file_params.dram_data), True)

    def map_chunk_id_single(self, shape_tuple, region_sz):
        shape_dims = ShapeDims("NCHW", shape_tuple)
        file_params = FileParams(0, "testfile_"+str(shape_tuple)+".npy", shape_dims, "float16", 2048, self.pearray_params, self.op_params_stride1)
        file_params.load_file()
        test_obj = FileMapper(96*1024, "float16")
        #test_obj.map_file(file_params, 0, True, region_sz=55*55*file_params.item_sz)
        test_obj.map_file(file_params, 0, True, region_sz=region_sz)
        list_of_accessors = [{'waveop_name' : "waveop_%d"%i} for i in range(100)]
        tile_size = file_params.fmap_full_tilex_sz * file_params.fmap_full_tiley_sz * 2
        num_tiles = ceildiv(file_params.file_dims.H * file_params.file_dims.W * 2, tile_size)
        last_tile_size = (file_params.file_dims.H * file_params.file_dims.W * 2) % tile_size
        if last_tile_size == 0: last_tile_size = tile_size
        current_offset = 0
        # check read_file_data_region
        for i in range(file_params.file_dims.N):
            last_batch_offset = current_offset
            for k in range(file_params.fmap_channels_folds):
                last_channel_offset = current_offset
                for j in range(num_tiles):
                    #print("fmap_full_tilex_sz %d fmap_full_tiley_sz %d num_tiles %d tile_size %d last_tile_size %d"%(file_params.fmap_full_tilex_sz, file_params.fmap_full_tiley_sz, num_tiles, tile_size, last_tile_size))
                    current_tile_size = last_tile_size if (j == num_tiles-1) else tile_size
                    (writers, readers, waveops) = test_obj.read_file_data_region(10, list_of_accessors, file_params, i, current_offset, current_tile_size)
                    current_offset += current_tile_size
                current_offset = last_channel_offset + file_params.file_addr_skip_per_fmap_fold
            current_offset = last_batch_offset + file_params.file_addr_skip_per_batch_item
        current_offset = 0
        # check write_file_data_region
        for i in range(file_params.file_dims.N):
            last_batch_offset = current_offset
            for k in range(file_params.fmap_channels_folds):
                last_channel_offset = current_offset
                for j in range(num_tiles):
                    current_tile_size = last_tile_size if (j == num_tiles-1) else tile_size
                    (writers, readers, waveops) = test_obj.write_file_data_region(10, list_of_accessors, file_params, i, current_offset, current_tile_size, False)
                    current_offset += current_tile_size
                current_offset = last_channel_offset + file_params.file_addr_skip_per_fmap_fold
            current_offset = last_batch_offset + file_params.file_addr_skip_per_batch_item
        # Check get_chunk_id_from_file_addr and get_file_addr_from_chunk_id
        current_offset = 0
        for i in range(file_params.file_dims.N):
            last_batch_offset = current_offset
            for k in range(file_params.fmap_channels_folds):
                last_channel_offset = current_offset
                for j in range(num_tiles):
                    current_tile_size = last_tile_size if (j == num_tiles-1) else tile_size
                    chunk_id = test_obj.get_chunk_id_from_file_addr(file_params, i, current_offset)
                    chunk_offset = test_obj.get_chunk_offset_from_file_addr(file_params, i, current_offset)
                    file_addr = test_obj.get_file_addr_from_chunk_id(file_params, i, chunk_id)
                    #print("current_offset %d -> chunk_id %d chunk_offset %d file_addr %d"%(current_offset, chunk_id, chunk_offset, file_addr))
                    self.assertEqual(file_addr + chunk_offset, current_offset) 
                    current_offset += current_tile_size
                current_offset = last_channel_offset + file_params.file_addr_skip_per_fmap_fold
            current_offset = last_batch_offset + file_params.file_addr_skip_per_batch_item

    # mapping tests
    def test_map_chunk_id_var_shape (self):
        self.map_chunk_id_single([4,256,55,55], 55*55*2)
        #self.map_chunk_id_single([4,2048,1,1], 4*2048*1*1//128)
        self.map_chunk_id_single([4,1000,1,1], 4*1000*1*1//128)

if __name__ == '__main__':
    unittest.main()
