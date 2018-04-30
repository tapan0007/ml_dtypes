import unittest
import numpy as np
from enum import Enum

def ceildiv(x, y):
    return -(-x//y)

# Class to manage head and tail pointers for circular buffer with two modes:
#   - endzone_sz = 0: Normal mode: advance pointer until capacity, where it wraps to 0
#   - endzone_sz > 0: End-sink mode: advance pointer until endzone, where it wraps to (capacity - endzone_sz)
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

    def test_endsink_exception(self):
        for capacity in range(10,100):
            for endzone_sz in range(1,10):
                #print("capacity %d endzone_sz %d"%(capacity, endzone_sz))
                test_obj = CircbufPtrs(capacity, endzone_sz)
                for i in range(150):
                    if ((i%capacity) == capacity-1):
                        with self.assertRaises(RuntimeError):
                            self.assertLess (test_obj.advance(CircbufPtrs.TAIL), capacity)
                    else:                            
                        self.assertLess (test_obj.advance(CircbufPtrs.TAIL), capacity)


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

    def check_format_str(self, format_str):
        if (format_str != self.format_str):
            raise RuntimeError("ERROR ShapeDims: format_str %s doesn't match initialized format %s"%(format_str, self.format_str))

    def check_shape(self, shape_tuple):
        if (shape_tuple != self.shape_tuple):
            raise RuntimeError("ERROR ShapeDims: shape_tuple %s doesn't match %s"%(str(shape_tuple), str(self.shape_tuple)))

    # from a window (subshape), extract start/end offsets
    def get_startend_for_subshape(self, subshape_tuple):        
        pass

    # get the size of per-partition chunk size that is less than size limit
    def compute_fmap_params(self, item_sz, chunk_sz_limit):
        self.item_sz = item_sz
        self.chunk_sz_limit = chunk_sz_limit
        pass

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

class FileParams():
    def __init__(self, file_name, file_dims, item_sz, chunk_sz_limit, pearray_params, op_params, abstract_mem):
        self.file_name = file_name
        self.file_dims = file_dims
        self.item_sz = item_sz
        self.chunk_sz_limit = chunk_sz_limit
        self.compute_params(pearray_params, op_params, abstract_mem)

    def load_file(self):
        if (self.file_name != None):
            self.dram_data = np.load(self.file_name)
            assert(self.dram_data.flags.c_contiguous == True)
            assert(self.dram_data.itemsize == self.item_sz)
            print("dram_data.size %d file_dims.tot_elems %d"%(self.dram_data.size, self.file_dims.tot_elems))
            assert(self.dram_data.size == self.file_dims.tot_elems)

    def reshape_data(self, data):
        pass

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

    def compute_params(self, pearray_params, op_params, abstract_mem):
        # Single FMAP elem count (unified formula for weights and FMAP)
        self.fmap_elem_count = self.file_dims.R * self.file_dims.S * self.file_dims.M * self.file_dims.H * self.file_dims.W
        self.fmap_data_len = self.fmap_elem_count * self.item_sz
        #if (C < PEArray.NUM_ROWS and (R > 1 or S > 1)):                                                                <
        #   self.replicate_multiple = min(PEArray.NUM_ROWS//C, self.total_filter_size)                                  <
        # per kaena-85, use noodle shapes for tiles
        # need to guard against small EF and build noodle tile to enable higher state buffer efficiency
        self.fmap_full_tilex_sz = min(self.file_dims.W, pearray_params.MAX_WAVE_SIZE)
        self.fmap_full_tiley_sz = min(self.file_dims.H, pearray_params.MAX_WAVE_SIZE // self.fmap_full_tilex_sz)
        # Chunk (aka atom) size computation for weights
        if self.file_dims.M > 1:
            m_data_len = self.file_dims.M * self.item_sz
            sm_data_len = self.file_dims.S * m_data_len
            folding_multiple = (self.file_dims.C // pearray_params.NUM_ROWS) * (self.file_dims.M // pearray_params.NUM_COLS)
            atom_sz_for_computation = self.chunk_sz_limit
            # TODO: simplify to just limiting to 64 output channels
            if (folding_multiple > 16):
                atom_sz_for_computation = self.chunk_sz_limit//4
            if (self.fmap_data_len <= atom_sz_for_computation):
                self.chunk_sz = self.fmap_data_len
            # Else find the largest   
            elif (sm_data_len <= atom_sz_for_computation):
                multiple = atom_sz_for_computation // sm_data_len
                self.chunk_sz = sm_data_len * min(self.file_dims.R, multiple)
            elif (m_data_len <= atom_sz_for_computation):
                multiple = atom_sz_for_computation // m_data_len
                self.chunk_sz = m_data_len * min(self.file_dims.S, multiple)
            else:
                self.chunk_sz = atom_sz_for_computation
        else:                
            ifmap_width_data_len = self.file_dims.W * self.item_sz
            # make atom size multiple of IFMAP if IFMAP is smaller than default atom size (CNHW)
            # For NCHW, just use ifmap size as atom size (see rule above: "different FMAPs folds will be in different atoms")
            if (self.fmap_data_len <= self.chunk_sz_limit):
                self.chunk_sz = self.fmap_data_len
            # Cannot handle crossing atom gaps for case where number of IFMAPs is larger than PEArray rows, and filter size > 1
            elif (self.file_dims.C > 128 and self.file_dims.R > 1):
                print("ERROR %s: cannot yet handle case where number of IFMAPs > 128, and filter size is > 1"%(self.circbuf_type))
                exit(-1)
            # make atom size multiple of width data length if it is smaller than default atom size
            # For FP32, use initial atom of 2KB to guarantee gapless spaces for 28x28 (without using skip-atoms), when folding is involved
            elif (ifmap_width_data_len <= self.chunk_sz_limit):
                input_fmap_full_tiley_sz = self.fmap_full_tiley_sz * op_params.stride_y
                if (abstract_mem):
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
                    if (self.fmap_data_len % self.chunk_sz) != 0:
                        print("WARNING: FMAP size %d is not a multiple of chunk size %d for shape %s, so c>=1 addresses don't align to chunks!"%(self.fmap_data_len, self.chunk_sz, str(self.file_dims.shape_tuple)))
            else:
                self.chunk_sz = self.chunk_sz_limit
            # need spare atoms for the case that OFMAP tile needs overlaps in IFMAP tile                            
            if (op_params.stride_y > 1 or self.file_dims.R > 1):
                self.need_spare_atoms = max(1, ceildiv(self.fmap_full_tiley_sz * self.fmap_full_tilex_sz * self.item_sz, self.chunk_sz))
                print("Reserve %d as spare atoms"%self.need_spare_atoms)
                #if self.circbuf_type == "scratch" and self.num_kickout_atoms > 0:
                #    self.num_kickout_atoms = self.need_spare_atoms + self.num_kickout_atoms

class TestFileParams(unittest.TestCase):
    class pearray_params():
        MAX_WAVE_SIZE=256
        NUM_ROWS=128
        NUM_COLS=64

    class op_params_stride2():
        stride_x = 2
        stride_y = 2

    class op_params_stride1():
        stride_x = 1
        stride_y = 1

    def test_file_params_instantiation(self):
        shape_dims = ShapeDims("CRSM", [1,7,7,64]) 
        test_obj = FileParams("testfile.npy", shape_dims, 2, 2048, self.pearray_params, self.op_params_stride1, False)
        self.assertEqual(test_obj.chunk_sz, 1792)
        shape_dims = ShapeDims("CRSM", [256,1,1,128]) 
        test_obj = FileParams("testfile.npy", shape_dims, 2, 2048, self.pearray_params, self.op_params_stride1, False)
        self.assertEqual(test_obj.chunk_sz, 256)
        self.assertEqual(test_obj.ravel_crsm(0,0,0,0), 0)
        self.assertEqual(test_obj.ravel_crsm(1,0,0,0), 128*test_obj.item_sz)
        shape_dims = ShapeDims("NHWC", [1,224,224,3]) 
        test_obj = FileParams("testfile.npy", shape_dims, 2, 2048, self.pearray_params, self.op_params_stride2, True)
        self.assertEqual(test_obj.chunk_sz, 896)
        shape_dims = ShapeDims("NHWC", [1,224,224,3]) 
        test_obj = FileParams("testfile.npy", shape_dims, 2, 2048, self.pearray_params, self.op_params_stride2, False)
        self.assertEqual(test_obj.chunk_sz, 1792)
        shape_dims = ShapeDims("NHWC", [1,112,112,64]) 
        test_obj = FileParams("testfile.npy", shape_dims, 2, 2048, self.pearray_params, self.op_params_stride1, False)
        self.assertEqual(test_obj.chunk_sz, 1792)
        shape_dims = ShapeDims("NHWC", [1,55,55,128]) 
        test_obj = FileParams("testfile.npy", shape_dims, 2, 2048, self.pearray_params, self.op_params_stride2, False)
        self.assertEqual(test_obj.chunk_sz, 1760)
        shape_dims = ShapeDims("NHWC", [1,55,55,128]) 
        test_obj = FileParams("testfile.npy", shape_dims, 2, 2048, self.pearray_params, self.op_params_stride1, False)
        self.assertEqual(test_obj.chunk_sz, 1760)
        shape_dims = ShapeDims("NHWC", [1,28,28,256]) 
        test_obj = FileParams("testfile.npy", shape_dims, 2, 2048, self.pearray_params, self.op_params_stride1, False)
        self.assertEqual(test_obj.chunk_sz, 1568)
        shape_dims = ShapeDims("NHWC", [1,14,14,256]) 
        test_obj = FileParams("testfile.npy", shape_dims, 2, 2048, self.pearray_params, self.op_params_stride1, False)
        self.assertEqual(test_obj.chunk_sz, 392)
        shape_dims = ShapeDims("NHWC", [1,7,7,256]) 
        test_obj = FileParams("testfile.npy", shape_dims, 2, 2048, self.pearray_params, self.op_params_stride1, False)
        self.assertEqual(test_obj.chunk_sz, 98)
        self.assertEqual(test_obj.ravel_nchw(0,0,0,0), 0)
        self.assertEqual(test_obj.ravel_nchw(0,0,0,1), 256*test_obj.item_sz)
        self.assertEqual(test_obj.ravel_nchw(0,0,1,0), 7*256*test_obj.item_sz)


if __name__ == '__main__':
    unittest.main()
