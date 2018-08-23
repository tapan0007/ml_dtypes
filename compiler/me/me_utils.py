"""
Copyright 2018, Amazon.com, Inc. or its affiliates. All Rights Reserved
"""

""" File/shape utility functions and classes for Middle Scheduler:

    - pad_and_split_file: function to pad and split image for replication
    - ShapeDims: a class to manage N-dimensional shapes
    - FileParams: class to manage file-related parameters
    - FileMapper: class to manage mapping file to SB regions
"""

import unittest
import numpy as np
import os.path
import copy
from enum import Enum
from me_models import PEArray

#np.set_printoptions(precision=3)
#np.set_printoptions(threshold=np.nan)

def ceildiv(x, y):
    return -(-x//y)

def data_type_to_item_sz(data_type):
    return np.dtype(data_type).itemsize

"""Align address to multiple of NB
"""
def align_addr_NB(addr, N):
    return ceildiv(addr, N) * N

def assert_align_addr_4B(addr):
    assert(addr == align_addr_NB(addr, 4))
"""Align address to multiple of 8B
"""
def align_addr_8B(addr):
    return align_addr_NB(addr, 8)

def assert_align_addr_8B(addr):
    assert(addr == align_addr_NB(addr, 8))

def assert_align_addr_sb_read(addr):
    assert(addr == align_addr_sb_read(addr))

def assert_align_addr_sb_write(addr):
    assert(addr == align_addr_sb_write(addr))


"""Align address to multiple of 16B
"""
def align_addr_16B(addr):
    return align_addr_NB(addr, 16)

"""Align address to multiple of 64B
"""
def align_addr_64B(addr):
    return align_addr_NB(addr, 64)

def align_addr_sb_read(addr):
    return align_addr_NB(addr, 2)

def align_addr_sb_write(addr):
    return align_addr_NB(addr, 4)

"""For IFMAP replication (https://sim.amazon.com/issues/kaena-141), need to:
 - pad image
 - for num_to_split=2: split W columns into HWe and HWo where HWe include even columns and HWo includes odd columns
 - for num_to_split>2: split W columns into multiple HWi, where HWi includes i columns, i = column idx % num_to_split
""" 
def pad_and_split_file(file_to_split, file_format, num_to_split, pad_west, pad_east, pad_north, pad_south):
    assert(file_format == "NCHW")
    dram_data = np.load(file_to_split)
    N, C, H, W = dram_data.shape
    # compute pad dimensions: round width to multiple of num_to_split
    new_width = W + pad_west + pad_east
    new_width = ceildiv(new_width, num_to_split) * num_to_split
    new_pad_east = new_width - (W + pad_west)
    new_height = H + pad_north + pad_south
    # pad the image
    new_shape = [N, C, new_height*num_to_split, new_width//num_to_split]
    new_dram_data = np.zeros(tuple(new_shape), dtype=dram_data.dtype)
    for n in range(N):
        for c in range(C):
            #print("original channel %d:"%c)
            #print(dram_data[n,c,:])
            new_hw_padded = np.pad(dram_data[n,c,:], ((pad_north, pad_south), (pad_west, new_pad_east)), 'constant')
            new_hw_split = []
            for i in range(num_to_split):
                # split even/odd elements in each row
                new_hw_split_w = new_hw_padded[:, i::num_to_split]
                # split even/odd rows
                new_hw_split_h = []
                for j in range(num_to_split):
                    new_hw_split_h.append(new_hw_split_w[j::num_to_split, :])
                new_hw_split.append(np.concatenate(new_hw_split_h, 0))
            new_dram_data[n, c, :] = np.concatenate(new_hw_split, 0)
            #print("all frames channel %d:"%c)
            #print(new_dram_data[n, c, :])
    new_file = file_to_split.replace(".npy", "_padsplit.npy")
    try:
        np.save(new_file, new_dram_data)
    except:
        raise RuntimeError("Cannot save numpy file %s"%(new_file))
    return (new_file, new_shape)

""" Unstack parameters: keep track of ifmap and ofmap shape dimensions
"""
class UnstackParams:
    def __init__(self, knode, unstack_idx):
        self.item_sz = knode.item_sz
        assert(knode.data['layer_type'] == "Unstack")
        input_shape = knode.prev[0].data['ofmap_shape']
        input_format = knode.prev[0].data['ofmap_format']
        output_shape = knode.data['ofmap_shape']
        output_format = knode.data['ofmap_format']
        self.ifmap_shape_dims = ShapeDims(input_format, input_shape)
        self.ofmap_shape_dims = ShapeDims(output_format, output_shape)
        self.unstack_axis = knode.data['unstack_axis']
        self.unstack_sz = self.ofmap_shape_dims[output_format[self.unstack_axis]]
        self.unstack_offset = unstack_idx * self.unstack_sz
        assert(self.unstack_offset >= 0 and self.unstack_offset < self.ifmap_shape_dims[input_format[self.unstack_axis]])

    """ Extract unstacked data for batch item from DRAM data
        Args: 
        - batch_item: the current batch item, value between 0 and N-1
        - dram_data: the stacked data
        Returns:
        - unstacked data for the unstack offset specified within this object
    """
    def extract_data(self, batch_item, dram_data):
        assert(dram_data is not None)
        assert(len(self.ifmap_shape_dims.format_str) == 4)
        assert(self.ifmap_shape_dims.shape_tuple == dram_data.shape)
        slicer = slice(self.unstack_offset, self.unstack_offset + self.unstack_sz)
        switcher = {    
                0: dram_data[slicer],
                1: dram_data[:, slicer],
                2: dram_data[:, :, slicer],
                3: dram_data[:, :, :, slicer],
                }
        extracted = switcher[self.unstack_axis]
        extracted_reshaped = extracted.reshape(self.ofmap_shape_dims.shape_tuple)
        return extracted_reshaped

    def get_address_in_original_file(self, batch_item, address):
        coord = [0, 0, 0, 0]
        coord['N'] = batch_item
        coord[unstack_idx] = self.unstack_offset
        offset = int(np.ravel_multi_index(coord, dims=self.ifmap_shape_dims.shape_tuple) * self.item_sz)
        return address + offset

""" Class to manage coordinates (points)
"""
class Coord():
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, coord):
        x = self.x + coord.x
        y = self.y + coord.y
        return self.__class__(x, y)

    def __sub__(self, coord):
        x = self.x - coord.x
        y = self.y - coord.y
        return self.__class__(x, y)

    def __neg__(self):
        x = - self.x
        y = - self.y
        return self.__class__(x, y)

    def __mul__(self, coord):
        x = self.x * coord.x
        y = self.y * coord.y
        return self.__class__(x, y)

    def __floordiv__(self, coord):
        x = self.x // coord.x
        y = self.y // coord.y
        return self.__class__(x, y)

    def __lt__(self, coord):
        return (self.x < coord.x and self.y <= coord.y) or (self.x <= coord.x and self.y < coord.y)
        #return (self.x < coord.x) and (self.y < coord.y)

    def __gt__(self, coord):
        return (self.x > coord.x and self.y >= coord.y) or (self.x >= coord.x and self.y > coord.y)
        #return (self.x > coord.x) and (self.y > coord.y)

    def __le__(self, coord):
        return (self.x <= coord.x) and (self.y <= coord.y)

    def __ge__(self, coord):
        return (self.x >= coord.x) and (self.y >= coord.y)

    def __eq__(self, coord):
        return (self.x == coord.x) and (self.y == coord.y)

    def __str__(self):
        return "(x=%d,y=%d)"%(self.x, self.y)

    """ Make a rectangle from coordinate, taking it as the upper coordinates, and 0,0 as the lower coordinates
    """
    def make_rect(self):
        return Rect(Coord(0,0), self)

    """ Return the size equivalent of the rectangle with 0,0 as lower coordinates and own values as upper coordinates
    """
    def make_dim2d(self):
        return Dim2D(self.x+1, self.y+1)

    """ Snap the coordinates up to nearest grid
        Args:
            - stride: the grid spacing
        Return:
            - none (coordinates are updated)
    """
    def snap_up_nearest_grid(self, origin, stride):
        diff = self - origin
        diff = ceildiv(diff, stride) * stride
        new = origin + diff
        self.x = new.x
        self.y = new.y
        
    """ Snap the coordinates down to nearest grid
        Args:
            - stride: the grid spacing
        Return:
            - none (coordinates are updated)
    """
    def snap_down_nearest_grid(self, origin, stride):
        diff = self - origin
        diff = (diff // stride) * stride
        new = origin + diff
        self.x = new.x
        self.y = new.y

""" Class to manage 2D sizes: related to Coord but is off by 1,1 generally
"""
class Dim2D(Coord):
    """ Make a rectangle from size, taking it as the upper coordinates, and 0,0 as the lower coordinates
    """
    def make_rect(self):
        #assert(self.x > 0 and self.y > 0)
        return Rect(Coord(0,0), Coord(self.x-1, self.y-1))

    def get_tot_size(self):
        return self.x * self.y

    def make_dim2d(self):
        raise RuntimeError("Can't make Dim2D within Dim2D")

""" Class to manage  rectangle

Args:
    - coord_lower: lower coordinates (lower in address)
    - coord_upper: upper coordinates (higher in address)
    - coord_stride: stride x and y
    - parent: parent (enclosing) rectangle, if exists

"""
class Rect():
    lower = None
    upper = None
    dim2d = None
    is_empty = False

    def __init__(self, lower, upper):
        #assert(lower <= upper)
        diff = upper - lower
        self.dim2d = diff.make_dim2d()
        self.lower = lower
        self.upper = upper
        self.is_empty = not (lower <= upper)

    def __eq__(self, another_rect):
        return (self.lower == another_rect.lower) \
            and (self.upper == another_rect.upper)

    def __str__(self):
        return "%s, %s"%(str(self.lower), str(self.upper))

    def __add__(self, offset_coord):
        new_lower = self.lower + offset_coord
        new_upper = self.upper + offset_coord
        return Rect(new_lower, new_upper)

    def __sub__(self, offset_coord):
        new_lower = self.lower - offset_coord
        new_upper = self.upper - offset_coord
        return Rect(new_lower, new_upper)

    def __mul__(self, mul_dim2d):
        new_lower = self.lower * mul_dim2d
        new_dim2d = self.dim2d * mul_dim2d
        new_upper = new_lower + Coord(new_dim2d.x-1, new_dim2d.y-1)
        return Rect(new_lower, new_upper)

    def __floordiv__(self, dim2d):
        new_lower = self.lower // dim2d
        new_dim2d = self.dim2d // dim2d
        new_upper = new_lower + Coord(new_dim2d.x-1, new_dim2d.y-1)
        return Rect(new_lower, new_upper)

    def get_width_height(self):
        return self.dim2d

    def get_tot_size(self):
        return self.dim2d.get_tot_size()

    def increase_size (self, incr_dim2d):
        self.dim2d += incr_dim2d
        self.upper += incr_dim2d
        return self

    def get_overlap (self, another_rect):
        new_lower_x = max(self.lower.x, another_rect.lower.x)
        new_lower_y = max(self.lower.y, another_rect.lower.y)
        new_upper_x = min(self.upper.x, another_rect.upper.x)
        new_upper_y = min(self.upper.y, another_rect.upper.y)
        #print ("new_lower_x=%d new_lower_y=%d new_upper_x=%d new_upper_y=%d"%\
        #       (new_lower_x,new_lower_y,new_upper_x,new_upper_y))
        return Rect(Coord(new_lower_x, new_lower_y),\
                           Coord(new_upper_x, new_upper_y))

    def get_offset_from (self, another_rect):
        return self.lower - another_rect.lower

    def set_waveop_pattern (self, waveop, prefix, stride):
        print(self)
        waveop[prefix + "x_step"] = stride.x 
        waveop[prefix + "x_num"] = self.dim2d.x
        waveop[prefix + "y_step"] = stride.y * self.dim2d.x
        waveop[prefix + "y_num"] = self.dim2d.y
        return waveop

    def set_lower(self, new_lower):
        self.lower = copy.copy(new_lower)
        self.upper = new_lower + Coord(self.dim2d.x - 1, self.dim2d.y - 1)

    def snap_rect_to_stride_grid(self, origin, stride):
        self.lower.snap_up_nearest_grid(origin, stride)
        self.upper.snap_down_nearest_grid(origin, stride)
        self.__init__(self.lower, self.upper)
        
class TestRect(unittest.TestCase):
    def test_coord0(self):
        # Check coord arithmetic
        a = Coord(10, -20)
        b = Coord(100, 200)
        c = b - a
        #print(str(c))
        self.assertEqual(c == Coord(100-10, 200+20), True)
        # Check heigh/width computation
        r = Rect(a, b)
        #(h,w) = r.height_width()
        #self.assertEqual(c + Coord(1,1) == Coord(w, h), True)
        # Check overlap computation
        r2 = Rect(Coord(0,0),Coord(20,20))
        x = r.get_overlap(r2)
        self.assertEqual(x == Rect(Coord(10,0),Coord(20,20)), True)
        offset = x.get_offset_from(r)
        #print(offset)
        self.assertEqual(offset == Coord(0, 20), True)
        # Check subrectangle
        r3 = Rect(Coord(10,0),Coord(29,49))
        self.assertEqual(r3.is_empty, False)
        s = r3.set_waveop_pattern({}, "dst_", Coord(2,5))
        print(s)
        r3 += Coord(100,100)
        print(r3)
        self.assertEqual(r3.lower == Coord(110, 100), True)
        self.assertEqual(r3.upper == Coord(129, 149), True)
        size = Dim2D(101,201)
        self.assertEqual(b.make_dim2d() == size, True)
        newsize = size + Dim2D(50,50)
        self.assertEqual(newsize.get_tot_size(), 151*251)
        c4 = Coord(0,0)
        c4.snap_up_nearest_grid(Coord(-1,-2), Dim2D(2,2))
        print(c4)
        self.assertEqual(c4 == Coord(1,0), True)
        r5 = Rect(Coord(200,0), Coord(100,0))
        self.assertEqual(r5.is_empty, True)

""" Class to manage head and tail pointers for circular buffer with two modes (for Middle Scheduler v1):
    - endzone_sz = 0: Normal mode: advance pointer until capacity, where it wraps to 0
    - endzone_sz > 0: End-sink mode: advance pointer until endzone, where it wraps to (capacity - endzone_sz)
    - endzone_sz < 0: No wrap mode: advance pointer until capacity, where it wraps to 0, but there's no overflow error if tail hits head, only warning
"""    
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

""" Class to manage N-dimensional shapes
"""
class ShapeDims():
    supported_dims = set(["N", "H", "W", "C", "M", "R", "S"])

    def __init__(self, format_str, shape_tuple):
        var_list = vars(self)            
        self.dim = {}
        self.axis = {}
        self.format_str = format_str
        self.shape_tuple = tuple(shape_tuple)
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

    # overload the [] operator
    def __getitem__(self, key):
        return self.dim[key]

""" Class to manage file-related parameters
"""
class FileParams():
    current_file_id = 0
    chunk_sz_limit = 2048

    def __init__(self, file_name, file_dims, data_type, op_params, args=None, contain_weights=False):
        self.args = args
        self.layer_name = "DEPRECATED"
        self.contain_weights = contain_weights # True for weights and bias and constants (used to tie-off BiasAdd in standalone Act)
        self.input_layer_ifmap = False
        self.final_layer_ofmap = False
        self.file_id = FileParams.current_file_id
        FileParams.current_file_id += 1
        self.file_name = file_name
        self.file_loaded = False
        self.file_dims = file_dims
        self.dram_data = None
        self.file_sz = 0
        self.file_addr_skip_per_batch_item = 0
        self.item_sz = data_type_to_item_sz(data_type)
        self.data_type = data_type
        self.chunk_sz = -1
        self.chunk_sz_padded = -1
        self.tot_partition_usage_sz = -1
        self.tot_partition_usage_sz_padded = -1
        self.tot_partition_usage_elems_padded = -1
        self.tot_num_chunks = -1
        self.fmap_data_len = -1
        self.fmap_data_len_padded = -1
        self.fmap_num_chunks = -1
        self.fmap_channels_folds = 1
        self.fmap_last_chunk_sz = -1
        self.batch_item_partition_usage_sz = -1
        self.batch_item_partition_usage_sz_padded = -1
        self.batch_item_partition_usage_elems_padded = -1
        self.batch_item_num_chunks = -1
        self.mapped_params = None
        self.file_addr_skip_per_fmap_fold = -1
        # Keeping a list of ops that writes to the same OFMAP, so that all can be transfered if OFMAP file is combined with another
        self.writers_of_shared_fmap = []
        self.readers_of_shared_fmap = []
        self.consumers = []
        self.compute_params(op_params, args)
        self.chunk_alignment_info = []

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

    def compute_params(self, op_params, args):
        # Single FMAP elem count (unified formula for weights and FMAP)
        fmap_elem_count = self.file_dims.R * self.file_dims.S * self.file_dims.M * self.file_dims.H * self.file_dims.W
        self.fmap_data_len = fmap_elem_count * self.item_sz
        self.stride = op_params.stride
        self.weights_S_dim = self.file_dims.S
        # per kaena-85, use noodle shapes for tiles
        # need to guard against small EF and build noodle tile to enable higher state buffer efficiency
        self.fmap_full_tilex_sz = min(self.file_dims.W, PEArray.MAX_WAVE_SIZE)
        self.fmap_full_tiley_sz = min(self.file_dims.H, PEArray.MAX_WAVE_SIZE // self.fmap_full_tilex_sz)
        # Chunk (aka atom) size computation for weights
        if self.file_dims.has_M:
            m_data_len = self.file_dims.M * self.item_sz
            sm_data_len = self.file_dims.S * m_data_len
            folding_multiple = (self.file_dims.C // PEArray.NUM_ROWS) * (self.file_dims.M // PEArray.NUM_COLS)
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
                input_fmap_full_tiley_sz = self.fmap_full_tiley_sz * op_params.stride.y
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
        self.fmap_channels_folds            = ceildiv(self.file_dims.C, PEArray.NUM_ROWS)  # num of folds (same as lowercase "c" computed elsewhere):
        self.batch_item_partition_usage_sz  = self.fmap_data_len * self.fmap_channels_folds
        self.tot_partition_usage_sz         = self.batch_item_partition_usage_sz * self.file_dims.N
        self.fmap_num_chunks                = ceildiv(self.fmap_data_len, self.chunk_sz)
        self.fmap_count                     = min(self.file_dims.C, PEArray.NUM_ROWS)
        self.fmap_last_fold_channels        = self.file_dims.C % PEArray.NUM_ROWS
        if self.fmap_last_fold_channels == 0: 
            self.fmap_last_fold_channels = PEArray.NUM_ROWS
        self.batch_item_num_chunks          = self.fmap_num_chunks * self.fmap_channels_folds
        self.tot_num_chunks                 = self.batch_item_num_chunks * self.file_dims.N
        self.fmap_last_chunk_sz             = self.fmap_data_len % self.chunk_sz
        if self.fmap_last_chunk_sz == 0:    
            self.fmap_last_chunk_sz         = self.chunk_sz
        self.file_addr_skip_per_fmap_fold   = self.fmap_data_len * min(self.file_dims.C, PEArray.NUM_ROWS)
        # Default unpadded sizes for internal FMAPs (see compute_padded_sizes for weights/input/output FMAPs)
        self.chunk_sz_padded                = self.chunk_sz                
        if (args.nname == 'generic'):
          self.fmap_data_len_padded           = align_addr_NB(self.fmap_data_len,4)
        else:
          self.fmap_data_len_padded           = self.fmap_data_len
        self.compute_padded_sizes()

    def compute_padded_sizes(self):
        # kaena-643: pad sizes to 8B to satisfy HW 8B alignment requirement             
        # Only for weights/bias, input IFMAP and final OFMAP.
        # Internal layers will gang-up pairs of chunks (FP16) to satisfy 4B alignment requirement.
        if self.contain_weights or self.final_layer_ofmap or self.input_layer_ifmap:
            self.chunk_sz_padded                      = align_addr_8B(self.chunk_sz)                
            self.fmap_data_len_padded                 = align_addr_8B(self.fmap_data_len)
        self.batch_item_partition_usage_sz_padded     = self.fmap_data_len_padded * self.fmap_channels_folds
        self.batch_item_partition_usage_elems_padded  = self.fmap_data_len_padded * self.fmap_channels_folds // self.item_sz
        self.tot_partition_usage_sz_padded            = self.batch_item_partition_usage_sz_padded * self.file_dims.N
        self.tot_partition_usage_elems_padded         = self.batch_item_partition_usage_elems_padded * self.file_dims.N
        print("INFO: file %s shape %s chunk_sz %d chunk_sz_padded %d fmap_data_len %d fmap_data_len_padded %d tot_partition_usage_sz %d batch_item_partition_usage_sz %d (compute_padded_sizes)"%(self.file_name, str(self.file_dims.shape_tuple), self.chunk_sz, self.chunk_sz_padded, self.fmap_data_len, self.fmap_data_len_padded, self.tot_partition_usage_sz, self.batch_item_partition_usage_sz))

#    def compute_alignment_info(self):
#        for i in self.tot_num_chunks:
#            sb_addr = self.get_sb_addr_from_chunk_id(

""" Class to hold map information related to a file
"""
class MappedParams():
    def __init__(self, N, start_addr, region_sz, num_region_chunks, num_file_chunks_per_batch_item, end_addr, modify_in_place, readers):
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
        self.dirty = []
        self.chunk_alignment_info = []
        for i in range(num_file_chunks_per_batch_item*N):
            self.dirty.append(False)
        # taemk
        # Mark if an instance of MappedParams has been consumed by its reader.
        # One example of this field is that when all readers have already
        # consumed the instance, we know that the mapped space in SB can be
        # freed.
        self.consumed_by_readers = dict()
        if (readers != None):
            self.init_consumers(readers)

    def init_consumers (self, consumers):
        print("taemk::MappedParams::init_consumers::readers ", end = "")
        for r in consumers:
            print ("%s "%r.data['layer_name'], end="")
            self.consumed_by_readers[r] = False
        print("")

    def is_consumed_by_all_readers(self):
        all_consumed = True
        for (k, v) in self.consumed_by_readers.items():
            all_consumed &= v
        return all_consumed

    def mark_consumed (self, reader):
        assert(reader in self.consumed_by_readers),\
          ("reader %s is not in reader list"%reader.data['layer_name'])
        self.consumed_by_readers[reader] = True

"""Class to represent a morsel of data
"""
class SbMorsel():
    def __init__(self):
        self.file_id = -1
        self.writer_id = -1
        self.reader_id = -1
        self.chunk_id = -1
        self.batch_item = -1

""" Class to manage mapping file to SB regions
"""
class FileMapper():
    def __init__(self, sb_partition_sz, data_type, args):
        self.item_sz         = data_type_to_item_sz(data_type)
        self.sb_partition_sz = sb_partition_sz
        self.data_type       = data_type
        self.file_params_list = {}
        self.morsels = [SbMorsel() for i in range(sb_partition_sz)]
        self.dramsaves = dict()
        self.enable_eviction = args.enable_eviction
        self.args = args

    def check_overlap(self, region0_start, region0_sz, region1_start, region1_sz):
        #print("DBG: checking overlap: region 0 start %d sz %d, region 1 start %d sz %d"%(region0_start, region0_sz, region1_start, region1_sz))
        if (region0_start <= region1_start):
            if (region0_start + region0_sz > region1_start):
                return True
        else:    
            if (region1_start + region1_sz > region0_start):
                return True
        return False

    def check_overlap100(self, region0_start, region0_sz, region1_start, region1_sz):
        return (region0_start == region1_start) and (region0_sz == region1_sz)

    """Adjust and return region0_start if there's overlap
    """
    def adjust0_if_overlap(self, region0_start, region0_sz, region1_start, region1_sz, min_region_start):
        region0_start_adjusted = align_addr_64B(region0_start)
        if region0_start_adjusted + region0_sz > self.sb_partition_sz:
            region0_start_adjusted = align_addr_64B(min_region_start)
        if self.check_overlap(region0_start, region0_sz, region1_start, region1_sz):
            region0_start_adjusted = align_addr_64B(region1_start + region1_sz)
            if region0_start_adjusted + region0_sz > self.sb_partition_sz:
                region0_start_adjusted = align_addr_64B(min_region_start)
        return region0_start_adjusted

    def skip_unfreed_region (self, cur_knode, start_addr):
        st = start_addr
        while 1:
            fid = self.morsels[st].file_id
            if (fid == -1):
                break
            else:
                fmap = self.file_params_list[fid]
                print("FileMapper::skip_unfreed_region::cur_knode = %s"%(
                    cur_knode.data['layer_name']))
                if (fmap.mapped_params.is_consumed_by_all_readers() == True):
                    print ("FileMapper::skip_unfreed_region::", end = " ")
                    print ("%s starting from %d is completely consumed"%(
                        fmap.file_name, st))
                    break;
                else:
                    print ("FileMapper::skip_unfreed_region::", end = " ")
                    print ("%s starting from %d is not consumed yet"%(
                        fmap.file_name, st), end = ", ")
                    for r in fmap.readers_of_shared_fmap:
                        if (fmap.mapped_params.consumed_by_readers[r] == False):
                            print ("%s", end = ", ")
                    print ("need to read %s"%fmap.file_name)
                    st = fmap.mapped_params.end_addr + fmap.item_sz
                    assert(st != start_addr)
        return st
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
        adj_region_sz = file_params.tot_partition_usage_sz_padded if region_sz == 0 else region_sz
        if not wrap_around:
            # If not wrapping around, check that file can fit into alloted region
            if file_params.tot_partition_usage_sz_padded > adj_region_sz:
                raise RuntimeError("File %s size %d is larger than region size %d, and wrap around is not enabled"%(file_params.file_name, file_params.tot_partition_usage_sz_padded, adj_region_sz))
            # Compute number of chunks, including the last odd chunk
            num_region_chunks = file_params.tot_num_chunks
            adj_region_sz     = file_params.tot_partition_usage_sz_padded
        else:
            if adj_region_sz >= file_params.fmap_data_len_padded:
                num_region_fmaps  = adj_region_sz // file_params.fmap_data_len_padded
                num_region_chunks = num_region_fmaps * file_params.fmap_num_chunks
                adj_region_sz     = num_region_fmaps * file_params.fmap_data_len_padded
            else:                
                # If wrapping around and FMAP too big, waste the last odd chunk and just make sure to have chunk_sz pieces
                num_region_chunks = adj_region_sz // file_params.chunk_sz_padded
                adj_region_sz     = num_region_chunks * file_params.chunk_sz_padded

        # Check number of chunks is reasonable            
        if num_region_chunks == 0:                
            raise RuntimeError("Region size %d cannot accomodate chunk size of %d"%(adj_region_sz, file_params.chunk_sz))
        # check end address            
        end_addr = start_addr + adj_region_sz - file_params.item_sz
        if end_addr >= self.sb_partition_sz:
            raise RuntimeError("End address %d falls outside partition size %d"%(end_addr, self.sb_partition_sz))
        # Save mapped information            
        print("taemk::FileParams to be consumed = %s"%file_params.file_name)
        file_params.mapped_params = MappedParams(file_params.file_dims.N, start_addr, adj_region_sz, num_region_chunks, file_params.batch_item_num_chunks, end_addr, modify_in_place, file_params.consumers)
        # Save file params in a list
        self.file_params_list[file_params.file_id] = file_params
        return end_addr + file_params.item_sz

    def get_chunk_id_from_file_addr(self, file_params, batch_item, addr):
        assert(file_params.file_dims.format_str == "NCHW" or file_params.file_dims.format_str == "CRSM")
        assert(addr >= 0)
        assert(addr >= batch_item * file_params.file_addr_skip_per_batch_item)
        addr_adj         = addr - batch_item * file_params.file_addr_skip_per_batch_item
        fold_idx         = addr_adj // file_params.file_addr_skip_per_fmap_fold 
        fold_offset      = addr_adj % file_params.file_addr_skip_per_fmap_fold 
        # only consider fold_offset in partition 0 (take care of cases where channel starts in mid partition) 
        fold_offset_part0 = fold_offset % file_params.fmap_data_len
        chunk_id_in_fold = fold_offset_part0 // file_params.chunk_sz
        chunk_id         = fold_idx * file_params.fmap_num_chunks + chunk_id_in_fold
        chunk_id        += batch_item * file_params.batch_item_num_chunks 
        return chunk_id

    def get_file_addr_from_chunk_id(self, file_params, batch_item, chunk_id):
        assert(file_params.file_dims.format_str == "NCHW" or file_params.file_dims.format_str == "CRSM")
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
        if file_params.mapped_params.region_sz >= file_params.fmap_data_len_padded:
            # if region_sz >= fmap_data_len, earlier computation guarantees that region_sz is multiple of fmap_data_len
            fold_idx         = chunk_id_offset // file_params.fmap_num_chunks
            chunk_id_in_fold = chunk_id_offset % file_params.fmap_num_chunks
            # kaena- : pad chunk size and FMAP data length to align to 8B (hard requirement)
            sb_addr          = fold_idx * file_params.fmap_data_len_padded + chunk_id_in_fold * file_params.chunk_sz_padded
        else:            
            sb_addr         = chunk_id_offset * file_params.chunk_sz_padded
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
        last_writer = -1
        last_reader = -1
        last_writer_except_current_nonload_waveop = -1
        list_of_waveops = []
        eviction_dict = dict()
        for i in range(start_chunk_id, end_chunk_id + 1):
            # TODO: fix start_fmap_addr to match start_addr
            start_fmap_addr = self.get_sb_addr_from_chunk_id(file_params, batch_item, i)
            chunk_len = self.get_chunk_len_from_chunk_id(file_params, batch_item, i)
            # kaena-643: mark also the padded region due to 8B alignment requirement
            if chunk_len == file_params.chunk_sz:
                chunk_len = file_params.chunk_sz_padded
            end_fmap_addr = start_fmap_addr + chunk_len
            eviction_dict.clear()
            for j in range(start_fmap_addr, end_fmap_addr, file_params.item_sz):
                sb_addr = j
                if (sb_addr >= start_sb_addr and sb_addr <= end_sb_addr) or (file_params.args is not None and file_params.args.relax_dependencies):
                    # return last of wniters/readers for dependency
                    last_writer = max(last_writer, self.morsels[sb_addr].writer_id)
                    last_reader = max(last_reader, self.morsels[sb_addr].reader_id)
                    if (self.morsels[sb_addr].writer_id != nonload_waveop_id):
                        last_w_id = last_writer_except_current_nonload_waveop
                        last_writer_except_current_nonload_waveop =\
                                max(last_w_id,self.morsels[sb_addr].writer_id)
                    # Evict old owner                    
                    # TODO: remap atom_id back to file chunk ID
                    file_id = self.morsels[sb_addr].file_id
                    chunk_id = self.morsels[sb_addr].chunk_id
                    owner_batch_item = self.morsels[sb_addr].batch_item
                    if file_id in self.file_params_list:
                        if (file_id != file_params.file_id) or (chunk_id != i):
                            #print("INFO: batch item %d: Writer waveop (non-load) ID %d is writing chunk_id %d (addr %d) of file %s (file ID %d), clearing previous owner which is chunk_id %d of file %s (file ID %d)"%(batch_item, nonload_waveop_id, i, sb_addr, file_params.file_name, file_params.file_id, chunk_id, self.file_params_list[file_id].file_name, file_id))
                            # taemk : when SB region for chunk i has been
                            # updated by other file_param, we need to evict it.
                            # Thus, we collect such file params here.
                            self.GetEvictionChunk(file_id\
                                                  , chunk_id\
                                                  , eviction_dict)
                            #self.file_params_list[file_id].mapped_params.chunk_is_mapped[chunk_id] = False
                    elif file_id != -1:
                        raise RuntimeError("File ID %d not found in list of file_params"%(file_id))
                    #print("INFO: batch item %d: Writer waveop (non-load) ID %d is writing chunk_id %d (addr %d) of file %s (file ID %d), clearing previous writer_id %s or reader_id %d, and replacing with writer_id %d"%(batch_item, nonload_waveop_id, i, sb_addr, file_params.file_name, file_params.file_id, self.morsels[sb_addr].writer_id, self.morsels[sb_addr].reader_id, nonload_waveop_id))
                    self.morsels[sb_addr].file_id = file_params.file_id
                    self.morsels[sb_addr].writer_id = nonload_waveop_id
#                    if (self.args.nname != 'generic' and\
                    if (self.args.enable_cleanup == False):
                      self.morsels[sb_addr].reader_id = -1
                    self.morsels[sb_addr].chunk_id = i
                    self.morsels[sb_addr].batch_item = batch_item
                # if data is not in SB, map region
                #if not file_params.mapped_params.chunk_is_mapped[batch_item][i]:
                #    file_params.mapped_params.chunk_is_mapped[batch_item][i] = True
                if not file_params.mapped_params.chunk_is_mapped[i]:
                    file_params.mapped_params.chunk_is_mapped[i] = True
                # taemk : Tracking the dirtiness of a chunk
                file_params.mapped_params.dirty[i] = True
                    #print("INFO: batch item %d: Writer waveop (non-load) ID %d is writing chunk_id %d (start %d, end %d) of file %s"%(batch_item, nonload_waveop_id, i, start_fmap_addr, end_fmap_addr, file_params.file_name))
            prev_waveops = []
            latest_accessor = max(last_writer, last_reader)
            if (self.enable_eviction == True):
                # allow for the fact that when generating matmul waveops, there could be read to the same space before waveop is added to nonload_waveop_list
                if latest_accessor >= 0 and latest_accessor < len(nonload_waveop_list):
                    latest_accessor_name = nonload_waveop_list[latest_accessor]['waveop_name']
                    if latest_accessor_name not in prev_waveops:
                        prev_waveops.append(latest_accessor_name)
                # Perform eviction and read back data from i-th chunk of a file
                # that is newly written in SB
                # taemk : 08-27-2018
                # Previous waveops for eviction waveop is the last writer to the
                # region to be evicted. Note that current waveop calling
                # this method should be excluded from the last writer. That's why
                # last_writer_except_current_nonload_waveop is created.
                prev_id = last_writer_except_current_nonload_waveop
                prev_op_evict = []
                if (prev_id != -1):
                    prev_op_evict=[nonload_waveop_list[prev_id]['waveop_name']]
                list_of_waveops, evicted =\
                        self.PerformEviction(start_fmap_addr
                                             ,end_fmap_addr
                                             ,nonload_waveop_id
                                             ,eviction_dict\
                                             ,batch_item\
                                             ,prev_op_evict\
                                             ,file_params\
                                             ,i)
                list_of_waveops.extend(prev_waveops)
        return (last_writer, last_reader, list_of_waveops)

    def UpdateDRAMSaves (self, file_params, chunk_id):
        prev_save = None
        if ((file_params, chunk_id) in self.dramsaves):
            prev_save = self.dramsaves[(file_params, chunk_id)]
            self.dramsaves.pop((file_params, chunk_id))
        return prev_save

    def UpdateMorselOwner (self, start_fmap_addr, end_fmap_addr
                           , nonload_waveop_id, evicted_file_params
                           , evicted_chunk_id, batch_item):
        if (evicted_chunk_id % 2 == 0):
            start_chunk_id = evicted_chunk_id
            len_multiplier = 1
        else:
            start_chunk_id = evicted_chunk_id - 1
            len_multiplier = 2
        assert(start_chunk_id >= 0)
        evicted_start_fmap_addr = self.get_sb_addr_from_chunk_id(
            evicted_file_params, batch_item, start_chunk_id)
        evicted_chunk_len = self.get_chunk_len_from_chunk_id(
            evicted_file_params, batch_item, evicted_chunk_id)
        if evicted_chunk_len == evicted_file_params.chunk_sz:
            evicted_chunk_len = evicted_file_params.chunk_sz_padded
        evicted_chunk_len *= len_multiplier
        evicted_end_fmap_addr = evicted_start_fmap_addr + evicted_chunk_len
        addr_range = range(evicted_start_fmap_addr
                           ,evicted_end_fmap_addr
                           ,evicted_file_params.item_sz)
        for sb_addr in addr_range:
            if (not (sb_addr >= start_fmap_addr and sb_addr <= end_fmap_addr)):
                #self.morsels[sb_addr].file_id = evicted_file_params.file_id
                #self.morsels[sb_addr].writer_id = -1
                self.morsels[sb_addr].reader_id = nonload_waveop_id
                #self.morsels[sb_addr].chunk_id = evicted_chunk_id
                #self.morsels[sb_addr].batch_item = batch_item
        return nonload_waveop_id

    # Evict chunks in eviction_chunks and then reload chunk associated with
    # load_file_params
    def EvictChunk (self, file_params, batch_item, chunk_id, prev_waveops):
        if (chunk_id % 2 == 0):
            start_chunk_id = chunk_id
            evict_two_chunks = False
        else:
            start_chunk_id = chunk_id - 1
            evict_two_chunks = True
        assert(start_chunk_id >= 0)
        ev_waveop = self.gen_dram_save_waveop(file_params
                                              ,batch_item
                                              ,start_chunk_id
                                              ,prev_waveops
                                              ,evict_two_chunks
                                             )
        self.dramsaves[(file_params, chunk_id)] = ev_waveop
        file_params.mapped_params.chunk_is_mapped[chunk_id]=False
        if (evict_two_chunks == True):
            ev_waveop['#comment']="this is evicted chunk %d and %d"%(
                chunk_id - 1, chunk_id)
            self.dramsaves[(file_params, chunk_id - 1)] = ev_waveop
            file_params.mapped_params.chunk_is_mapped[chunk_id - 1]=False
        else:
            ev_waveop['#comment']="this is evicted chunk %d"%chunk_id
        return ev_waveop

    def PerformEviction (self\
                         ,start_fmap_addr
                         ,end_fmap_addr
                         ,most_recent_nonload_waveop_id
                         ,eviction_chunks\
                         ,batch_item
                         ,prev_waveops\
                         ,load_file_params = None\
                         ,chunk_id = -1\
                        ):
        #import inspect
        #print ("INFO: %s is calling PerformEviction"%inspect.stack()[1][3])
        eviction_waveops = []
        if (self.enable_eviction == True):
            ev_id = most_recent_nonload_waveop_id + 1
            for (file_params, chunk_ids) in eviction_chunks.items():
                if (not hasattr(chunk_ids, "__len__")):
                    chunk_ids = [chunk_ids]
                for i_chunk_id in chunk_ids:
                    print("INFO: chunk_id %d of %s is evicted"\
                          %(i_chunk_id, file_params.file_name))
                    #ev_waveop = self.gen_dram_save_waveop(file_params\
                    #                                      ,batch_item\
                    #                                      ,i_chunk_id\
                    #                                      ,prev_waveops)
                    #ev_waveop['#comment']="this is evicted chunk %d"%i_chunk_id
                    #self.dramsaves[(file_params, i_chunk_id)] = ev_waveop
                    #file_params.mapped_params.chunk_is_mapped[i_chunk_id]=False
                    #eviction_waveops.append(ev_waveop)
                    ev_waveop = self.EvictChunk(
                        file_params, batch_item, i_chunk_id, prev_waveops)
                    eviction_waveops.append(ev_waveop)
                    self.UpdateMorselOwner(start_fmap_addr, end_fmap_addr
                                           , ev_id, file_params, i_chunk_id
                                           , batch_item)
                    ev_id += 1
            if (len(eviction_chunks) != 0 and load_file_params != None):
                if (chunk_id % 2 == 0):
                    loading_chunk_id = chunk_id
                    read_two_chunks = False
                else:
                    loading_chunk_id = chunk_id - 1
                    read_two_chunks = True
                assert(loading_chunk_id >= 0)
                prev_save = self.UpdateDRAMSaves(
                    load_file_params, loading_chunk_id)
                dram_read_prevs = self.GetWaveOpNames(eviction_waveops)
                if (prev_save != None):
                    #print (prev_save['waveop_name'])
                    dram_read_prevs.append(prev_save['waveop_name'])
                    reload_waveop = self.gen_dram_read_waveop(\
                                                              load_file_params
                                                              ,batch_item
                                                              ,loading_chunk_id
                                                              ,dram_read_prevs
                                                              ,0
                                                              ,read_two_chunks
                                                             )
                    load_file_params.mapped_params.chunk_is_mapped[chunk_id] =\
                            True
                    load_file_params.mapped_params.chunk2waveop_map[chunk_id] =\
                            reload_waveop 
                    if (read_two_chunks == True):
                        reload_waveop['#comment'] =\
                                "this is reloaded chunk %d and %d"%\
                                (chunk_id-1, chunk_id)
                        load_file_params.mapped_params.\
                                chunk_is_mapped[chunk_id-1] = True
                        load_file_params.mapped_params.\
                                chunk2waveop_map[chunk_id-1] = reload_waveop 
                    else:
                        reload_waveop['#comment'] =\
                                "this is reloaded chunk %d"%(chunk_id)
                    eviction_waveops.append(reload_waveop)
        return (eviction_waveops, len(eviction_waveops) != 0)
        #if (eviction_waveops == []):
        #    return (prev_waveops, False)
        #else:
        #    return (eviction_waveops, True)

    def GetWaveOpNames (self, waveops):
        n = []
        for w in waveops:
            if (w['waveop_name'] != None):
                n.append(w['waveop_name'])
            else:
                n.append(w)
        return n

    def GetEvictionChunk (self\
                          ,evicted_file_id\
                          ,evicted_chunk_id\
                          ,eviction_container):
        org_fparam = self.file_params_list[evicted_file_id]
        if (self.args.nname == 'generic'):
          if (self.enable_eviction == True):
              if (org_fparam.mapped_params.chunk_is_mapped[evicted_chunk_id] ==\
                  True):
                  if (org_fparam.mapped_params.dirty[evicted_chunk_id] == True):
                      org_fparam.mapped_params.dirty[evicted_chunk_id] = False
                      if (org_fparam in eviction_container):
                          eviction_container[org_fparam].append(evicted_chunk_id)
                      else:
                          eviction_container[org_fparam] = [evicted_chunk_id]
              org_fparam.mapped_params.\
                      chunk_is_mapped[evicted_chunk_id] = False
        else:
            org_fparam.mapped_params.\
                    chunk_is_mapped[evicted_chunk_id] = False
        return

    # Save data to file 
    def flush_file (self, nonload_waveop_id, nonload_waveop_list, file_params, batch_item):
        nonload_waveop_id_tmp = nonload_waveop_id
        start_chunk_id      = batch_item * file_params.batch_item_num_chunks
        end_chunk_id        = start_chunk_id + file_params.batch_item_num_chunks - 1
        num_chunks          = file_params.batch_item_num_chunks
        if num_chunks > file_params.mapped_params.num_region_chunks:
            raise RuntimeError("Number of chunks written %d is larger than mapped number of chunks %d"%(num_chunks, file_params.mapped_params.num_region_chunks))
        list_of_waveops = []
        eviction_dict = dict()
        for i in range(start_chunk_id, end_chunk_id + 1):
            list_of_writers_per_chunk = []
            list_of_readers_per_chunk = []
            start_fmap_addr = self.get_sb_addr_from_chunk_id(file_params, batch_item, i)
            end_fmap_addr = start_fmap_addr + self.get_chunk_len_from_chunk_id(file_params, batch_item, i)
            eviction_dict.clear()
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
                    #self.GetEvictionChunk(file_id, chunk_id, eviction_dict)
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
            #list_of_accessors = list_of_writers_per_chunk + list_of_readers_per_chunk
            list_of_accessors = list_of_writers_per_chunk
            prev_waveops = []
            if list_of_accessors != []:
                # include all accessors for saving to DRAM, instead of just the latest accessor
                for accessor in list_of_accessors:
                    # allow for the fact that when generating matmul waveops, there could be read to the same space before waveop is added to nonload_waveop_list
                    if accessor >= 0 and accessor < len(nonload_waveop_list):
                        accessor_name = nonload_waveop_list[accessor]['waveop_name']
                        if accessor_name not in prev_waveops:
                            prev_waveops.append(accessor_name)
            #evicted_waveops, evicted =\
            #        self.PerformEviction(\
            #                             eviction_dict\
            #                             , batch_item
            #                             , prev_waveops\
            #                             , file_params\
            #                             , i\
            #                            )
            #if (len(evicted_waveops) > 0):
            #    new_dram_waveop = self.gen_dram_save_waveop(file_params, batch_item, i, [evicted_waveops[-1]['waveop_name']])
            #else:
                new_dram_waveop = self.gen_dram_save_waveop(file_params, batch_item, i, prev_waveops)
            #evicted_waveops.append(new_dram_waveop)
            list_of_waveops.append(new_dram_waveop)
            #list_of_waveops += evicted_waveops
            #print("INFO: batch item %d: DRAM saver (SBAtomSave) waveop (non-load) ID %d is reading chunk_id %d (start %d, end %d) of file %s"%(batch_item, nonload_waveop_id_tmp, i, start_fmap_addr, end_fmap_addr, file_params.file_name))
            # trace waveop_id for newly created SBAtomSaves (to trace dependency so that new writer to same space need to wait for this save to complete)
            nonload_waveop_id_tmp += 1
        return list_of_waveops

    # Always read the maximum number of channels (min(C, 128))
    # TODO: add case for replication
    def read_file_data_region(self, nonload_waveop_id, nonload_waveop_list, file_params, batch_item, start_addr, length, repl_multiple_of_C=1, dont_put_prev_ops = False):
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
        last_writer = -1
        last_reader = -1
        list_of_waveops = []
        eviction_dict = dict()
        for i in range(start_chunk_id, end_chunk_id + 1):
            start_fmap_addr = self.get_sb_addr_from_chunk_id(file_params, batch_item, i)
            chunk_len = self.get_chunk_len_from_chunk_id(file_params, batch_item, i)
            # kaena-643: mark also the padded region due to 8B alignment requirement
            if chunk_len == file_params.chunk_sz:
                chunk_len = file_params.chunk_sz_padded
            end_fmap_addr = start_fmap_addr + chunk_len
            prev_file_id = -1
            prev_chunk_id = -1
            eviction_dict.clear()
            for j in range(start_fmap_addr, end_fmap_addr, file_params.item_sz):
                sb_addr = j
                # return last of wniters/readers for dependency
                last_writer = max(last_writer, self.morsels[sb_addr].writer_id)
                last_reader = max(last_reader, self.morsels[sb_addr].reader_id)
                # Evict old owner                    
                file_id = self.morsels[sb_addr].file_id
                chunk_id = self.morsels[sb_addr].chunk_id
                owner_batch_item = self.morsels[sb_addr].batch_item
                if file_id in self.file_params_list:
                    if (file_id != file_params.file_id) or (chunk_id != i):
                        #if (file_id != prev_file_id or\
                        #    chunk_id != prev_chunk_id):
                            #print("INFO: batch item %d: Reader waveop (non-load) ID %d is reading chunk_id %d (addr %d) of file %s (file ID %d), clearing previous owner which is chunk_id %d of file %s (file ID %d)"%(batch_item, nonload_waveop_id, i, sb_addr, file_params.file_name, file_params.file_id, chunk_id, self.file_params_list[file_id].file_name, file_id))
                        self.GetEvictionChunk(file_id, chunk_id, eviction_dict)
                        #self.file_params_list[file_id].mapped_params.chunk_is_mapped[chunk_id] = False
                elif file_id != -1:
                    raise RuntimeError("File ID %d not found in list of file_params"%(file_id))
                self.morsels[sb_addr].file_id = file_params.file_id
#                if (self.args.nname != 'generic' and\
                if (self.args.enable_cleanup == False):
                  self.morsels[sb_addr].writer_id = -1
                self.morsels[sb_addr].reader_id = nonload_waveop_id
                self.morsels[sb_addr].chunk_id = i
                self.morsels[sb_addr].batch_item = batch_item
                prev_file_id = file_id
                prev_chunk_id = chunk_id
            # if data is not in SB, issue DRAM loads
            #if not file_params.mapped_params.chunk_is_mapped[batch_item][i]:
            if not file_params.mapped_params.chunk_is_mapped[i]:
                file_params.mapped_params.chunk_is_mapped[i] = True
                # kaena-141,330: replication hack: squash unneeded weight reads waveops
                replication_squash = False
                if repl_multiple_of_C > 1 and file_params.file_dims.has_M:
                    num_of_filter_rows = repl_multiple_of_C // file_params.weights_S_dim
                    replication_squash = i != start_chunk_id
                    print("chunk %d num_of_filter_rows %d squash %d"%(i, num_of_filter_rows, replication_squash))
                # If modifying in place, don't create DRAM waveops for region
                if not file_params.mapped_params.modify_in_place and not replication_squash:
                    prev_waveops = []
                    latest_accessor = max(last_writer, last_reader)
                    # allow for the fact that when generating matmul waveops, there could be read to the same space before waveop is added to nonload_waveop_list
                    if latest_accessor >= 0 and latest_accessor < len(nonload_waveop_list):
                        latest_accessor_name = nonload_waveop_list[latest_accessor]['waveop_name']
                        if latest_accessor_name not in prev_waveops:
                            prev_waveops.append(latest_accessor_name)
                    prev_op_evict = []
                    if (last_writer != -1):
                        prev_op_evict =\
                            [nonload_waveop_list[last_writer]['waveop_name']]
                    evicted_waveops, evicted =\
                            self.PerformEviction(
                                                 start_fmap_addr
                                                 , end_fmap_addr
                                                 , nonload_waveop_id
                                                 , eviction_dict
                                                 , batch_item
                                                 , prev_op_evict
                                                )
                    prev_save = self.UpdateDRAMSaves(file_params, i)
                    dram_read_prevs = self.GetWaveOpNames(evicted_waveops)
                    if (prev_save != None):
                        dram_read_prevs.append(prev_save['waveop_name'])
                        prev_waveops.append(prev_save['waveop_name'])
                    if (len(evicted_waveops) > 0):
                        new_dram_waveop = self.gen_dram_read_waveop(file_params, batch_item, i, dram_read_prevs, repl_multiple_of_C)
                    else:
                        if (dont_put_prev_ops == False):
                            new_dram_waveop = self.gen_dram_read_waveop(file_params, batch_item, i, prev_waveops, repl_multiple_of_C)
                        else:
                            new_dram_waveop = self.gen_dram_read_waveop(file_params, batch_item, i, [], repl_multiple_of_C)
                    file_params.mapped_params.dirty[i] = False
                    evicted_waveops.append(new_dram_waveop)
                    list_of_waveops += evicted_waveops
                    file_params.mapped_params.chunk2waveop_map[i] = new_dram_waveop
                    #file_params.mapped_params.chunk_is_mapped[batch_item][i] = True
                    #print("INFO: batch item %d: Reader ID %d is reading chunk_id %d (start %d, end %d) of file %s, creating DRAM load waveops"%(batch_item, nonload_waveop_id, i, start_fmap_addr, end_fmap_addr, file_params.file_name))
                #else:
                #    print("INFO: batch item %d: Reader ID %d is reading chunk_id %d (start %d, end %d) of file %s, which is being modified in place, so not creating DRAM load waveops"%(batch_item, nonload_waveop_id, i, start_fmap_addr, end_fmap_addr, file_params.file_name))
        return (last_writer, last_reader, list_of_waveops)

    def gen_dram_read_waveop(self, file_params, batch_item, chunk_id, previous_waveops, repl_multiple_of_C, read_two_chunks = False):
        length          = self.get_chunk_len_from_chunk_id(file_params, batch_item, chunk_id)
        offset_in_file  = self.get_file_addr_from_chunk_id(file_params, batch_item, chunk_id)
        sb_addr         = self.get_sb_addr_from_chunk_id(file_params, batch_item, chunk_id)
        fmap_count      = self.get_fmap_count_from_chunk_id(file_params, batch_item, chunk_id)
        assert (fmap_count > 0)
        if (read_two_chunks == True):
            length_chunk_odd =\
                    self.get_chunk_len_from_chunk_id(
                        file_params, batch_item,chunk_id+1)
        else: length_chunk_odd = 0
        length += length_chunk_odd
        last_byte_offset = offset_in_file + (file_params.fmap_data_len * (fmap_count-1)) + length - file_params.item_sz
        assert (length > 0)           
        # IFMAP replication parameters
        stride = 1
        ifmap_replication_num_rows = 0
        ifmap_replication_resolution = 0
        ifmap_replication_step_bytes = 0
        if repl_multiple_of_C > 1:
            fmap_count = fmap_count * repl_multiple_of_C
            if file_params.file_dims.has_M:
                ifmap_replication_num_rows = file_params.file_dims.C
                ifmap_replication_resolution = file_params.file_dims.C
                ifmap_replication_step_bytes = file_params.file_dims.M * file_params.item_sz
                length = file_params.file_dims.M * file_params.item_sz  # TODO: adjust chunk size to match
                # compute last byte offset to check out of file bound
                last_byte_offset  = offset_in_file + length - file_params.item_sz
                last_byte_offset += ifmap_replication_step_bytes * (ceildiv(fmap_count, ifmap_replication_resolution) - 1)
                last_byte_offset += file_params.fmap_data_len * ((fmap_count-1)%ifmap_replication_resolution) 
            else:
                stride = file_params.stride.x
                ifmap_replication_num_rows = file_params.file_dims.C * file_params.weights_S_dim
                ifmap_replication_resolution = file_params.file_dims.C * file_params.stride.x
                ifmap_replication_step_bytes = (file_params.file_dims.W // file_params.stride.x) * file_params.item_sz
                length = length // (file_params.stride.x * file_params.stride.x) # TODO: adjust chunk size to match
                offset_in_file_batch_item = batch_item * file_params.file_addr_skip_per_batch_item
                # Adjust for the even/odd split
                offset_in_file = (offset_in_file - offset_in_file_batch_item) // (file_params.stride.x * file_params.stride.x) + offset_in_file_batch_item # TODO: adjust chunk size to match
                # compute last byte offset to check out of file bound
                last_byte_offset  = offset_in_file + length - file_params.item_sz
                last_byte_offset += file_params.fmap_data_len // (file_params.stride.x * file_params.stride.x)   # jump to the odd half
                last_byte_offset += ifmap_replication_step_bytes * (ceildiv(fmap_count, ifmap_replication_resolution) - 1)
                last_byte_offset += file_params.fmap_data_len * ((fmap_count-1)%file_params.file_dims.C)

        # Kaena-530: check that the last byte doesn't go outside of file
        if (file_params.args is not None and file_params.args.debug > 3): print("DBG: last_byte_offset %d file_sz %d"%(last_byte_offset, file_params.file_sz))
        if repl_multiple_of_C > 1:
            assert(last_byte_offset < file_params.file_sz + length*file_params.file_dims.C)
        else:            
            assert(last_byte_offset < file_params.file_sz)

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
        #assert_align_addr_sb_write(file_params.fmap_data_len)
        assert_align_addr_sb_write(sb_addr)
        return {
              'previous_waveops' : previous_waveops,
              'waveop_type'      : "SBAtomLoad",
              'waveop_name'      : waveop_name,
              'layer_name'       : file_params.layer_name,
              'sb_address'       : sb_addr,
              'data_type'        : file_params.data_type,
              'contain_weights'  : file_params.contain_weights,
              'ref_file'         : simout_file,
              'ref_file_sz'      : file_params.file_sz,
              'ref_file_format'  : file_params.file_dims.format_str,
              'ref_file_shape'   : file_params.file_dims.shape_tuple,
              'offset_in_file'   : offset_in_file,
              'length'           : length,
              'start_at_mid_part' : False,  # TODO: is this always false for loads?
              'num_partitions'    : fmap_count,  # if this is larger than C, replicate fmap_count/C times
              'partition_step_bytes': file_params.fmap_data_len,
              'stride'              : stride,
              'ifmap_replication_resolution' : ifmap_replication_resolution, 
              'ifmap_replication_num_rows' : ifmap_replication_num_rows,
              'ifmap_replication_step_bytes' : ifmap_replication_step_bytes,
            }

    def gen_dram_save_waveop(
        self, file_params, batch_item, chunk_id, previous_waveops
        , save_two_chunks = False):
        length          = self.get_chunk_len_from_chunk_id(file_params, batch_item, chunk_id)
        offset_in_file  = self.get_file_addr_from_chunk_id(file_params, batch_item, chunk_id)
        sb_addr         = self.get_sb_addr_from_chunk_id(file_params, batch_item, chunk_id)
        fmap_count      = self.get_fmap_count_from_chunk_id(file_params, batch_item, chunk_id)
        if (save_two_chunks == True):
            length +=\
                    self.get_chunk_len_from_chunk_id(
                        file_params, batch_item, chunk_id+1)
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
        #assert_align_addr_sb_read(file_params.fmap_data_len)
        assert_align_addr_sb_read(sb_addr)
        #if (waveop_name == 'trivnet_activation_1__Relu__0_NCHW-simout.npy_0'):
        #    import inspect
        #    print ("INFO: %s is calling gen_dram_save_waveop"%(
        #        inspect.stack()[1][3]))
        return {
              'previous_waveops' : previous_waveops,
              'waveop_type'      : "SBAtomSave",
              'waveop_name'      : waveop_name,
              'layer_name'       : file_params.layer_name,
              'sb_address'       : sb_addr,
              'data_type'        : self.data_type,
              'ref_file'         : simout_file,
              'ref_file_sz'      : file_params.file_sz,
              'ref_file_format'  : file_params.file_dims.format_str,
              'ref_file_shape'   : file_params.file_dims.shape_tuple,
              'offset_in_file'   : offset_in_file,
              'length'           : length,
              'start_at_mid_part' : False, #(tile_id.m_id%2) == 1,
              'ofmaps_fold_idx'  : 0,   # TODO: is this still needed?
              'batch_fold_idx'   : 0,   # TODO: is this still needed?
              'num_partitions'   : fmap_count,
              'partition_step_bytes': file_params.fmap_data_len,
              'last_save_of_file' : last_atom_of_file,
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
    class op_params_stride2():
        stride = Dim2D(2,2)

    class op_params_stride1():
        stride = Dim2D(1,1)

    def test_file_params_instantiation(self):
        shape_dims = ShapeDims("CRSM", [1,7,7,64]) 
        test_obj = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 896)
        shape_dims = ShapeDims("CRSM", [256,1,1,128]) 
        test_obj = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 256)
        self.assertEqual(test_obj.ravel_crsm(0,0,0,0), 0)
        self.assertEqual(test_obj.ravel_crsm(1,0,0,0), 128*test_obj.item_sz)
        shape_dims = ShapeDims("NHWC", [1,112,112,64]) 
        test_obj = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 1792)
        shape_dims = ShapeDims("NHWC", [1,55,55,128]) 
        test_obj = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride2)
        self.assertEqual(test_obj.chunk_sz, 1760)
        shape_dims = ShapeDims("NHWC", [1,55,55,128]) 
        test_obj = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 1760)
        #shape_dims = ShapeDims("NHWC", [1,55,55,128]) 
        #test_obj = FileParams("testfile.npy", shape_dims, "float16", 1210, self.op_params_stride1)
        #self.assertEqual(test_obj.chunk_sz, 880)
        shape_dims = ShapeDims("NHWC", [1,28,28,256]) 
        test_obj = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 1568)
        shape_dims = ShapeDims("NHWC", [1,14,14,256]) 
        test_obj = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 392)
        shape_dims = ShapeDims("NHWC", [1,7,7,256]) 
        test_obj = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride1)
        self.assertEqual(test_obj.chunk_sz, 98)
        self.assertEqual(test_obj.ravel_nchw(0,0,0,0), 0)
        self.assertEqual(test_obj.ravel_nchw(0,0,0,1), 256*test_obj.item_sz)
        self.assertEqual(test_obj.ravel_nchw(0,0,1,0), 7*256*test_obj.item_sz)
        shape_dims = ShapeDims("NHWC", [4,1,1,2048]) 
        test_obj = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride1)
        self.assertEqual(test_obj.tot_partition_usage_sz, 2048*4//128*test_obj.item_sz)
        self.assertEqual(test_obj.batch_item_partition_usage_sz, 2048//128*test_obj.item_sz)
        shape_dims = ShapeDims("NHWC", [4,55,55,512]) 
        test_obj = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride1)
        self.assertEqual(test_obj.tot_partition_usage_sz, 55*55*512*4//128*test_obj.item_sz)
        self.assertEqual(test_obj.batch_item_partition_usage_sz, 55*55*512//128*test_obj.item_sz)
        shape_dims = ShapeDims("NHWC", [4,1,1,1000]) 
        test_obj = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride1)
        self.assertEqual(test_obj.tot_partition_usage_sz, 64)
        self.assertEqual(test_obj.batch_item_partition_usage_sz, 16)
        shape_dims = ShapeDims("NHWC", [16,1,1,1000]) 
        test_obj = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride1)
        self.assertEqual(test_obj.tot_partition_usage_sz, 4*64)
        self.assertEqual(test_obj.batch_item_partition_usage_sz, 16)
        # Test padding and split (replication)
        shape_dims = ShapeDims("NCHW", [1,3,224,224]) 
        test_obj = FileParams("testfile_1_3_224_224.npy", shape_dims, "float16", self.op_params_stride2)
        self.assertEqual(test_obj.chunk_sz, 1792)
        test_obj.load_file()
        (new_file, new_shape) = pad_and_split_file("testfile_1_3_224_224.npy", "NCHW", 2, 3, 2, 3, 2)
        shape_dims = ShapeDims("NCHW", new_shape)
        test_obj = FileParams(new_file, shape_dims, "float16", self.op_params_stride2)
        self.assertEqual(test_obj.file_dims.H, 458)
        self.assertEqual(test_obj.file_dims.W, 115)
        test_obj.load_file()
        self.assertEqual(test_obj.dram_data.shape, (1, 3, 458, 115))

class TestFileMapper(unittest.TestCase):
    class op_params_stride2():
        stride = Dim2D(2,2)

    class op_params_stride1():
        stride = Dim2D(1,1)

    def test_map_file(self):
        shape_dims = ShapeDims("CRSM", [256,7,7,64]) 
        file_params = FileParams("testfile.npy", shape_dims, "float16", self.op_params_stride1)
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
        (last_writer, last_reader, waveops) = test_obj.read_file_data_region(10, list_of_accessors, file_params, 0, 0, 100)
        self.assertEqual(len(waveops), 1)
        self.assertEqual(waveops[0]['previous_waveops'], [])
        self.assertEqual(last_writer, -1)
        self.assertEqual(last_reader, -1)
        (last_writer, last_reader, waveops) = test_obj.write_file_data_region(20, list_of_accessors, file_params, 0, 0, 10, False)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(last_writer, -1)
        self.assertEqual(last_reader, 10)
        (last_writer, last_reader, waveops) = test_obj.write_file_data_region(30, list_of_accessors, file_params, 0, 0, 100, False)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(last_writer, 20)
        self.assertEqual(last_reader, 10)
        (last_writer, last_reader, waveops) = test_obj.write_file_data_region(40, list_of_accessors, file_params, 0, 100, 200, False)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(last_writer, -1)
        self.assertEqual(last_reader, 10)
        (last_writer, last_reader, waveops) = test_obj.read_file_data_region(40, list_of_accessors, file_params, 0, 100, 200)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(last_writer, 40)
        self.assertEqual(last_reader, 10)
        self.assertEqual(test_obj.get_chunk_id_from_file_addr(file_params, 0, 100), 0)
        #self.assertEqual(file_params.mapped_params.chunk_is_mapped[0][test_obj.get_chunk_id_from_file_addr(file_params, 0, 100)], True)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[test_obj.get_chunk_id_from_file_addr(file_params, 0, 100)], True)
        (last_writer, last_reader, waveops) = test_obj.read_file_data_region(40, list_of_accessors, file_params, 0, 100, 200)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(last_writer, -1)
        self.assertEqual(last_reader, 40)
        (last_writer, last_reader, waveops) = test_obj.read_file_data_region(50, list_of_accessors, file_params, 0, 50, 150)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(last_writer, -1)
        self.assertEqual(last_reader, 40)
        # test reading from file
        (last_writer, last_reader, waveops) = test_obj.read_file_data_region(10, list_of_accessors, file_params, 0, 128*7*7*64*file_params.item_sz, file_params.chunk_sz)
        self.assertEqual(len(waveops), 1)
        self.assertEqual(waveops[0]['previous_waveops'], [])
        self.assertEqual(waveops[0]["sb_address"], 40*1024 + 7*7*64*file_params.item_sz)
        self.assertEqual(waveops[0]["offset_in_file"], 128*7*7*64*file_params.item_sz)
        # test writing to file
        (last_writer, last_reader, waveops) = test_obj.write_file_data_region(40, list_of_accessors, file_params, 0, 100, 200, False)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(last_writer, -1)
        self.assertEqual(last_reader, 50)
        (last_writer, last_reader, waveops) = test_obj.write_file_data_region(40, list_of_accessors, file_params, 0, 100, 200, False)
        self.assertEqual(len(waveops), 0)
        self.assertEqual(last_writer, 40)
        self.assertEqual(last_reader, -1)
        waveops = test_obj.flush_file(0, list_of_accessors, file_params, 0)
        self.assertEqual(len(waveops), file_params.tot_num_chunks)
        self.assertEqual(waveops[0]['previous_waveops'], ["waveop_40", "waveop_50"])
        self.assertEqual(waveops[1]['previous_waveops'], [])

    def test_map_file2(self):
        shape_dims = ShapeDims("NCHW", [16,3,224,224]) 
        file_params = FileParams("testfile2.npy", shape_dims, "float16", self.op_params_stride2)
        file_params.load_file()
        test_obj = FileMapper(96*1024, "float16")
        test_obj.map_file(file_params, 50240, True, region_sz=55*55*file_params.item_sz)
        self.assertEqual(file_params.mapped_params.start_addr, 50240)
        self.assertEqual(file_params.mapped_params.region_sz, 3*file_params.chunk_sz)
        self.assertEqual(file_params.mapped_params.num_region_chunks, 3)
        self.assertEqual(file_params.mapped_params.num_file_chunks_per_batch_item, 56)
        self.assertEqual(file_params.mapped_params.end_addr, 50240 + 3*file_params.chunk_sz - file_params.item_sz)
        list_of_accessors = [{'waveop_name' : "waveop_%d"%i} for i in range(100)]
        (last_writer, last_reader, waveops) = test_obj.write_file_data_region(10, list_of_accessors, file_params, 15, 15*3*224*224*file_params.item_sz + 0, 100, False)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[15*file_params.batch_item_num_chunks + 0], True)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[15*file_params.batch_item_num_chunks + 1], False)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[0*file_params.batch_item_num_chunks + 0], False)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[0*file_params.batch_item_num_chunks + 1], False)
        self.assertEqual(last_writer, -1)
        self.assertEqual(last_reader, -1)
        (last_writer, last_reader, waveops) = test_obj.write_file_data_region(20, list_of_accessors, file_params, 0, 0, 10, False)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[15*file_params.batch_item_num_chunks + 0], False)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[15*file_params.batch_item_num_chunks + 1], False)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[0*file_params.batch_item_num_chunks + 0], True)
        self.assertEqual(file_params.mapped_params.chunk_is_mapped[0*file_params.batch_item_num_chunks + 1], False)
        self.assertEqual(last_writer, 10)
        self.assertEqual(last_reader, -1)

    def test_map_file_55x55(self):
        shape_dims = ShapeDims("NCHW", [1,256,55,55]) 
        file_params = FileParams("testfile_55x55.npy", shape_dims, "float16", self.op_params_stride2)
        file_params.load_file()
        test_obj = FileMapper(96*1024, "float16")
        #test_obj.map_file(file_params, 0, True, region_sz=55*55*file_params.item_sz)
        test_obj.map_file(file_params, 0, True, region_sz=12544)
        self.assertEqual(file_params.chunk_sz, 1760)
        self.assertEqual(file_params.mapped_params.start_addr, 0)
        #self.assertEqual(file_params.mapped_params.region_sz, 6*file_params.chunk_sz)
        self.assertEqual(file_params.mapped_params.num_region_chunks, 8)
        self.assertEqual(file_params.mapped_params.num_file_chunks_per_batch_item, 8)
        #self.assertEqual(file_params.mapped_params.end_addr, (55*55*2 - 1)*file_params.item_sz) 
        self.assertEqual(file_params.mapped_params.end_addr, (align_addr_8B(55*55*2) - 1)*file_params.item_sz) # padded
        list_of_accessors = [{'waveop_name' : "waveop_%d"%i} for i in range(100)]
        (last_writer, last_reader, waveops) = test_obj.read_file_data_region(10, list_of_accessors, file_params, 0, 0, 100)
        self.assertEqual(waveops[0]["sb_address"], 0)
        self.assertEqual(waveops[0]["offset_in_file"], 0)
        (last_writer, last_reader, waveops) = test_obj.read_file_data_region(10, list_of_accessors, file_params, 0, 128*55*55*file_params.item_sz, 100)
        self.assertEqual(test_obj.get_chunk_id_from_file_addr(file_params, 0, 0), 0)
        self.assertEqual(test_obj.get_file_addr_from_chunk_id(file_params, 0, 0), 0)
        self.assertEqual(test_obj.get_chunk_id_from_file_addr(file_params, 0, 128*55*55*file_params.item_sz), 4)
        self.assertEqual(test_obj.get_sb_addr_from_file_addr(file_params, 0, 128*55*55*file_params.item_sz), align_addr_8B(55*55*file_params.item_sz))
        self.assertEqual(test_obj.get_file_addr_from_chunk_id(file_params, 0, 4), 128*55*55*file_params.item_sz)
        self.assertEqual(waveops[0]["sb_address"], align_addr_8B(55*55*file_params.item_sz))
        self.assertEqual(waveops[0]["offset_in_file"], 128*55*55*file_params.item_sz)

    def test_zero_file(self):
        shape_dims = ShapeDims("NCHW", [16,3,224,224]) 
        file_params = FileParams("testfile2.npy", shape_dims, "float16", self.op_params_stride2)
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
        file_params = FileParams("testfile_"+str(shape_tuple)+".npy", shape_dims, "float16", self.op_params_stride1)
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
                    (last_writer, last_reader, waveops) = test_obj.read_file_data_region(10, list_of_accessors, file_params, i, current_offset, current_tile_size)
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
                    # check write_file_data_region per tile
                    current_tile_size = last_tile_size if (j == num_tiles-1) else tile_size
                    (last_writer, last_reader, waveops) = test_obj.write_file_data_region(10, list_of_accessors, file_params, i, current_offset, current_tile_size, False)
                    # Check get_chunk_id_from_file_addr and get_file_addr_from_chunk_id
                    chunk_id = test_obj.get_chunk_id_from_file_addr(file_params, i, current_offset)
                    chunk_offset = test_obj.get_chunk_offset_from_file_addr(file_params, i, current_offset)
                    file_addr = test_obj.get_file_addr_from_chunk_id(file_params, i, chunk_id)
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
