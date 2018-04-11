import unittest
from enum import Enum

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

if __name__ == '__main__':
    unittest.main()
