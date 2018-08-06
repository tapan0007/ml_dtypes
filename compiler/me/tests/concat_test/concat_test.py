import unittest

import os
import sys
kaena_path = os.environ['KAENA_PATH']+"/compiler/me"
sys.path.append(kaena_path)
import me_concat
from collections import deque
import numpy as np

#ifmaps = list()
#channels = [32, 118, 19]
#ofmap_channel = sum(channels[0:len(channels)])
#for i in channels:
#    ifmap = me_concat.FMAPSpec(False, i)
#    ifmaps.append(ifmap)
#concat = me_concat.Concat(ifmaps, ofmap_channel)
#move_filters = concat.ComputeMoveFilters()
#
#for i in move_filters:
#    i.print()

class TestConcat (unittest.TestCase):
    waveops = []
    def compare(self, golden, test):
        for i in range(len(golden)):
            if (golden[i] != test[i]):
                return False
        return True

    def ExtractMM(self):
        mm_id = 0
        mm = []
        for i in self.waveops:
            if (i.__class__.__name__ == "MMWaveOpInfo"):
                mm.append(i)
                mm_id += 1
        return mm

    def gen_test(self, channels, forward_move):
        ifmaps = list()
        ofmap_channel = sum(channels[0:len(channels)])
        ifmap_id = 0
        for i in channels:
            file_name = "ifmap_"+str(ifmap_id)+"_"+str(i)+".npy"
            waveop_name = "ifmap_"+str(ifmap_id)+"_"+str(i)+".npy_0"
            ifmap =\
              me_concat.FMAPSpec(False, [1, i, 35, 35], file_name, waveop_name)
            ifmaps.append(ifmap)
            ifmap_id += 1
        concat = me_concat.Concat(ifmaps, ofmap_channel, np.float16)
        self.waveops = concat.waveops
        test = concat.PerformConcatDecomposition(forward_move)
#        concat.FilterWeightFileGeneration()
#        for i in test:
#            i.print()
#        concat.print_graph()
#        for i in concat.waveops:
#            print("cur_waveop = %s"%i.name)
#            i.print_prev_ops()
        return test

    def gen_filters_32_118_19_tensors(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((0, 22), 19))
        filters.append(me_concat.MoveFilterSpec((96, 0), 22))
        filters.append(me_concat.MoveFilterSpec((32, 0), 64))
        filters.append(me_concat.MoveFilterSpec((0, 32), 32))
        filters.append(me_concat.MoveFilterSpec((0, 0), 32))
        return filters

    def test_filters_32_118_19_tensors(self):
        print("...test_filters_32_118_19_tensors")
        golden = self.gen_filters_32_118_19_tensors()
        test = self.gen_test([32, 118, 19], False)
        self.assertTrue(self.compare(golden, test))
        mm = self.ExtractMM()
        self.assertTrue(mm[0].start_tensor_calc)
        self.assertFalse(mm[1].start_tensor_calc)
        self.assertTrue(mm[2].start_tensor_calc)
        self.assertTrue(mm[3].start_tensor_calc)
        self.assertFalse(mm[4].start_tensor_calc)

    def gen_filters_32_118_19_tensors_forward(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((0, 0), 32))
        filters.append(me_concat.MoveFilterSpec((0, 32), 32))
        filters.append(me_concat.MoveFilterSpec((32, 0), 64))
        filters.append(me_concat.MoveFilterSpec((96, 0), 22))
        filters.append(me_concat.MoveFilterSpec((0, 22), 19))
        return filters

    def test_filters_32_118_19_tensors_forward(self):
        print("...test_filters_32_118_19_tensors_forward")
        golden = self.gen_filters_32_118_19_tensors_forward()
        test = self.gen_test([32, 118, 19], True)
        self.assertTrue(self.compare(golden, test))
        mm = self.ExtractMM()
        self.assertTrue(mm[0].start_tensor_calc)
        self.assertFalse(mm[1].start_tensor_calc)
        self.assertTrue(mm[2].start_tensor_calc)
        self.assertTrue(mm[3].start_tensor_calc)
        self.assertFalse(mm[4].start_tensor_calc)
        
    def gen_filters_32_118_tensors(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((96, 0), 22))
        filters.append(me_concat.MoveFilterSpec((32, 0), 64))
        filters.append(me_concat.MoveFilterSpec((0, 32), 32))
        filters.append(me_concat.MoveFilterSpec((0, 0), 32))
        return filters

    def test_filters_32_118_tensors(self):
        print("...test_filters_32_118_tensors")
        golden = self.gen_filters_32_118_tensors()
        test = self.gen_test([32, 118], False)
        self.assertTrue(self.compare(golden, test))
        mm = self.ExtractMM()
        self.assertTrue(mm[0].start_tensor_calc)
        self.assertTrue(mm[1].start_tensor_calc)
        self.assertTrue(mm[2].start_tensor_calc)
        self.assertFalse(mm[3].start_tensor_calc)

    def gen_filters_32_118_tensors_forward(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((0, 0), 32))
        filters.append(me_concat.MoveFilterSpec((0, 32), 32))
        filters.append(me_concat.MoveFilterSpec((32, 0), 64))
        filters.append(me_concat.MoveFilterSpec((96, 0), 22))
        return filters

    def test_filters_32_118_tensors_forward(self):
        print("...test_filters_32_118_tensors_forward")
        golden = self.gen_filters_32_118_tensors_forward()
        test = self.gen_test([32, 118], True)
        self.assertTrue(self.compare(golden, test))
        mm = self.ExtractMM()
        self.assertTrue(mm[0].start_tensor_calc)
        self.assertFalse(mm[1].start_tensor_calc)
        self.assertTrue(mm[2].start_tensor_calc)
        self.assertTrue(mm[3].start_tensor_calc)

    def gen_filters_32_32_tensors(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((0, 32), 32))
        filters.append(me_concat.MoveFilterSpec((0, 0), 32))
        return filters

    def test_filters_32_32_tensors (self):
        print("...test_filters_32_32_tensors")
        golden = self.gen_filters_32_32_tensors()
        test = self.gen_test([32, 32], False)
        self.assertTrue(self.compare(golden, test))
        mm = self.ExtractMM()
        self.assertTrue(mm[0].start_tensor_calc)
        self.assertFalse(mm[1].start_tensor_calc)

    def gen_filters_32_32_tensors_forward(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((0, 0), 32))
        filters.append(me_concat.MoveFilterSpec((0, 32), 32))
        return filters

    def test_filters_32_32_tensors_forward (self):
        print("...test_filters_32_32_tensors_forward")
        golden = self.gen_filters_32_32_tensors_forward()
        test = self.gen_test([32, 32], True)
        self.assertTrue(self.compare(golden, test))
        mm = self.ExtractMM()
        self.assertTrue(mm[0].start_tensor_calc)
        self.assertFalse(mm[1].start_tensor_calc)

    def gen_filters_1_1_tensors(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((0, 1), 1))
        filters.append(me_concat.MoveFilterSpec((0, 0), 1))
        return filters

    def test_filters_1_1_tensors (self):
        print("...test_filters_1_1_tensors")
        golden = self.gen_filters_1_1_tensors()
        test = self.gen_test([1, 1], False)
        self.assertTrue(self.compare(golden, test))
        mm = self.ExtractMM()
        self.assertTrue(mm[0].start_tensor_calc)
        self.assertFalse(mm[1].start_tensor_calc)

    def gen_filters_1_1_tensors_forward(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((0, 0), 1))
        filters.append(me_concat.MoveFilterSpec((0, 1), 1))
        return filters

    def test_filters_1_1_tensors_forward (self):
        print("...test_filters_1_1_tensors_forward")
        golden = self.gen_filters_1_1_tensors_forward()
        test = self.gen_test([1, 1], True)
        self.assertTrue(self.compare(golden, test))
        mm = self.ExtractMM()
        self.assertTrue(mm[0].start_tensor_calc)
        self.assertFalse(mm[1].start_tensor_calc)

    def gen_filters_first_concat_inceptv3(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((0, 0), 64))
        filters.append(me_concat.MoveFilterSpec((0, 0), 64))
        filters.append(me_concat.MoveFilterSpec((0, 32), 32))
        filters.append(me_concat.MoveFilterSpec((64, 0), 32))
        filters.append(me_concat.MoveFilterSpec((0, 0), 64))
        return filters

    def test_filters_first_concat_inceptv3 (self):
        print("...test_filters_first_concat_inceptv3")
        golden = self.gen_filters_first_concat_inceptv3()
        test = self.gen_test([96, 32, 64, 64], False)
        self.assertTrue(self.compare(golden, test))
        mm = self.ExtractMM()
        self.assertTrue(mm[0].start_tensor_calc)
        self.assertTrue(mm[1].start_tensor_calc)
        self.assertTrue(mm[2].start_tensor_calc)
        self.assertFalse(mm[3].start_tensor_calc)
        self.assertTrue(mm[4].start_tensor_calc)
        
    def gen_filters_first_concat_inceptv3_forward(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((0, 0), 64))
        filters.append(me_concat.MoveFilterSpec((64, 0), 32))
        filters.append(me_concat.MoveFilterSpec((0, 32), 32))
        filters.append(me_concat.MoveFilterSpec((0, 0), 64))
        filters.append(me_concat.MoveFilterSpec((0, 0), 64))
        return filters

    def test_filters_first_concat_inceptv3_forward (self):
        print("...test_filters_first_concat_inceptv3_forward")
        golden = self.gen_filters_first_concat_inceptv3_forward()
        test = self.gen_test([96, 32, 64, 64], True)
#        for i in test:
#            i.print()
        self.assertTrue(self.compare(golden, test))
        mm = self.ExtractMM()
        self.assertTrue(mm[0].start_tensor_calc)
        self.assertTrue(mm[1].start_tensor_calc)
        self.assertFalse(mm[2].start_tensor_calc)
        self.assertTrue(mm[3].start_tensor_calc)
        self.assertTrue(mm[4].start_tensor_calc)
        
if __name__ == '__main__':
    unittest.main()
