import unittest

import os
import sys
kaena_path = os.environ['KAENA_PATH']+"/compiler/me"
sys.path.append(kaena_path)
import me_common_ds
import me_concat
import me_utils
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

    def gen_test(self, channels, forward_move, init_from_file_params = True):
        self.init_from_file_params = init_from_file_params
        ifmaps = list()
        ofmap_channel = sum(channels[0:len(channels)])
        ifmap_id = 0
        ifmap_file_params = []
        class op_params_stride2():
            stride = me_utils.Dim2D(1,1)
        for i in channels:
            file_name = "ifmap_"+str(ifmap_id)+"_"+str(i)+".npy"
            waveop_name = "ifmap_"+str(ifmap_id)+"_"+str(i)+".npy_0"
            ifmap =\
              me_common_ds.FMAPSpec(False,[1, i, 35, 35],file_name,waveop_name)
            ifmap_file_param =\
              me_utils.FileParams(\
                                  file_name\
                                  , me_utils.ShapeDims("NCHW", (1, i, 35, 35))\
                                  , "float16"\
                                  , op_params_stride2\
                                 )
            ifmap_file_params.append(ifmap_file_param)
            ifmaps.append(ifmap)
            ifmap_id += 1
        if (init_from_file_params == False):
            self.concat = me_concat.Concat(ifmaps, ofmap_channel, np.float16)
            self.waveops = self.concat.waveops
            test = self.concat.PerformConcatDecomposition(forward_move)
        else:
            self.concat =\
                    me_concat.Concat.init_from_file_params(ifmap_file_params)
            test = self.concat.PerformConcatDecomposition(forward_move)
            #self.concat.PrintSubTileInfos()
            self.waveops = self.concat.waveops
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

#    def test_filters_32_118_19_tensors(self):
#        print("...test_filters_32_118_19_tensors")
#        golden = self.gen_filters_32_118_19_tensors()
#        test = self.gen_test([32, 118, 19], False)
#        self.assertTrue(self.compare(golden, test))
#        mm = self.ExtractMM()
#        self.assertTrue(mm[0].start_tensor_calc)
#        self.assertFalse(mm[1].start_tensor_calc)
#        self.assertTrue(mm[2].start_tensor_calc)
#        self.assertTrue(mm[3].start_tensor_calc)
#        self.assertFalse(mm[4].start_tensor_calc)

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
        if (self.init_from_file_params == True):
            keys = list(self.concat.subtile_infos.keys())
            self.assertTrue(keys[0] == (0, 63))
            self.assertTrue(len(self.concat.subtile_infos[keys[0]][0]) == 2)
            self.assertTrue(keys[1] == (64, 127))
            self.assertTrue(len(self.concat.subtile_infos[keys[1]][0]) == 1)
            self.assertTrue(keys[2] == (128, 168))
            self.assertTrue(len(self.concat.subtile_infos[keys[2]][0]) == 2)
        
    def gen_filters_32_118_tensors(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((96, 0), 22))
        filters.append(me_concat.MoveFilterSpec((32, 0), 64))
        filters.append(me_concat.MoveFilterSpec((0, 32), 32))
        filters.append(me_concat.MoveFilterSpec((0, 0), 32))
        return filters

#    def test_filters_32_118_tensors(self):
#        print("...test_filters_32_118_tensors")
#        golden = self.gen_filters_32_118_tensors()
#        test = self.gen_test([32, 118], False)
#        self.assertTrue(self.compare(golden, test))
#        mm = self.ExtractMM()
#        self.assertTrue(mm[0].start_tensor_calc)
#        self.assertTrue(mm[1].start_tensor_calc)
#        self.assertTrue(mm[2].start_tensor_calc)
#        self.assertFalse(mm[3].start_tensor_calc)

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
        if (self.init_from_file_params == True):
            keys = list(self.concat.subtile_infos.keys())
            self.assertTrue(keys[0] == (0, 63))
            self.assertTrue(len(self.concat.subtile_infos[keys[0]][0]) == 2)
            self.assertTrue(keys[1] == (64, 127))
            self.assertTrue(len(self.concat.subtile_infos[keys[1]][0]) == 1)
            self.assertTrue(keys[2] == (128, 149))
            self.assertTrue(len(self.concat.subtile_infos[keys[2]][0]) == 1)

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
        if (self.init_from_file_params == True):
            keys = list(self.concat.subtile_infos.keys())
            self.assertTrue(keys[0] == (0, 63))
            self.assertTrue(len(self.concat.subtile_infos[keys[0]][0]) == 2)

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
        if (self.init_from_file_params == True):
            keys = list(self.concat.subtile_infos.keys())
            self.assertTrue(keys[0] == (0, 1))
            self.assertTrue(len(self.concat.subtile_infos[keys[0]][0]) == 2)

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
        if (self.init_from_file_params == True):
            keys = list(self.concat.subtile_infos.keys())
            self.assertTrue(keys[0] == (0, 63))
            subtile_info = self.concat.GetSubTile(keys[0])
            self.assertTrue(len(subtile_info[0]) == 1)
            self.assertTrue(keys[1] == (64, 127))
            subtile_info = self.concat.GetSubTile(keys[1])
            self.assertTrue(len(subtile_info[0]) == 2)
            self.assertTrue(keys[2] == (128, 191))
            subtile_info = self.concat.GetSubTile(keys[2])
            self.assertTrue(len(subtile_info[0]) == 1)
            self.assertTrue(keys[3] == (192, 255))
            subtile_info = self.concat.GetSubTile(keys[3])
            self.assertTrue(len(subtile_info[0]) == 1)

    def gen_filters_first_concat_inceptv3_4_forward(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((0, 0), 64))
        filters.append(me_concat.MoveFilterSpec((64, 0), 32))
        filters.append(me_concat.MoveFilterSpec((0, 32), 32))
        filters.append(me_concat.MoveFilterSpec((32, 0), 64))
        filters.append(me_concat.MoveFilterSpec((96, 0), 32))
        filters.append(me_concat.MoveFilterSpec((0, 32), 32))
        filters.append(me_concat.MoveFilterSpec((32, 0), 64))
        filters.append(me_concat.MoveFilterSpec((96, 0), 32))
        filters.append(me_concat.MoveFilterSpec((0, 32), 32))
        filters.append(me_concat.MoveFilterSpec((0, 0), 64))
        filters.append(me_concat.MoveFilterSpec((64, 0), 64))
        filters.append(me_concat.MoveFilterSpec((0, 0), 64))
        filters.append(me_concat.MoveFilterSpec((64, 0), 64))
        filters.append(me_concat.MoveFilterSpec((0, 0), 64))
        filters.append(me_concat.MoveFilterSpec((64, 0), 64))
        return filters

    # 4-th Concat in InceptionV3
    def test_filters_first_concat_inceptv3_4_forward (self):
        print("...test_filters_first_concat_inceptv3_forward")
        golden = self.gen_filters_first_concat_inceptv3_4_forward()
        test = self.gen_test([96, 288, 384], True)
#        for i in test:
#            i.print()
        self.assertTrue(self.compare(golden, test))
        mm = self.ExtractMM()
        self.assertTrue(mm[0].start_tensor_calc)
        self.assertTrue(mm[1].start_tensor_calc)
        self.assertFalse(mm[2].start_tensor_calc)
        self.assertTrue(mm[3].start_tensor_calc)
        self.assertTrue(mm[4].start_tensor_calc)
        self.assertFalse(mm[5].start_tensor_calc)
        self.assertTrue(mm[6].start_tensor_calc)
        self.assertTrue(mm[7].start_tensor_calc)
        self.assertFalse(mm[8].start_tensor_calc)
        self.assertTrue(mm[9].start_tensor_calc)
        self.assertTrue(mm[10].start_tensor_calc)
        self.assertTrue(mm[11].start_tensor_calc)
        self.assertTrue(mm[12].start_tensor_calc)
        self.assertTrue(mm[13].start_tensor_calc)
        self.assertTrue(mm[14].start_tensor_calc)

        if (self.init_from_file_params == True):
            keys = list(self.concat.subtile_infos.keys())
            self.assertTrue(keys[0] == (0, 63))
            subtile_info = self.concat.GetSubTile(keys[0])
            self.assertTrue(len(subtile_info[0]) == 1)
            self.assertTrue(keys[1] == (64, 127))
            subtile_info = self.concat.GetSubTile(keys[1])
            self.assertTrue(len(subtile_info[0]) == 2)
            self.assertTrue(keys[2] == (128, 191))
            subtile_info = self.concat.GetSubTile(keys[2])
            self.assertTrue(len(subtile_info[0]) == 1)
            self.assertTrue(keys[3] == (192, 255))
            subtile_info = self.concat.GetSubTile(keys[3])
            self.assertTrue(len(subtile_info[0]) == 2)
            self.assertTrue(keys[4] == (256, 319))
            subtile_info = self.concat.GetSubTile(keys[4])
            self.assertTrue(len(subtile_info[0]) == 1)
            self.assertTrue(keys[5] == (320, 383))
            subtile_info = self.concat.GetSubTile(keys[5])
            self.assertTrue(len(subtile_info[0]) == 2)
            self.assertTrue(keys[6] == (384, 447))
            subtile_info = self.concat.GetSubTile(keys[6])
            self.assertTrue(len(subtile_info[0]) == 1)
            self.assertTrue(keys[7] == (448, 511))
            subtile_info = self.concat.GetSubTile(keys[7])
            self.assertTrue(len(subtile_info[0]) == 1)
            self.assertTrue(keys[8] == (512, 575))
            subtile_info = self.concat.GetSubTile(keys[8])
            self.assertTrue(len(subtile_info[0]) == 1)
            self.assertTrue(keys[9] == (576, 639))
            subtile_info = self.concat.GetSubTile(keys[9])
            self.assertTrue(len(subtile_info[0]) == 1)
            self.assertTrue(keys[10] == (640, 703))
            subtile_info = self.concat.GetSubTile(keys[10])
            self.assertTrue(len(subtile_info[0]) == 1)
            self.assertTrue(keys[11] == (704, 767))
            subtile_info = self.concat.GetSubTile(keys[11])
            self.assertTrue(len(subtile_info[0]) == 1)
    def gen_filters_63_180_forward(self):
        filters = deque()
        filters.append(me_concat.MoveFilterSpec((0, 0), 63))
        filters.append(me_concat.MoveFilterSpec((0, 63), 1))
        filters.append(me_concat.MoveFilterSpec((1, 0), 64))
        filters.append(me_concat.MoveFilterSpec((65, 0), 63))
        filters.append(me_concat.MoveFilterSpec((0, 63), 1))
        filters.append(me_concat.MoveFilterSpec((1, 0), 51))
        return filters

    def test_filters_63_180_forward (self):
        print("...test_filters_63_180_forward")
        golden = self.gen_filters_63_180_forward()
        test = self.gen_test([63, 180], True)
        for i in test:
            i.print()
        self.assertTrue(self.compare(golden, test))
        mm = self.ExtractMM()
        self.assertTrue(mm[0].start_tensor_calc)
        self.assertFalse(mm[1].start_tensor_calc)
        self.assertTrue(mm[2].start_tensor_calc)
        self.assertTrue(mm[3].start_tensor_calc)
        self.assertFalse(mm[4].start_tensor_calc)
        self.assertTrue(mm[5].start_tensor_calc)
        if (self.init_from_file_params == True):
            keys = list(self.concat.subtile_infos.keys())
            self.assertTrue(keys[0] == (0, 63))
            subtile_info = self.concat.GetSubTile(keys[0])
            self.assertTrue(len(subtile_info[0]) == 2)
            self.assertTrue(keys[1] == (64, 127))
            subtile_info = self.concat.GetSubTile(keys[1])
            self.assertTrue(len(subtile_info[0]) == 1)
            self.assertTrue(keys[2] == (128, 191))
            subtile_info = self.concat.GetSubTile(keys[2])
            self.assertTrue(len(subtile_info[0]) == 2)
            self.assertTrue(keys[3] == (192, 242))
            subtile_info = self.concat.GetSubTile(keys[3])
            self.assertTrue(len(subtile_info[0]) == 1)
        
        
if __name__ == '__main__':
    unittest.main()