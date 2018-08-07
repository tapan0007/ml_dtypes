import unittest
import os
import sys
kaena_path = os.environ['KAENA_PATH']+"/compiler/me"
sys.path.append(kaena_path)
import me_concat
import me_pool
import me_common_ds
from collections import deque
import numpy as np
import sim_pool
import math

# CHW
class SimTest:
    ifmap_files = []
    ifmap_data = []
    ifmap_specs = []
    MAX_WINDOW_SIZE = 5
    def ifmap_gen(self, dims):
        ifmap_id = 0
        for dim in dims:
            c = dim[0]
            h = dim[1]
            w = dim[2]
            self.ifmap = np.expand_dims(np.random.rand(c, h, w), axis = 0)
            file_name = "ifmap_"+str(ifmap_id)+"_"+str(dim[0])+".npy"
            self.ifmap_files.append(file_name)
            self.ifmap_data.append(self.ifmap)
            np.save(file_name, self.ifmap)
            ifmap_id += 1

    def gen_fmap_specs (self):
        for i in range(len(self.ifmap_data)):
            self.ifmap_specs.append(\
                me_common_ds.FMAPSpec(\
                    False, self.ifmap_data[i].shape\
                    , self.ifmap_files[i], self.ifmap_files[i] +"_0")\
                                   )
#        print (len(self.ifmap_specs))

    def gen_window(self):
        #r = np.random.randint(1, self.MAX_WINDOW_SIZE)
        #s = r
        #th = self.MAX_WINDOW_SIZE
        #while th > r:
        #    th = np.random.randint(1, self.MAX_WINDOW_SIZE - 1)
        #    tv = th
        r = s = 3
        th = tv = 1
        pool_window = me_common_ds.FilterSpec(r, s, th, tv)
        return pool_window

    def get_padding(self, window):
        h = self.ifmap.shape[2]
        w = self.ifmap.shape[3]
        th = window.Th
        tv = window.Tv
        r = window.R
        s = window.S
        padding_horizontal = w * (th - 1) + s - th
        padding_vertical = h * (tv - 1) + r - tv
        pW = int(math.floor(padding_vertical / 2))
        pE = int(math.ceil(padding_vertical / 2))
        pN = int(math.floor(padding_horizontal / 2))
        pS = int(math.ceil(padding_horizontal / 2))
        print ("r = %d s = %d th = %d tv = %d pN = %d pS = %d pW = %d pE = %d"\
               %(r, s, th, tv, pN, pS, pW, pE))
        return me_common_ds.PaddingSpec(pN, pS, pW, pE)

    # dims.shape = (C, H, W)
    def run_concat (self, dims):
        self.ifmap_gen(dims)
        self.gen_fmap_specs()
        ofmap_channel = 0
        for i in dims:
            ofmap_channel += i[0]
        concat = me_concat.Concat(self.ifmap_specs, ofmap_channel, np.float32)
        concat.PerformConcatDecomposition(True)
        concat.FilterWeightFileGeneration()
        concat_sim = sim_concat.ConcatSim(concat)
        return concat_sim.run()
    
    def run_pool (self, dims):
        self.ifmap_gen(dims)
        self.gen_fmap_specs()
        ofmap_channel = 0
        ofmap_file_name = "OFMAP_Padded_Pool.npy"
        ofmap_spec = me_common_ds.FMAPSpec(False, self.ifmap.shape\
            , ofmap_file_name, ofmap_file_name + "_0")
        for i in dims:
            ofmap_channel += i[0]
        window = self.gen_window()
        pool = me_pool.Pool(\
            self.ifmap_specs[0], window, self.get_padding(window)\
            , "AvgPool", False, False, ofmap_spec, np.float16\
                           )
        pool.ComputePool()
        pool_sim = sim_pool.PoolSim(pool)
        return pool_sim.run()

#class TestConcatSim (unittest.TestCase):
#    sim_test = SimTest()
#
#    def test_random_dim(self):
#        Hmax = 139
#        IFMAPCntMax = 12
#        IFMAPCOLmax = 128
#        H = int(np.random.randint(1, Hmax))
#        dims = []
#        num_ifmaps = int(np.random.randint(1, IFMAPCntMax))
#        for i in range(num_ifmaps):
#            c_dim = int(np.random.randint(1, IFMAPCOLmax))
#            dims.append((c_dim, H, H))
#        print ("IFMAP dimensions in CHW format = ",dims)
#        self.assertTrue(self.sim_test.run_concat(dims))

class TestPoolSim (unittest.TestCase):
    sim_test = SimTest()

    def test_random_dim(self):
        Hmax = 139
        IFMAPCntMax = 12
        IFMAPCOLmax = 128
        H = int(np.random.randint(1, Hmax))
        dims = []
        num_ifmaps = 1
        for i in range(num_ifmaps):
            c_dim = int(np.random.randint(1, IFMAPCOLmax))
#            dims.append((c_dim, H, H))
        dims.append((196, 35, 35))
        print ("IFMAP dimensions in CHW format = ",dims)
        self.assertTrue(self.sim_test.run_pool(dims))

if __name__ == '__main__':
    unittest.main()

#s = SimTest()
#s.run()
