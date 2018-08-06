import unittest
import os
import sys
kaena_path = os.environ['KAENA_PATH']+"/compiler/me"
sys.path.append(kaena_path)
import me_concat
from collections import deque
import numpy as np
import sim_concat

# CHW
class SimTest:
    ifmap_files = []
    ifmap_data = []
    ifmap_specs = []
    def ifmap_gen(self, dims):
        ifmap_id = 0
        for dim in dims:
            c = dim[0]
            h = dim[1]
            w = dim[2]
            ifmap = np.expand_dims(np.random.rand(c, h, w), axis = 0)
            file_name = "ifmap_"+str(ifmap_id)+"_"+str(dim[0])+".npy"
            self.ifmap_files.append(file_name)
            self.ifmap_data.append(ifmap)
            np.save(file_name, ifmap)
            ifmap_id += 1
#        print (len(self.ifmap_data))

    def gen_fmap_specs (self):
        for i in range(len(self.ifmap_data)):
            self.ifmap_specs.append(\
                me_concat.FMAPSpec(\
                    False, self.ifmap_data[i].shape\
                    , self.ifmap_files[i], self.ifmap_files[i] +"_0")\
                                   )
#        print (len(self.ifmap_specs))

    # dims.shape = (C, H, W)
    def run (self, dims):
#        H = 35
#        W = 35
#        dims = [(34, H, W), (15, H, W), (128, H, W), (49, H, W)]
        self.ifmap_gen(dims)
        self.gen_fmap_specs()
        ofmap_channel = 0
        for i in dims:
            ofmap_channel += i[0]
#        print ("ofmap_channel sum = %d"%ofmap_channel)
#        print ("size of self.ifmap_specs = %d"%len(self.ifmap_specs))
        concat = me_concat.Concat(self.ifmap_specs, ofmap_channel, np.float32)
#        print ("size of concat.ifmaps = %d"%len(concat.ifmaps))
        concat.PerformConcatDecomposition(True)
        concat.FilterWeightFileGeneration()
        concat_sim = sim_concat.ConcatSim(concat)
        return concat_sim.run()

class TestConcatSim (unittest.TestCase):
    sim_test = SimTest()

    def test_random_dim(self):
        Hmax = 139
        IFMAPCntMax = 12
        IFMAPCOLmax = 128
        H = int(np.random.randint(1, Hmax))
        dims = []
        num_ifmaps = int(np.random.randint(1, IFMAPCntMax))
        for i in range(num_ifmaps):
            c_dim = int(np.random.randint(1, IFMAPCOLmax))
            dims.append((c_dim, H, H))
        print ("IFMAP dimensions in CHW format = ",dims)
        self.assertTrue(self.sim_test.run(dims))

if __name__ == '__main__':
    unittest.main()

#s = SimTest()
#s.run()
