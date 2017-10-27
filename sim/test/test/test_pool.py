#!/usr/bin/python 
"""

"""
import unittest
import argparse
import sys
import numpy as np
import os
import random
import mock
import test_utils
from os.path import abspath
from pkgs import cmd_line
from utils import npdiff
from subprocess import call

pool_func = {
        "max_pool" : "0",
        "avg_pool" : "1"
        }
class TestPool(unittest.TestCase):
    def gold_pool(self, test_name, ifmap, kernel, stride, pf):
        ofmap = 'output/' + test_name + "_gold.npy"
        line = ["mn", pf, "-i", abspath(ifmap), "-ksize"] + kernel + \
                ["-stride"] + stride
        print "Running: " + " ".join(line)
        args = cmd_line.parser.parse_args(line)
        output = args.func(args)
        np.save(ofmap, output)
        return ofmap
    def tpu_pool(self, test_name, ifmap, kernel, stride, pf):
        binary = 'binary/' + test_name + '.bin'
        ofmap = 'output/' + test_name + "_tpu.npy"
        stdoutf = open('stdout/' + test_name + ".txt", 'w')
	binary_line = [test_utils.TEST_POOL]
        binary_line += ["-p", pool_func[pf], abspath(ifmap)] + map(str,kernel) + map(str, stride) + \
           [abspath(ofmap), binary]
        print "Compiling: " + " ".join(binary_line)
        call(binary_line)
        sim_line = [test_utils.SIM, binary]
        print "Running: " + " ".join(sim_line)
        call(sim_line, stdout = stdoutf)
        return ofmap
    def pool_test(self, test_name, ifmap, kernel, stride, pf):
        print ""
        o_gold = self.gold_pool(test_name, ifmap, kernel, stride, pf)
        o_tpu = self.tpu_pool(test_name, ifmap, kernel, stride, pf)
        npdiff.diff(o_gold, o_tpu)
    def rand_pool_test(self, test_name, itype, i_dims, kernel, stride, pf):
        ifmap =  test_utils.randf('i', itype, i_dims)
        self.pool_test(test_name, ifmap, kernel, stride, pf)

    def test_pool_small(self):
        itype = 'uint8'
        idims = [1,3,2,2]
        k = ["1", "1", "2", "2"]
        s = ["1", "1", "1", "1"]
        pf = 'avg_pool'
        tn = sys._getframe().f_code.co_name
        self.rand_pool_test(tn, itype, idims, k, s, pf)

    def test_pool_medium(self):
        itype = 'uint8'
        idims = [1,64,56,56]
        k = ["1", "1", "2", "2"]
        s = ["1", "1", "2", "2"]
        pf = 'max_pool'
        tn = sys._getframe().f_code.co_name
        self.rand_pool_test(tn, itype, idims, k, s, pf)

if __name__ == '__main__':
    unittest.main()
