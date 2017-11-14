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
import math

activation_func = {
        "relu" : "2",
        "leakyrelu" : "3",
        "sigmoid" : "4",
        "tanh" : "5"
        }

rtol = {
        "relu" : .1,
        "leakyrelu" : .1,
        "sigmoid" : .1,
        "tanh" : .2
        }
class TestActivation(unittest.TestCase):
    def gold_activation(self, test_name, ifmap, af):
        ofmap = 'output/' + test_name + "_gold.npy"
        line = ["mn", af, "-i", abspath(ifmap)]
        print "Running: " + " ".join(line)
        args = cmd_line.parser.parse_args(line)
        output = args.func(args)
        np.save(ofmap, output)
        return ofmap
    def tpu_activation(self, test_name, ifmap, af):
        binary = 'binary/' + test_name + '.bin'
        ofmap = 'output/' + test_name + "_tpu.npy"
        stdoutf = open('stdout/' + test_name + ".txt", 'w')
	binary_line = [test_utils.TEST_ACTIVATION]
        binary_line += ["-a", activation_func[af], abspath(ifmap),
                abspath(ofmap), binary]
        print "Compiling: " + " ".join(binary_line)
        call(binary_line)
        sim_line = [test_utils.SIM, binary]
        print "Running: " + " ".join(sim_line)
        call(sim_line, stdout = stdoutf)
        return ofmap
    def activation_test(self, test_name, ifmap, af):
        print ""
        o_gold = self.gold_activation(test_name, ifmap, af)
        o_tpu = self.tpu_activation(test_name, ifmap, af)
        npdiff.diff(o_gold, o_tpu, rtol = rtol[af])
    def rand_activation_test(self, test_name, itype, i_dims, af, 
            vmin=None, vmax=None):
        ifmap =  test_utils.randf('i', itype, i_dims, vmin, vmax)
        self.activation_test(test_name, ifmap, af)

    def test_relu_int_small(self):
        itype = 'int8'
        idims = [1,3,2,2]
        af = 'relu'
        tn = sys._getframe().f_code.co_name
        self.rand_activation_test(tn, itype, idims, af)
    def test_relu_fp16_small(self):
        itype = 'float16'
        idims = [1,3,4,2]
        af = 'relu'
        tn = sys._getframe().f_code.co_name
        self.rand_activation_test(tn, itype, idims, af)
    def test_leaky_relu_int_small(self):
        itype = 'int8'
        idims = [1,3,2,2]
        af = 'leakyrelu'
        tn = sys._getframe().f_code.co_name
        self.rand_activation_test(tn, itype, idims, af)
    def test_leaky_relu_int_med(self):
        itype = 'int8'
        idims = [1,3,32,8]
        af = 'leakyrelu'
        tn = sys._getframe().f_code.co_name
        self.rand_activation_test(tn, itype, idims, af)
    def test_tanh(self):
        itype = 'int8'
        idims = [1,3,16,16]
        af = 'tanh'
        tn = sys._getframe().f_code.co_name
        self.rand_activation_test(tn, itype, idims, af)
    def test_fp16_tanh(self):
        itype = 'float16'
        idims = [1,3,12,16]
        af = 'tanh'
        tn = sys._getframe().f_code.co_name
        self.rand_activation_test(tn, itype, idims, af)
    def test_sigmoid(self):
        itype = 'int8'
        idims = [1,3,16,16]
        af = 'sigmoid'
        vmin = -math.log(np.iinfo(itype).max)
        vmax = -1
        tn = sys._getframe().f_code.co_name
        self.rand_activation_test(tn, itype, idims, af, vmin, vmax)
    def test_fp16_sigmoid(self):
        itype = 'float16'
        idims = [1,4,4,4]
        af = 'sigmoid'
        vmin = -math.log(np.finfo(itype).max)
        vmax = -1
        tn = sys._getframe().f_code.co_name
        self.rand_activation_test(tn, itype, idims, af, vmin, vmax)
if __name__ == '__main__':
    unittest.main()
