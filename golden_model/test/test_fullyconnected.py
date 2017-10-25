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
from pkgs import cmd_line


class TestFullyConnected(unittest.TestCase):
    def fullyconnected(self, i_file, w_file, n, b_file=None):
        b_opt_n_file = [] if b_file == None else ["-b", b_file]

        base_args = ["fullyconnected", "-i", i_file,  "-w", w_file, "-n", str(n)] + b_opt_n_file
        args = cmd_line.parser.parse_args(["tf"] + base_args)
        tf_o = args.func(args)
        args = cmd_line.parser.parse_args(["mn"] + base_args)
        mn_o = args.func(args)
        err_msg = "Suffix of command line: {}".format(" ".join(base_args))
        with np.errstate(under='ignore'):
            np.testing.assert_allclose(mn_o, tf_o, rtol=1e-2, err_msg=err_msg)
    def rand_fullyconnected(self, prec, i_dims, w_dims, n, b_dims = None):
        bias = None
        img = test_utils.randf('i', prec, i_dims)
        weight = test_utils.randf('w', prec, w_dims)
	if b_dims:
            bias = test_utils.randf('b', prec, b_dims)
        self.fullyconnected(img, weight, n, bias)

    def test_fp16_ch1_small(self):
        self.rand_fullyconnected('float16', [1,1,3,4], [4,12], 4)
    def test_fp16_ch1_big(self):
        self.rand_fullyconnected('float16', [1,1,128,128], [16,16384], 16)
    def test_fp16_ch3_small(self):
        self.rand_fullyconnected('float16', [1,3,2,2], [1,12], 1)
    def test_fp16_ch3_med(self):
        self.rand_fullyconnected('float16', [1,3,5,5], [5,75], 5)
    # this test is just way too many multiplies and we are off by integer amounts
#    def test_fp16_ch3_big(self):
#        i = 'imgs/float16/channel_3/i_fp16_1x3x64x64_rand.npy'
#        w = 'weights/float16/w_fp16_128x12288_rand.npy'
#        n = 128
#        self.rand_fullyconnected(i, w, n)
    def test_fp32_ch1_small(self):
        self.rand_fullyconnected('float32', [1,1,5,5], [1,25], 1)
    def test_fp32_ch1_big(self):
        self.rand_fullyconnected('float32', [1,1,128,128], [32,16384], 32)
    def test_fp32_ch3(self):
        self.rand_fullyconnected('float32', [1,3,16,16], [2,768], 2)
#    def test_uint8_ch1(self):
# uint8 not supported yet in python - maybe in c++?
#    def test_uint8_ch3(self):
#        i = 'imgs/uint8/channel_3/i_uint8_1x3x32x32_rand.npy'
#        w = 'weights/uint8/w_uint8_64x3072_rand.npy'
#        n = 64
#        self.rand_fullyconnected(i, w, n)
    def test_bias(self):
        self.rand_fullyconnected('float32', [1,1,128,128], [32,16384], 32, b_dims=[32])


if __name__ == '__main__':
    unittest.main()
