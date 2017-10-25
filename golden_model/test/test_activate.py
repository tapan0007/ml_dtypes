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


class TestActivate(unittest.TestCase):
    def activate(self, prefix, i_file):
        base_args = [prefix, "-i", i_file]
        args = cmd_line.parser.parse_args(["tf"] + base_args)
        tf_o = args.func(args)
        args = cmd_line.parser.parse_args(["mn"] + base_args)
        mn_o = args.func(args)
        err_msg = "Suffix of command line: {}".format(" ".join(base_args))
        rtol_dict={'relu': 1e-07,
                   'leakyrelu': 1e-07,
                   'tanh': 1e-04,
                   'sigmoid': 1e-03}
        with np.errstate(under='ignore'):
            np.testing.assert_allclose(mn_o, tf_o, rtol=rtol_dict[prefix], err_msg=err_msg)
    def rand_activate(self, prefix, prec, i_dims):
        ifile = test_utils.randf('i', prec, i_dims)
        self.activate(prefix, ifile)

    def test_relu_fp16_ch3(self):
        self.rand_activate('relu', 'float16', [1,3,32,32])
    def test_relu_fp32_ch3(self):
        self.rand_activate('relu', 'float32', [1,3,64,64])
    # uint8 not supported yet in tensorflow
#    def test_relu_uint8_ch3(self):
#        self.rand_activate('relu', 'uint8', [1,3,32,32])


    def test_leakyrelu_fp16_ch3(self):
        self.rand_activate('leakyrelu', 'float16', [1,3,48,48])
    def test_leakyrelu_fp32_ch3(self):
        self.rand_activate('leakyrelu', 'float32', [1,3,32,32])
#    def test_leakyrelu_uint8_ch3(self):
#        self.rand_activate('leakyrelu', 'uint8', [1,3,32,32])

    def test_tanh_fp16_ch3(self):
        self.rand_activate('tanh', 'float16', [1,3,32,32])
    def test_tanh_fp32_ch3(self):
        self.rand_activate('tanh', 'float32', [1,3,32,32])
#    def test_tanh_uint8_ch3(self):
#        self.rand_activate('tanh', 'uint8', [1,3,32,32])

    def test_sigmoid_fp16_ch3(self):
        self.rand_activate('sigmoid', 'float16', [1,3,32,32])
    def test_sigmoid_fp32_ch3(self):
        self.rand_activate('sigmoid', 'float32', [1,3,32,32])
#    def test_sigmoid_uint8_ch3(self):
#        self.rand_activate('sigmoid', 'uint8', [1,3,32,32])

if __name__ == '__main__':
    unittest.main()
