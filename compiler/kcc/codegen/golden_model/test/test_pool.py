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


class TestPool(unittest.TestCase):
    def pool(self, prefix, ifile, k, s):
        base_args = [prefix, "-i", ifile,  "-ksize"] + [str(x) for x in k] + ["-stride"] + [str(x) for x in s]
        args = cmd_line.parser.parse_args(["tf"] + base_args)
        tf_o = args.func(args)
        args = cmd_line.parser.parse_args(["mn"] + base_args)
        mn_o = args.func(args)
        err_msg = "Suffix of command line: {}".format(" ".join(base_args))
        rtol_dict={'max_pool': 0.05,
                   'avg_pool': 0.15}
        with np.errstate(under='ignore'):
            np.testing.assert_allclose(tf_o, mn_o, rtol=rtol_dict[prefix], err_msg=err_msg)
    def rand_pool(self, prefix, prec, i_dims, k, s):
        if prec == 'uint8':
            img = test_utils.randf('i', prec, i_dims)
        else:
            img = test_utils.randf('i', prec, i_dims)
        self.pool(prefix, img, k, s)

    def test_max_fp16_ch1(self):
        self.rand_pool('max_pool', 'float16', [1,1,64,64], [1, 1, 1, 1], [1, 1, 1, 1])
    def test_max_fp16_ch3(self):
        self.rand_pool('max_pool', 'float16', [1,3,32,32], [1, 1, 3, 3], [1, 1, 1, 1])
    def test_max_fp32_ch1(self):
        self.rand_pool('max_pool', 'float32', [1,1,64,64], [1, 1, 1, 1], [1, 1, 2, 2])
    def test_max_fp32_ch3(self):
        self.rand_pool('max_pool', 'float32', [1,3,32,32], [1, 1, 5, 5], [1, 1, 5, 5])
    def test_max_uint8_ch1(self):
        self.rand_pool('max_pool', 'uint8', [1,1,64,64], [1, 1, 5, 5], [1, 1, 2, 2])
    def test_max_uint8_ch3(self):
        self.rand_pool('max_pool', 'uint8', [1,3,32,32], [1, 1, 2, 2], [1, 1, 3, 3])

    def test_avg_fp16_ch1(self):
        self.rand_pool('avg_pool', 'float16', [1,1,64,64], [1, 1, 1, 1], [1, 1, 1, 1])
    def test_avg_fp16_ch3(self):
        self.rand_pool('avg_pool', 'float16', [1,3,32,32], [1, 1, 3, 3], [1, 1, 1, 1])
    def test_avg_fp32_ch1(self):
        self.rand_pool('avg_pool', 'float32', [1,1,64,64], [1, 1, 1, 1], [1, 1, 2, 2])
    def test_avg_fp32_ch3(self):
        self.rand_pool('avg_pool', 'float32', [1,3,32,32], [1, 1, 5, 5], [1, 1, 5, 5])
    def test_avg_uint8_ch1(self):
        self.rand_pool('avg_pool', 'uint8', [1,1,64,64], [1, 1, 5, 5], [1, 1, 2, 2])
    def test_avg_uint8_ch3(self):
        self.rand_pool('avg_pool', 'uint8', [1,3,32,32], [1, 1, 2, 2], [1, 1, 3, 3])

if __name__ == '__main__':
    unittest.main()
