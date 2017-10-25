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


class TestConvolve(unittest.TestCase):
    def convolve(self, img, filtr, stride=None, pad=None, dilate=None):
        base_args = ["convolve", "-i", img,  "-f", filtr]
        s_args = [] if stride == None else ["-s"] + [str(x) for x in stride]
        p_args = [] if pad    == None else ["-p"] + [str(x) for x in pad]
        d_args = [] if dilate == None else ["-d"] + [str(x) for x in dilate]
        all_args = base_args + s_args + p_args + d_args
        args = cmd_line.parser.parse_args(["tf"] + all_args)
        tf_o = args.func(args)
        args = cmd_line.parser.parse_args(["mn"] + all_args)
        mn_o = args.func(args)
        err_msg = "Suffix of command line mn vs tf: {}".format(" ".join(all_args))
        with np.errstate(under='ignore'):
            np.testing.assert_allclose(mn_o, tf_o, rtol=1e-03, err_msg=err_msg)
    def rand_convolve(self, prec, i_dims, f_dims, stride=None, pad=None, dilate=None):
        img = test_utils.randf('i', prec, i_dims)
        filtr = test_utils.randf('f', prec, f_dims)
        self.convolve(img, filtr, stride, pad, dilate)
    def test_fp16_ch1(self):
        self.rand_convolve('float16', [1,1,256,256], [1,1,3,3])
    def test_fp16_ch3(self):
        self.rand_convolve('float16', [1,3,64,64], [1,3,3,3])
    def test_fp32_ch1(self):
        self.rand_convolve('float32', [1,1,128,128], [1,1,3,3])
    def test_fp32_ch3(self):
        self.rand_convolve('float32', [1,3,16,16], [1,3,5,5])
    def test_uint8_ch1(self):
        self.rand_convolve('uint8', [1,1,128,128], [1,1,3,3])
    def test_uint8_ch3(self):
        self.rand_convolve('uint8', [1,3,32,32], [1,3,3,3])
    def test_stride(self):
        self.rand_convolve('uint8', [1,3,32,32], [1,3,3,3], stride=[4,4])
    def test_fp_pad(self):
        self.rand_convolve('float16', [1,1,32,32], [1,1,3,3], pad=[3, 3])
    def test_fp_dilate(self):
        self.rand_convolve('float16', [1,1,48,48], [1,1,4,4], dilate=[2,2])
    def test_uint8_pad(self):
        self.rand_convolve('uint8', [1,1,32,32], [1,1,3,3], pad=[2, 2])
    def test_stride_pad(self):
        self.rand_convolve('uint8', [1,1,64,64], [1,1,5,5], stride=[2, 2], pad=[4, 4])
    def test_stride_dilate(self):
        self.rand_convolve('float16', [1,1,32,32], [1,1,3,3], stride=[2, 2], dilate=[3, 3])
    def test_pad_dilate(self):
        self.rand_convolve('float16', [1,1,32,32], [1,1,3,3], pad=[2, 2], dilate=[4,4])
    def test_stride_pad_dilate(self):
        self.rand_convolve('float16', [1,1,32,32], [1,1,3,3], stride=[1, 1], pad=[4, 4], dilate=[2,2])
    @mock.patch('pkgs.mn_primitives.logging')
    def test_uint8_ch2000(self, mock_logging):
	infile = 'test/input/nonrandom/i_uint8_1x2000x6x6_onehot.npy'
	fltrfile = 'test/input/nonrandom/f_uint8_1x2000x5x5_max.npy'
        self.convolve(infile, fltrfile)
       # also check overflow statements were made
        self.assertTrue(mock_logging.warn.called)

if __name__ == '__main__':
    unittest.main()
