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

class TestConvolve(unittest.TestCase):
    def gold_convolve(self, test_name, ifmap, filtr, stride=None, pad=None,
            dilate=None):
        ofmap = 'output/' + test_name + "_gold.npy"
        line = ["tf", "convolve", "-i", abspath(ifmap), "-f", abspath(filtr)]
        if pad:
            line = line + ["-p"] + pad;
        if stride:
            line = line + ["-s"] + stride;
        print "Running: " + " ".join(line)
        args = cmd_line.parser.parse_args(line)
        output = args.func(args)
        np.save(ofmap, output)
        return ofmap
    def tpu_convolve(self, test_name, ifmaps, filtrs, stride=None, pad=None, 
            dilate=None):
        ofmap = 'output/' + test_name + "_tpu.npy"
        binary = 'binary/' + test_name + '.bin'
        stdoutf = open('stdout/' + test_name + '.txt', "w")
        binary_line = [test_utils.TEST_CONVOLVE]
        if pad:
            binary_line += ["-p"] + map(str,pad)
        if stride:
            binary_line += ["-s"] + map(str,stride)
        for i in ifmaps:
            binary_line += ["-i"] + [abspath(i)]
        for f in filtrs:
            binary_line += ["-f"] + [abspath(f)]
        binary_line += [abspath(ofmap)]
        binary_line += [binary]
        sim_line = [test_utils.SIM,  binary]
        print "Compiling: " + " ".join(binary_line)
	call(binary_line)
        print "Running: " + " ".join(sim_line)
	call(sim_line, stdout=stdoutf)
        return ofmap
    def convolve_test(self, test_name, ifmap, filtr, stride=None, pad=None, 
            dilate=None, tpu_ifmaps=None, tpu_filters=None):
        print ""
        if not tpu_ifmaps:
            tpu_ifmaps = [ifmap]
        if not tpu_filters:
            tpu_filters = [filtr]
        o_gold = self.gold_convolve(test_name, ifmap, filtr, pad=pad, stride=stride)
        o_tpu = self.tpu_convolve(test_name, tpu_ifmaps, tpu_filters, pad=pad, 
                stride=stride)
        npdiff.diff(o_gold, o_tpu)
    def rand_convolve_test(self, test_name, iprec, idims, fprec, fdims, 
            stride=None, pad=None, dilate=None):
        if type(idims[0]) is list: # many ifmaps/filters 
           # build many ifmaps for tpu
           tpu_ifmaps = [test_utils.randf('i', iprec, i) for i in idims]
           tpu_filtrs = [test_utils.randf('f', fprec, f) for f in fdims]

           # concatenate ifmaps for tolden model
           np_ifmap = np.concatenate(tuple([np.load(i) for i in tpu_ifmaps]), 
			   axis=1)
           np_filtr = np.concatenate(tuple([np.load(f) for f in tpu_filtrs]),
			   axis=1)
           ifmap = test_utils.rand_fname('i', iprec, np.shape(np_ifmap))
           filtr = test_utils.rand_fname('f', fprec, np.shape(np_filtr))
           np.save(ifmap, np_ifmap)
           np.save(filtr, np_filtr)
        else:
           ifmap = test_utils.randf('i', iprec, idims)
           filtr = test_utils.randf('f', fprec, fdims)
           tpu_ifmaps = None
           tpu_filtrs = None
        self.convolve_test(test_name, ifmap, filtr, stride, pad, dilate, tpu_ifmaps, tpu_filtrs)
    def test_many_ifmap(self):
        #idims =  'input/ifmaps/i_uint8_1x128x24x24_rand.npy'
        #tpu_i = ['input/ifmaps/i_uint8_1x128x24x24_rand.npy']
        #f = 'input/filters/f_uint8_8x128x3x3_rand.npy'
        prec = 'uint8'
        idims = [[1,128,24,24], [1,4,24,24]]
        fdims = [[6,128,3,3], [6,4,3,3]]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
        #self.rand_convolve_test(tn, i, f)
    def test_3x3_filter_med_img_stride_2(self):
        prec = 'uint8'
        idims = [1,3,16,16]
        fdims = [1,3,3,3]
        tn = sys._getframe().f_code.co_name
        s = ["2","2"]
        self.rand_convolve_test(tn, prec, idims, prec, fdims, stride = s)
#    def test_3x3_filter_big_img_pad_1_stride_3(self):
#        prec = 'uint8'
#        idims = [1,3,32,32]
#        fdims = [1,3,3,3]
#        tn = sys._getframe().f_code.co_name
#        p = ["1", "1"]
#        s = ["2","3"]
#        self.rand_convolve_test(tn, prec, idims, prec, fdims, stride = s, pad = p)
    def test_1x1_filter_small_img_pad_1(self):
        prec = 'uint8'
        idims = [1,3,2,2]
        fdims = [2,3,1,1]
        p = ["1", "1"]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims) #, pad = p)
    def test_3x3_filter_small_img_pad_1(self):
        prec = 'uint8'
        idims = [1,3,3,3]
        fdims = [1,3,3,3]
        p = ["1", "1"]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)#, pad=p)
    def test_fp16_3x3_filter_small_img(self):
        prec = 'float16'
        idims = [1,3,3,3]
        fdims = [1,3,3,3]
        p = ["1", "1"]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
    def test_3x3_filter_big_img_pad_1(self):
        prec = 'uint8'
        idims = [1,3,32,32]
        fdims = [1,3,3,3]
        tn = sys._getframe().f_code.co_name
        p = ["1", "1"]
        self.rand_convolve_test(tn, prec, idims, prec, fdims)#, pad=p)
    def test_3x3_filter_med_img_pad_2(self):
        prec = 'uint8'
        idims = [1,3,16,16]
        fdims = [1,3,3,3]
        p = ["2", "2"]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
    def test_1x1_filter_small_img(self):
        prec = 'uint8'
        idims =  [1,3,2,2]
        fdims = [2,3,1,1]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
    def test_fp16_1x1_filter_small_img(self):
        prec = 'float16'
        idims =  [1,3,2,2]
        fdims = [2,3,1,1]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
    def test_1x1_filter_med_img(self):
        prec = 'uint8'
        idims =  [1,3,16,16]
        fdims = [2,3,1,1]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
    def test_3x3_filter_small_img(self):
        prec = 'uint8'
        idims =  [1,3,3,3]
        fdims = [1,3,3,3]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
    def test_3x3_filter_med_img(self):
        prec = 'uint8'
        idims =  [1,3,16,16]
        fdims = [1,3,3,3]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
    def test_fp16_3x3_filter_med_img(self):
        prec = 'float16'
        idims =  [1,3,16,16]
        fdims = [1,3,3,3]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
    def test_3x3_filter_big_img(self):
        prec = 'uint8'
        idims = [1,3,32,32]
        fdims = [1,3,3,3]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
    def test_1x1_filter_med_imgs_multich(self):
        prec = 'uint8'
        idims =  [1,3,16,16]
        fdims = [2,3,1,1]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
    def test_3x3_filter_med_imgs_multich(self):
        prec = 'uint8'
        idims =  [1,3,16,16]
        fdims = [2,3,3,3]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
    def test_3x2_filter_med_imgs_multich(self):
        prec = 'uint8'
        idims =  [1,3,16,16]
        fdims = [2,3,3,2]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)
    def test_3x3_filter_medrect_imgs_multich(self):
        prec = 'uint8'
        idims =  [1,3,16,8]
        fdims = [2,3,3,3]
        tn = sys._getframe().f_code.co_name
        self.rand_convolve_test(tn, prec, idims, prec, fdims)

if __name__ == '__main__':
    unittest.main()
