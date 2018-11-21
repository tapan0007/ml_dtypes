import os
import argparse
import numpy as np
import unittest
import random
import copy
from subprocess import call
from me_models import PEArray

def ceildiv(x, y):
    return -(-x//y)

"""Fold data in NCHW/CRSM format to SB format CNcHW/CcRSM
    Args:
        dram_data: Numpy tensor data in NCHW
        file_format: string indicating format of tensor
        pad_all: when true, pad C to PEArray.NUM_ROWS channels; when false, keep original C
"""
def fold_data(dram_data, file_format, pad_all):
    assert(file_format == "NCHW" or file_format == "CRSM")

    def process_C(C):
        c_folds = ceildiv(C, PEArray.NUM_ROWS)
        if C < PEArray.NUM_ROWS and not pad_all:
            C_occupied = C
            C_padded = C
        else:        
            C_occupied = PEArray.NUM_ROWS
            C_padded = c_folds * PEArray.NUM_ROWS
        return (c_folds, C_occupied, C_padded)            

    if file_format == "NCHW":
        N, C, H, W = dram_data.shape
        (c_folds, C_occupied, C_padded) = process_C(C)
        new_format = "CNcHW"                    
        new_padded = np.pad(dram_data, ((0,0), (0, C_padded - C), (0,0), (0,0)), 'constant')
        new_folded = np.zeros((C_occupied, N, c_folds, H, W), dtype = dram_data.dtype)
        for n_id in range(N):
            for c_fold_id in range(c_folds):
                for i in range(C_occupied):
                    C_id = c_fold_id * PEArray.NUM_ROWS + i
                    new_folded[i, n_id, c_fold_id] = new_padded[n_id, C_id]
    else:        
        C, R, S, M = dram_data.shape
        (c_folds, C_occupied, C_padded) = process_C(C)
        new_format = "CcRSM"                    
        new_padded = np.pad(dram_data, ((0, C_padded - C), (0,0), (0,0), (0,0)), 'constant')
        new_folded = np.zeros((C_occupied, c_folds, R, S, M), dtype = dram_data.dtype)
        for c_fold_id in range(c_folds):
            for i in range(C_occupied):
                C_id = c_fold_id * PEArray.NUM_ROWS + i
                new_folded[i, c_fold_id] = new_padded[C_id]
    return (new_folded, new_format)                

"""Unfold data in SB format CNcHW/CcRSM to standard NCHW/CRSM format
    Args:
        dram_data: Numpy tensor data in NCHW
        file_format: string indicating format of tensor
        channel_count: Original unfolded channel count; 0 means computation needed.
"""
def unfold_data(dram_data, file_format, channel_count):
    assert(file_format == "CNcHW" or file_format == "CcRSM")

    def process_C(C):
        c_folds = ceildiv(C, PEArray.NUM_ROWS)
        C_occupied = PEArray.NUM_ROWS
        C_padded = c_folds * PEArray.NUM_ROWS
        return (c_folds, C_occupied, C_padded)            

    C = channel_count
    (c_folds, C_occupied, C_padded) = process_C(C)
    if file_format == "CNcHW":
        num_rows, N, c, H, W = dram_data.shape
        #assert(num_rows == min(PEArray.NUM_ROWS, C)), "num_rows %d is not equal to min between %d and %d"%(num_rows, PEArray.NUM_ROWS, C)
        assert(c_folds == c)
        new_format = "NCHW"                    
        new_unfolded = np.zeros((N, C, H, W), dtype = dram_data.dtype)
        for n_id in range(N):
            for c_fold_id in range(c_folds):
                for i in range(C_occupied):
                    C_id = c_fold_id * PEArray.NUM_ROWS + i
                    if C_id < C:
                        new_unfolded[n_id, C_id] = dram_data[i, n_id, c_fold_id]
    else:        
        num_rows, c, R, S, M = dram_data.shape
        assert(num_rows == PEArray.NUM_ROWS)
        assert(c_folds == c)
        new_format = "CcRSM"                    
        new_unfolded = np.zeros((C, R, S, M), dtype = dram_data.dtype)
        for c_fold_id in range(c_folds):
            for i in range(C_occupied):
                C_id = c_fold_id * PEArray.NUM_ROWS + i
                if C_id < C:
                    new_unfolded[C_id] = dram_data[i, c_fold_id]
    return (new_unfolded, new_format)                

"""Fold file data (NCHW or CRSM) to SB format (CNcHW or CcRSM)
"""
def fold_file(file_to_fold, file_format, file_to_save, pad_all):
    assert(file_format == "NCHW" or file_format == "NHWC" or file_format == "CRSM")
    dram_data = np.load(file_to_fold)
    format_temp = file_format
    if file_format == "NHWC":
        dram_data = np.transpose(dram_data, (0,3,1,2))
        format_temp = "NCHW"
    (new_folded_data, new_format) = fold_data(dram_data, format_temp, pad_all)
    np.save(file_to_save, new_folded_data)
    print("Old file %s format %s -> new file %s format %s"%(file_to_fold, file_format, file_to_save, new_format))

"""Unfold file data from SB format (CNcHW/CcRSM) to unfolded format (NCHW/CRSM)
"""
def unfold_file(file_to_unfold, file_format, file_to_save, channel_count):
    assert(file_format == "CNcHW" or file_format == "CcRSM")
    dram_data = np.load(file_to_unfold)
    format_temp = file_format
    (new_unfolded_data, new_format) = unfold_data(dram_data, format_temp, channel_count)
    np.save(file_to_save, new_unfolded_data)
    print("Old file %s format %s -> new file %s format %s"%(file_to_unfold, file_format, file_to_save, new_format))

"""Add folded data to combined file: only works if inner dimensions match
    returns: 
        c_fold: the fold number for the newly added data
"""
def add_folded_data_to_file(new_folded_data, file_to_save, file_format="CNcHW"):    
    assert(file_format == "CNcHW" or file_format == "CcRSM")
    try:
        dram_data = np.load(file_to_save)
    except Exception as e:
        print(e)
        print("WARNING: file %s doesn't exist, so create new file with new data"%file_to_save)
        np.save(file_to_save, new_folded_data)
        return 0

    c_fold_dim = 2 if file_format == "CNcHW" else 1
    old_shape = dram_data.shape
    add_shape = new_folded_data.shape 
    # Check that shape is 5D and H, W, N are the same
    assert(len(old_shape) == 5)
    assert(len(add_shape) == 5)
    assert(old_shape[3:] == add_shape[3:])
    # Extend the c dimension
    if c_fold_dim == 1:
        new_folded = np.zeros((old_shape[0], old_shape[1]+add_shape[1], old_shape[2], old_shape[3], old_shape[4]), dtype = dram_data.dtype)
    else:                    
        new_folded = np.zeros((old_shape[0], old_shape[1], old_shape[2]+add_shape[2], old_shape[3], old_shape[4]), dtype = dram_data.dtype)
    c_fold = old_shape[c_fold_dim]
    if c_fold_dim == 1:
        new_folded[:, 0:c_fold] = dram_data
        new_folded[:, c_fold:] = new_folded_data
    else:            
        new_folded[:, :, 0:c_fold] = dram_data
        new_folded[:, :, c_fold:] = new_folded_data
    print("DBG: added data to %s at fold index %d"%(file_to_save, c_fold))
    np.save(file_to_save, new_folded)
    return c_fold

"""Unit tests
"""
class TestFolding(unittest.TestCase):
    def test_nchw_padded(self):
        for i in range(10):
            C = random.randrange(1, 512, 10)
            N = random.randint(1,16)
            H = random.randint(1,64)
            W = random.randint(1,32)
            t = np.random.random((N,C,H,W))
            np.save("testfile.npy", t)
            fold_file("testfile.npy", "NCHW", "testfile2.npy", pad_all=True)
            c_folds = ceildiv(C, PEArray.NUM_ROWS)
            C_padded = c_folds * PEArray.NUM_ROWS
            t2 = np.load("testfile2.npy")
            t2_shape = t2.shape
            self.assertEqual(t2_shape, (PEArray.NUM_ROWS, N, c_folds, H, W))
            for n_id in range(N):
                for c_id in range(c_folds):
                    for r_id in range(PEArray.NUM_ROWS):
                        C_id = c_id * PEArray.NUM_ROWS + r_id
                        if C_id < C:
                            self.assertEqual((t[n_id, C_id] == t2[r_id, n_id, c_id]).all(), True)
            unfold_file("testfile2.npy", "CNcHW", "testfile3.npy", channel_count=C)
            t3 = np.load("testfile3.npy")
            t3_shape = t3.shape
            self.assertEqual(t3_shape, t.shape)
            self.assertEqual((t==t3).all(), True)

    def test_crsm_padded(self):
        for i in range(10):
            C = random.randrange(1, 512, 10)
            R = random.randint(1,16)
            S = random.randint(1,64)
            M = random.randrange(1, 512, 9)
            t = np.random.random((C,R,S,M))
            np.save("testfile.npy", t)
            fold_file("testfile.npy", "CRSM", "testfile2.npy", pad_all=True)
            c_folds = ceildiv(C, PEArray.NUM_ROWS)
            C_padded = c_folds * PEArray.NUM_ROWS
            t2 = np.load("testfile2.npy")
            t2_shape = t2.shape
            self.assertEqual(t2_shape, (PEArray.NUM_ROWS, c_folds, R, S, M))
            for c_id in range(c_folds):
                for r_id in range(PEArray.NUM_ROWS):
                    C_id = c_id * PEArray.NUM_ROWS + r_id
                    if C_id < C:
                        self.assertEqual((t[C_id] == t2[r_id, c_id]).all(), True)
            unfold_file("testfile2.npy", "CcRSM", "testfile3.npy", channel_count=C)
            t3 = np.load("testfile3.npy")
            t3_shape = t3.shape
            self.assertEqual(t3_shape, t.shape)
            self.assertEqual((t==t3).all(), True)

    def test_nhwc_unpadded(self):
        for i in range(10):
            C = random.randrange(1, 512, 7)
            N = random.randint(16,32)
            H = random.randint(16,32)
            W = random.randint(32,64)
            t = np.random.random((N,H,W,C))
            np.save("testfile.npy", t)
            fold_file("testfile.npy", "NHWC", "testfile2.npy", pad_all=False)
            c_folds = ceildiv(C, PEArray.NUM_ROWS)
            C_padded = c_folds * PEArray.NUM_ROWS
            t2 = np.load("testfile2.npy")
            t2_shape = t2.shape
            self.assertEqual(t2_shape, (min(PEArray.NUM_ROWS, C), N, c_folds, H, W))
            for n_id in range(N):
                for c_id in range(c_folds):
                    for r_id in range(PEArray.NUM_ROWS):
                        C_id = c_id * PEArray.NUM_ROWS + r_id
                        if C_id < C:
                            self.assertEqual((t[n_id, :, :, C_id] == t2[r_id, n_id, c_id]).all(), True)
            unfold_file("testfile2.npy", "CNcHW", "testfile3.npy", channel_count=C)
            t3 = np.load("testfile3.npy")
            t3_shape = t3.shape
            t_nchw = np.transpose(t, (0,3,1,2))
            self.assertEqual(t3_shape, t_nchw.shape)
            self.assertEqual((t_nchw==t3).all(), True)

class TestPacking(unittest.TestCase):
    def test_pack(self):
        call(["rm", "-f", "testfile.npy"])
        t_list = []
        c_fold_list = []
        for i in range(10):
            C = random.randrange(1, 512, 7)
            N = 2
            H = 1
            W = 1
            t = np.random.random((N,C,H,W))
            t_list.append(t)
            (new_folded_data, new_format) = fold_data(t, "NCHW", True)
            c_fold_offset = add_folded_data_to_file(new_folded_data, "testfile.npy")
            c_fold_list.append(c_fold_offset)
            data = np.load("testfile.npy")
            print(i, " tensor shape ", t.shape, " new folded data shape ", new_folded_data.shape, " combined file shape ", data.shape)
        data = np.load("testfile.npy")
        c_fold_offset = 0
        for i in range(10):
            t = t_list[i]
            C = t.shape[1]
            c_folds = ceildiv(C, PEArray.NUM_ROWS)
            extracted_data = data[:, :, c_fold_offset : c_fold_offset + c_folds]
            self.assertEqual(c_fold_offset, c_fold_list[i])
            c_fold_offset += c_folds
            (new_unfolded_data, _) = unfold_data(extracted_data, "CNcHW", C)
            self.assertEqual((new_unfolded_data==t).all(), True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", help="Input numpy file for folding")
    parser.add_argument("--in_format", default="NCHW", help="Format of the input numpy file for folding (NCHW, NHWC, or CRSM) or unfolding (CNcHW, CcRSM)")
    parser.add_argument("--out_file", help="Post-folded numpy file")
    parser.add_argument("--channel_count", type=int, default=0, help="Original unfolded channel count; 0 means computation needed.")
    parser.add_argument("--pad_all", action='store_true', help="Pad even when C<PEArray.NUM_ROWS")
    args = parser.parse_args()

    if args.in_format == "NCHW" or args.in_format == "NHWC" or args.in_format == "CRSM":
        fold_file(args.in_file, args.in_format, args.out_file, args.pad_all)
    elif args.in_format == "CNcHW" or args.in_format == "CcRSM":
        unfold_file(args.in_file, args.in_format, args.out_file, args.channel_count)
    else:            
        raise RuntimeError("Input format must be one of NCHW, NHWC, or CRSM for folding, and CNcHW or CcRSM for unfolding")
