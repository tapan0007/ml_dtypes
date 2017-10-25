import argparse
import random
import os
import numpy as np

def create(dtype, values, dims):
    if values:
        tgt_dim = reduce((lambda x,y:x*y), dims)
        if len(values) == 1:
            values = [values] * tgt_dim
        elif len(values) != tgt_dim:
            print "values {} can't be reshaped into {} arrays of dimension {}".format(values, dims)
            exit(1)
        A = np.array(values, dtype=dtype).reshape(dims)
    else:
        if "int" in dtype:
            A = np.random.randint(np.iinfo(dtype).max, size=dims, dtype=dtype)
        else:
            A = np.array(np.random.rand(*dims), dtype=dtype)
    return A	

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',type=str, default="stdout", help='array file output')
    parser.add_argument('-d',type=int, nargs='+', required=True, help='dimensions of array (space seperated list)')
    parser.add_argument('--values', '-v', nargs='+', help='space-delineated value list (1D, will be reshaped)')
    parser.add_argument('--type', '-t', type=str, default='float64', help='data type', choices=['float64', 'float32', 'float16', 'int8', 'uint8', 'int16'])
    args = parser.parse_args()
    A = create(args.type, args.values, args.d)
    if args.o == "stdout":
        print A
    else:
        np.save(args.o, A)
