import argparse
import random
import os
import numpy as np

def create(dtype, values, dims, vmin=None, vmax=None):
    if values:
        tgt_dim = reduce((lambda x,y:x*y), dims)
        if len(values) == 1:
            values = [values] * tgt_dim
        elif len(values) != tgt_dim:
            print "values {} can't be reshaped into {} arrays of dimension {}".format(values, dims)
            exit(1)
        A = np.array(values, dtype=dtype).reshape(dims)
    else:
        if vmin == None:
            if 'int' in dtype: 
                vmin = np.iinfo(dtype).min
            else:
                vmin = np.finfo(dtype).min
        if vmax == None:
            if 'int' in dtype: 
                vmax = np.iinfo(dtype).max
            else:
                vmax = np.finfo(dtype).max
        A = np.random.uniform(vmin, vmax, dims).astype(dtype, order='C')
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
