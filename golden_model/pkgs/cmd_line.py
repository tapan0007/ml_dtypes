#!/usr/bin/python 
"""

"""
import argparse
import sys
import numpy as np
import tf_primitives  as tfp
import mn_primitives as mnp

pkg={'tf':tfp, 'mn':mnp}

def convolve(args):
    i = np.load(args.i)
    f = np.load(args.f)
    s = args.stride
    d = args.dilate
    p = args.padding
    return pkg[args.model].convolve(i, f, stride=s, dilate=d, padding=p)


def max_pool(args):
    i = np.load(args.i)
    ksize  = args.ksize
    stride = args.stride
    return pkg[args.model].max_pool(ifmaps=i, ksize=ksize, strides=stride)

def avg_pool(args):
    i = np.load(args.i)
    ksize  = args.ksize
    stride = args.stride
    return pkg[args.model].avg_pool(ifmaps=i, ksize=ksize, strides=stride)

def relu(args):
    i = np.load(args.i)
    return pkg[args.model].relu(i)

def leakyrelu(args):
    i = np.load(args.i)
    return pkg[args.model].leakyrelu(i)

def tanh(args):
    i = np.load(args.i)
    return pkg[args.model].tanh(i)

def sigmoid(args):
    i = np.load(args.i)
    return pkg[args.model].sigmoid(i)

def fullyconnected(args):
    i = np.load(args.i)
    w = np.load(args.weight)
    b = None if args.no_bias == False else np.load(args.bias) #mxnet uses args.bias
    h = args.num_hidden
    return pkg[args.model].fullyconnected(i, weight=w, bias=b, num_hidden=h)

parser = argparse.ArgumentParser()

parser.add_argument('model', type=str, choices=["tf","mn"], help="model to run, tf=tensorflow, mn=mariana")

sp = parser.add_subparsers(title='action')

# convolve
sp_convolve = sp.add_parser('convolve', help='run convolve')
sp_convolve.add_argument('-i', type=str, required=True, help='input file')
sp_convolve.add_argument('-f', type=str, required=True, help='filter file')
sp_convolve.add_argument('-s', '--stride', type=int, nargs=2, default=[1,1], help='row_stride col_stride')
sp_convolve.add_argument('-d', '--dilate', type=int, nargs=2, default=[0,0], help='row_dilate col_dilate')
sp_convolve.add_argument('-p', '--padding', type=int, nargs=2, default=[0,0], help='row_padding col_padding')
sp_convolve.add_argument('-o', type=str, default=None, help="numpy output file")
sp_convolve.set_defaults(func=convolve)

# max_pool
sp_max_pool = sp.add_parser('max_pool', help='run max_pooling')
sp_max_pool.add_argument('-i', type=str, required=True, help='input file')
sp_max_pool.add_argument('-ksize', nargs=4, type=int, required=True, help='kernel size (#batches, #channels, #rows, #cols)')
sp_max_pool.add_argument('-stride', nargs=4, type=int, required=True, help='strides (#batches, #channels, #rows, #cols)')
sp_max_pool.add_argument('-o', type=str, default=None, help="numpy output file")
sp_max_pool.set_defaults(func=max_pool)

# avg_pool
sp_avg_pool = sp.add_parser('avg_pool', help='run avg_pooling')
sp_avg_pool.add_argument('-i', type=str, required=True, help='input file')
sp_avg_pool.add_argument('-ksize', nargs=4, type=int, required=True, help='kernel size (#batches, #channels, #rows, #cols)')
sp_avg_pool.add_argument('-stride', nargs=4, type=int, required=True, help='strides (#batches, #channels, #rows, #cols)')
sp_avg_pool.add_argument('-o', type=str, default=None, help="numpy output file")
sp_avg_pool.set_defaults(func=avg_pool)

# relu activation
sp_relu = sp.add_parser('relu', help='run relu activation')
sp_relu.add_argument('-i', type=str, required=True, help='input file')
sp_relu.add_argument('-o', type=str, default=None, help="numpy output file")
sp_relu.set_defaults(func=relu)

# leakyrelu activation
sp_leakyrelu = sp.add_parser('leakyrelu', help='run leakyrelu activation')
sp_leakyrelu.add_argument('-i', type=str, required=True, help='input file')
sp_leakyrelu.add_argument('-o', type=str, default=None, help="numpy output file")
sp_leakyrelu.set_defaults(func=leakyrelu)

# sigmoid activation
sp_sigmoid = sp.add_parser('sigmoid', help='run sigmoid activation')
sp_sigmoid.add_argument('-i', type=str, required=True, help='input file')
sp_sigmoid.add_argument('-o', type=str, default=None, help="numpy output file")
sp_sigmoid.set_defaults(func=sigmoid)

# tanh activation
sp_tanh = sp.add_parser('tanh', help='run tanh activation')
sp_tanh.add_argument('-i', type=str, required=True, help='input file')
sp_tanh.add_argument('-o', type=str, default=None, help="numpy output file")
sp_tanh.set_defaults(func=tanh)

# fullyconnected
sp_fullyconnected = sp.add_parser('fullyconnected', help='run fullyconnected activation')
sp_fullyconnected.add_argument('-i', type=str, required=True, help='input file')
sp_fullyconnected.add_argument('-w', "--weight",  type=str, required=True, help='weights file')
sp_fullyconnected.add_argument('-n', "--num_hidden",  type=int, required=True, help='num hidden')
sp_fullyconnected.add_argument('-b', "--bias", default=None, type=str, help='bias file')
sp_fullyconnected.add_argument("--no_bias", action='store_true', help='ignore bias')
sp_fullyconnected.add_argument('-o', type=str, default=None, help="numpy output file")
sp_fullyconnected.set_defaults(func=fullyconnected)

if __name__ == "__main__":
    np.set_printoptions(threshold='nan')
    args = parser.parse_args()
    output = args.func(args)
    if args.o == None:
        print output
    else:   
        np.save(args.o, output)
        
