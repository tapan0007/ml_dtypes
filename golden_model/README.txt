======================================================================
README for Inkling Simulator 
======================================================================

OVERVIEW
--------
The Inkling simulator is a non-timing functional simulator that gives the 
expected mathematical results for DNN primitives run on Mariana hardware.  It
also includes support for running these primitives on tensorflow (done) and mxnet
(todo) for verification purposes. 

The primitives supported include:
Convolution
	w/ striding	
	w/ padding
	w/dilation
Pooling	
	max	
	avg
Activation
	tanh
	sigmoid	
	relu
	leaky-relu
MLP	
RNN

The data supported include:
fp16
fp32
uint8

The simulator gives the expected output and reports overflow and underflow with
warning logs.

MODEL SUPPORT
----------------
All combinations of data types and primitives run on the Mariana model.  The
MxNet model supports all primitives at fp32.  The Tensorflow model supports all
primitives at fp32 and fp16, only convolution and pooling for uint8

RUNNING
-------------
All input feature maps and output features maps are in numpy array format.  The
data type used in the model is the same input type as the input files.  The
input files follow mxnet conventions for ordering of dimensions, i.e. 
[#Batches, #Channels, #Rows, #Columns]

The primitives can be called from the python package directly
pkgs/mn_primitives.py for mariana
pkgs/tf_primitives.py for tensorflow
pkgs/mx_primitives.py for mxnet

The primitives can be called from the command line through the wrapper 
pkgs/cmd_line.py.  It outputs the numpy array. 
The format of the command is:
./cmd_line [framework abbrev] [primitive name] [primitive options]
./cmd_line [tf/mx/mn] [convolve,max_pool,avg_pool,relu,leakyrelu,sigmoid,tanh,fullyconnected] [primitive options]

TESTING
------------
The base directory includes a ./test.sh file that uses pythons unittest pkg to
 run all of the tests in the tests directory

