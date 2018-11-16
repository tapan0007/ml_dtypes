#!/apollo/sbin/envroot "$ENVROOT/bin/python"
# Copyright (C) 2018 Amazon Inc. All Rights Reserved.
#
# Bartosz Putrycz <bartosz@amazon.com>
import os
import sys
sys.path.insert(0, os.environ["KAENA_EXT_PATH"] + "/apps/tf/wavernn")
import argparse
from collections import OrderedDict
import logging
import pickle
from timeit import default_timer as timer
from Gru_tf import GRU_local

#import mxnet as mx
#import mxnet.ndarray as nd
#from mxnet.gluon import *
import numpy as np

logging.root.setLevel(logging.INFO)
import tensorflow as tf

def main():
    args = parse_args()

    batch_size = 1
    frame_to_sample_ratio = 300
    #set seed so tf produces same results across runs
    tf.set_random_seed(0)
    with open(args.affines_input, "rb") as reader:
        affines = pickle.load(reader)

    print('affines read: ', [k for k, _ in affines.items()])
    # ;-)
    F = np


    aff = affines['RNN']
    rnn_w = F.array(aff['weights'])
    print('RNN weight shape ',rnn_w.shape)


    aff = affines['O1']
    o1_w = F.array(aff['weights'])
    o1_w = o1_w.reshape((aff['rows'], aff['cols']))
    o1_b = F.array(aff['bias'])

    print('o1 weight shape ',o1_w.shape)
    print('o1 bias shape ',o1_b.shape)


    aff = affines['O2']
    o2_w = F.array(aff['weights'])
    o2_w = o2_w.reshape((aff['rows'], aff['cols']))
    o2_b = F.array(aff['bias'])

    print('o2 weight shape ',o2_w.shape)
    print('o2 bias shape ',o2_b.shape)


    aff = affines['input_audio']
    input_w = F.array(aff['weights'])
    #input_w = input_w.reshape((aff['rows'], aff['cols']))
    input_w = input_w.reshape((aff['cols'], aff['rows']))
    #input_w = input_w.T


    print('inpt_w  ',input_w[1])
    print('input_w weight shape ',input_w.shape)


    sample_size = input_w.shape[1]
    hidden_size = o1_w.shape[1]
    o_size = o1_b.shape[0]


    #vig
    mel_dim =256
    timesteps = 1
    rnn_time = 10
    hidden_dim = 896
    out_dim = 1024
    bs = 1
    in_dim = 1280
    print('in_shape, hidden_, 0_sh',sample_size,hidden_size,o_size)

    print('in elem ' ,[k for k, _ in aff.items()])
    conditioning = np.load(args.conditioning_input)
    cond_size = conditioning.items()[0][1].shape[-1]
    #select first condition
    cond = np.array(conditioning.items()[0][1][0])
    ######## Start
    cond = np.random.randn(1,bs*mel_dim*timesteps)
    cond_in = tf.placeholder(name='cond',shape=[None,bs*mel_dim*timesteps],dtype='float')
    prev_sample_in = tf.placeholder(name='prev',shape=[1],dtype='int32')

    #first prev_sample Init with random 1
    prev_sample = np.array([1],dtype='int32')
    #embedding defintion
    inp_lkup = tf.get_variable("inp_lkup",[1024,1024],initializer = tf.constant_initializer(input_w))
    embed_out = tf.gather_nd(inp_lkup,prev_sample_in)
    #embed_out = tf.gather_nd(inp_lkup,prev_sample)
    RNN_in = embed_out
    RNN_in = tf.reshape(RNN_in,(1,1024))
    sh = tf.shape(RNN_in)

    #GRU Cell
    gru_l = GRU_local(in_dim,hidden_dim)
    #ideally we dont need this, but adding a seperate GRU to enable easier cutting, so W params are not shared across timesteps;
    gru_2 = GRU_local(in_dim,hidden_dim)
    #lstm_out , state = tf.nn.static_rnn(lstm1,RNN_in,sequence_length=200)
    cond_in_0 = tf.split(cond_in,timesteps,1)
    cond_in_0 = tf.reshape(cond_in,(1,mel_dim))
    sh_cond_in_1 = tf.shape(cond_in_0)
    #GRU inputs first timestrep 0
    gru_in_0 = tf.concat([RNN_in,cond_in_0],1)
    #init_state =  tf.Variable(lstm0.zero_state(1,tf.float32),trainable=False)
    state0_var = np.zeros([1,896])
    init_state = tf.placeholder(name='init_state',shape=[1,896],dtype='float')
    

    #Fully Conncted Layer Weigths 1 : 896X1024
    w_fc0 = tf.Variable(tf.truncated_normal(dtype='float', shape=(896, 1024), mean=0, stddev=0.01), name='w_fc0')
    w_fc1 = tf.Variable(tf.truncated_normal(dtype='float', shape=(1024, 1024), mean=0, stddev=0.01), name='w_fc1')


    #Forward pass graph - Timestep0
    gru_state = gru_l.forward_pass(init_state,gru_in_0)
    gru_in_1 = tf.nn.relu(tf.matmul(gru_state,w_fc0))
    gru_in_1 = tf.matmul(gru_in_1,w_fc1)

    gru_in_1 = tf.nn.softmax(gru_in_1)
    prev_sample_1 = tf.multinomial(gru_in_1,1,output_dtype='int32',seed=0)
     
    #FP time step -1  
    embed_out = tf.gather_nd(inp_lkup,prev_sample_1)
    RNN_in_1 = embed_out
    RNN_in_1 = tf.reshape(RNN_in_1,(1,1024))
    RNN_in = gru_in_1

    gru_in_1 = tf.concat([RNN_in_1,cond_in_0],1)
    gru_state_1 = gru_2.forward_pass(gru_state,gru_in_1)

    #Global initialisation
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    np.save('cond.npy',cond)
    np.save('prev_samp.npy',prev_sample)

    with tf.Session() as session :
      session.run(init)
      #result = session.run(prev_sample_1,feed_dict={prev_sample_in : prev_sample , cond_in : cond , init_state : state0_var })
      result = session.run(gru_state_1,feed_dict={prev_sample_in : prev_sample , cond_in : cond , init_state : state0_var })
      tf.train.write_graph(session.graph,'.','wavernn_tf_ts1_cb_seed1.pbtxt')
      saver.save(session,'./wavernn_tf_ts1_cb_seed1')
      print(result)

def parse_args():
    """
    Parses arguments provided through cmd
    :return: structure with arguments
    """
    arg_parser = argparse.ArgumentParser(description='Inference from WaveRNN models')
    arg_parser.add_argument('--affines-input', required=True,
                            help='Input Amawave model file')
    arg_parser.add_argument('--model-type', required=True,
                            help='Model type (R1, curnn)')
    arg_parser.add_argument('--curnn-type',
                            help='curnn processing type (standard, expanded(not supported))')
    arg_parser.add_argument('--conditioning-input', required=True,
                            help='Input conditioning')
    arg_parser.add_argument('--array-output', required=True,
                            help='npz file to write to')
    arg_parser.add_argument('--use-cpu', action='store_true',
                            help='If to run on cpu, gpu is by default')
    arg_parser.add_argument('--no-hybridize', action='store_true',
                            help='If not to perform mxnet model hybridization')
    arg_parser.add_argument('--report-freq', type=int, default=10,
                            help='How often, every x seconds to report plan loop')

    args = arg_parser.parse_args()
    return args




if __name__ == '__main__':
    #test_compare_curnns()
    main()
