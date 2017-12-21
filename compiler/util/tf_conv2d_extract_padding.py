#!/usr/bin/env python3

# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Extract padding info for conv2d by running sweeps over image, weight, filter sizes

# To run:
#  gpu -> this looks correct
#    python3 tf_conv2d_extract_padding.py >& log-gpu
#  cpu
#    ( setenv CUDA_VISIBLE_DEVICES ; python3 tf_conv2d_extract_padding.py) >& log-cpu

# Get padding for stride 2, image 6, filter 3
#   cat log-gpu | grep S=2 | grep H=6 | grep R=3 
#   egrep -B 2 'S=2 R=3 H=6' log-gpu

import tensorflow as tf
import numpy as np
import sys
import re

np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
tfDataType = np.float32
padding = "SAME"


def calcPadLeft(val, weight):
  pad = 0
  i = 0
  while (val % (10 *weight[i])) // weight[i] == 0:
    pad += 1
    i += 1
  assert i <= len(weight)
  return pad

def calcPadRight(val, weight):
  pad = 0
  i = len(weight) - 1
  while (val % (10 *weight[i])) // weight[i] == 0:
    pad += 1
    i -= 1
  assert(i >= 0)
  return pad

# Tasks - only horizontal dimension:
# weight R, R, C, M = 1, R, 1, 1
# image  B, H, H, C = 1, 1, H, 1
# stride 1, S, S, 1 = 1, 1, S, 1

tasks = []

for S in range(1, 6):
  for R in range(1, 6):
    for H in range(1, 10):
      w = np.logspace(R-1, 0, R, dtype=tfDataType).reshape(1, R, 1, 1)
      i = np.linspace(1, H, H, dtype=tfDataType).reshape(1, 1, H, 1)
      s = [1, 1, S, 1]
      SRH = (S, R, H)
      tasks.append((w, i, s, SRH))
      #print("Added task ", w.ravel(), i.ravel(), s, SRH)

for (w, i, strides, SRH) in tasks:

  tf.reset_default_graph()
  w1 = tf.get_variable(name="weight1",
                     initializer = w, dtype=tfDataType)
  i0 = tf.placeholder(tfDataType, shape=i.shape, name="input")
  i1 = tf.nn.conv2d(i0, w1, strides, padding, name="i1")
  output = tf.identity(i1, name="output")

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(output, feed_dict={"input:0" : i})
    print("  Input= ", i.ravel().astype(np.int64),
          "  Weight= ", w.ravel().astype(np.int64),
          "\n  Res=", res.ravel().astype(np.int64))
    resList = list(res.ravel().astype(np.int64))
    weightList = list(w.ravel().astype(np.int64))
    padL = calcPadLeft(resList[0], weightList)
    padR = calcPadRight(resList[-1], weightList)
    print("    S=%d R=%d H=%d" % SRH, "PadLeft=", padL, "PadRight=", padR)
    print() 

