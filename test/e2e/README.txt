# Copyright (C) 2017, Amazon.com. All Rights Reserved

Steps to run, and extract TPB instruction streams
-------------------------------------------------
1) Setup:

  a) Env
    export KAENA_PATH=/home/ubuntu/work/git/Kaena
    export INKLING_PATH=/home/ubuntu/work/git/Inkling
  b) Repos
    You can scp them or better use git. The setup is explained in
      new_hire.txt
        https://amazon.awsapps.com/workdocs/index.html#/document/5e982012aeac869ab97f4e233e0df65551e18e677a47d78268dbeb45092a14d1
        Search for "Code sharing from Kaena repo"
  
  c) Tools
     Use ubuntu ML AMI. On other platforms such as mac laptop you need to install various tools as explained in 
        https://code.amazon.com/packages/Kaena/blobs/mainline/--/compiler/tffe/test/Makefile
  

2) Run and analyze data

  a) Test your setup by running an existence test
      cd /any/empty/dir
      $KAENA_PATH/compiler/tffe/test/RunAll --verbose --test 0-1conv0
  b) Run
      /bin/rm -r [0-9]* ; $KAENA_PATH/compiler/tffe/test/RunAll

  b) Locate the TPB instruction streams
      find . -name \*.tpb -o -name \*.asm

  c) Details
      The .tpb and .asm are the machine  and assembly code respectively.
      As of Dec 2017 instead of stream processor there are "pseudo" instructions that load and write numpy files
      Utilities to compare, visualize, edit numpy files are in
        https://code.amazon.com/packages/Kaena/trees/mainline/--/compiler/util
  

Harness list
------------
  Harnesses are small python scripts that create fully functional neural
  networks (more accurately compute graphs). Typically they are heavily
  parametrizable using a simple text string.

Single-operator:
  trivnet_act.py -> activation tanh, relu
  trivnet_biasadd.py -> bias add
  trivnet_ap1.py -> average pool
  trivnet_mp1.py -> max pool
  trivnet_conv1.py -> conv2d
  trivnet_matmult1.py

Short operator sequences:
  trivnet_add.py -> bias_add with residual add
  trivnet_conv2.py -> 2 layers of Conv2D
  trivnet_conv_pool.py
  trivnet_conv_pool_conv.py
  trivnet_matmul_add.py
  trivnet_scaleadd.py -> scalar multiply followed by add

Longer patterns
  trivnet_conv_ba.py - conv + bias_add + relu/tanh
  trivnet_conv_ba_add.py -> residual add with resnet-like conv branches
  trivnet_conv_ba_mult.py -> residual multioply with resnet-like conv branches
  trivnet_lin.py - multi layer sequence of conv2d followed by act (tanh or relu)

Full networks (and their slices):
  tf_pb -> bypass harness that just passes an existing  tensorflow freeze
           graph neural network representation directly to Kaena compiler

Obsoleted (same can be done more efficiently by other hanesses)
  trivnet_conv1_padvalid.py

  
