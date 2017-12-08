# Copyright (C) 2017, Amazon.com. All Rights Reserved

Steps to run, and extract TPB instruction streams
-------------------------------------------------
1) Setup:

  a) Env
    export KAENA_PATH=/home/ubuntu/work/git/Kaena
    export INKLING_PATH=/home/ubuntu/work/git/Inkling
  b) Repos
    You can scp them or betetr use git. The setup is explained in
      new_hire.txt
        https://amazon.awsapps.com/workdocs/index.html#/document/5e982012aeac869ab97f4e233e0df65551e18e677a47d78268dbeb45092a14d1
        Search for "code sharing from kaena repo"
  
  c) Tools
     Use ubuntu ML AMI. On other platforms such as mac laptop you need to install tool explained in tffe/tests/Makefile
       https://amazon.awsapps.com/workdocs/index.html#/document/5e982012aeac869ab97f4e233e0df65551e18e677a47d78268dbeb45092a14d1
  
  d) Run
      cd /any/empty/dir
      \rm -r [0-9]* ; ./RunAll --verbose
  
  e) Locate the TPB instruction streams
      find . -name \*.tpb -o -name \*.asm
      # The .tpb and .asm are the machine  and assembly code.
      # As of Dec 2017 instead of stream processor theree are "pseudo" instructions that load and write numpy files
  
