########################################################
MODULE:=layers
MODS := $(MODULE) utils nets serialize

FILES := activ relu tanh sqrt combine subsample conv \
		 const input resadd biasadd pool maxpool avgpool \
		 matmul reshape

SRCS := layer $(addsuffix layer,$(FILES))

########################################################
include ../Makefile.common

