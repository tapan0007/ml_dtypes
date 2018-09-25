########################################################
MODULE:=wave
MODS := $(MODULE) utils nets serialize

FILES := pooleng matmul sbatom sbatomload sbatomsave pool activation resadd barrier nop scaleadd clipbyvalue

SRCS := waveedge waveop $(addsuffix waveop,$(FILES))

########################################################
include ../Makefile.common

