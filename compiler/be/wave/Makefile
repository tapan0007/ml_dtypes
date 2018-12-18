########################################################
MODULE:=wave
MODS := $(MODULE) utils nets serialize

FILES := pooleng matmul sbatom sbatomload sbatomsave pool activation \
tensor tensortensor tensorscalar tensorscalarptr \
barrier nop scaleadd clipbyvalue reciprocal \
reg regload regstore

SRCS := waveedge waveop $(addsuffix waveop,$(FILES))

########################################################
include ../Makefile.common

