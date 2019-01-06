########################################################
MODULE:=wave
MODS := $(MODULE) utils nets serialize

FILES := pooleng matmul pool activation \
datamove sbatom sbatomload sbatomsave  tpbcopy \
tensor tensortensor tensorscalar tensorscalarptr \
nop scaleadd clipbyvalue reciprocal \
reg regload regstore

SRCS := waveedge waveop $(addsuffix waveop,$(FILES))

########################################################
include ../Makefile.common

