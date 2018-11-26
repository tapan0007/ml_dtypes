########################################################
MODULE:=wavecode
MODS := $(MODULE) kelf utils nets layers wave

########################################################
FILES := waveop sbatom sbatomload sbatomsave matmul pool \
activation clipbyvalue tensortensor tensorscalarconst barrier nop \
reciprocal

SRCS := wavecode wavecode_gen $(addprefix $(MODULE),$(FILES))

########################################################
include ../Makefile.common

