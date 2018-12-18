########################################################
MODULE:=wavecode
MODS := $(MODULE) kelf utils nets layers wave

########################################################
FILES := waveop sbatom sbatomload sbatomsave matmul pool \
activation clipbyvalue barrier nop reciprocal \
regload regstore \
tensortensor tensorscalar tensorscalarptr

SRCS := wavecode wavecode_gen $(addprefix $(MODULE),$(FILES))

########################################################
include ../Makefile.common

