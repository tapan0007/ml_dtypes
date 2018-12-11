########################################################
MODULE:=wavecode
MODS := $(MODULE) kelf utils nets layers wave

########################################################
FILES := waveop matmul pool activation              \
sbatom sbatomload sbatomload_sim sbatomload_kelf    \
       sbatomsave sbatomsave_sim sbatomsave_kelf    \
tpbcopy 											\
reciprocal clipbyvalue barrier nop regload regstore \
tensortensor tensorscalar tensorscalarptr

SRCS := wavecode wavecode_gen $(addprefix $(MODULE),$(FILES))

########################################################
include ../Makefile.common

