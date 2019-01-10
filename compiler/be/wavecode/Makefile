########################################################
MODULE:=wavecode
MODS := $(MODULE) kelf utils nets layers wave

########################################################
FILES := waveop matmul pool activation               \
sbatom sbatomload sbatomload_sim sbatomload_kelf     \
tpbcopy sbatomsave sbatomsave_sim sbatomsave_kelf    \
reciprocal clipbyvalue nop                           \
regload regstore regshuffle                          \
tensortensor tensorscalar tensorscalarptr

SRCS := wavecode wavecode_gen $(addprefix $(MODULE),$(FILES))

########################################################
include ../Makefile.common

