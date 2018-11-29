########################################################
MODULE:=utils
MODS := $(MODULE) nets layers serialize schedule

########################################################
SRCS := fmapdesc debug datatype types asserter version
## consts

GIT_SHA_LONG:=$(shell git describe --always --abbrev=12)
GIT_SHA_SHORT:=$(shell git describe --always --abbrev=7 | sed -e 's/.*-g//')
GIT_SHA_LONG := -DGIT_SHA_LONG='"'$(GIT_SHA_LONG)'"'
GIT_SHA_SHORT := -DGIT_SHA_SHORT='"'$(GIT_SHA_SHORT)'"'
########################################################
include ../Makefile.common

