top : exe
########################################################
MODULE:=compiler
#MODS := utils nets layers arch memmgr schedule serialize codegen wave
MODS :=  \
	wavecode	\
	codegen		\
	schedule	\
	memmgr		\
	codegen		\
	wavecode	\
	nets		\
	serialize	\
	layers		\
	events		\
	wave		\
	arch		\
	utils		\
	compisa


SRCS := compiler


include ../Makefile.common


########################################################
TCC_LIB := $(INKLING_PATH)/tcc/libs/libtcc.a

LIBS1 := $(foreach L,$(MODS),$(COMPILER_BE)/$(L)/$(OBJDIR)/lib$(L).a)
LIBS := $(LIBS1) $(TCC_LIB)
########################################################

########################################################
ADDITIONAL_CLEAN_FILES := $(MODULE).exe
########################################################

########################################################

exe : $(MODULE).exe
########################################################

$(MODULE).exe : $(OFILES) $(LIBS)
	@echo LD $@
	$(COMPILER) $(LDOPTS) $^ -o $@

