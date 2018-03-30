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
	utils

SRCS := compiler


include ../Makefile.common


########################################################
TCC_LIB := $(INKLING_PATH)/tcc/libs/libtcc.a

LIBS1 := $(foreach L,$(MODS),$(COMPILER_BE)/$(L)/$(OBJDIR)/lib$(L).a)
LIBS := $(LIBS1) $(TCC_LIB)
########################################################
## LIBS cannot be assigned using ':=' because OBJDIR
## is defined in Makefile.common that is incl
LIBS2 := \
    $(COMPILER_BE)/wavecode/$(OBJDIR)/libwavecode.a  \
    $(COMPILER_BE)/codegen/$(OBJDIR)/libcodegen.a  \
    $(COMPILER_BE)/schedule/$(OBJDIR)/libschedule.a \
    $(COMPILER_BE)/memmgr/$(OBJDIR)/libmemmgr.a \
    $(COMPILER_BE)/codegen/$(OBJDIR)/libcodegen.a  \
    $(COMPILER_BE)/wavecode/$(OBJDIR)/libwavecode.a  \
    $(COMPILER_BE)/nets/$(OBJDIR)/libnets.a \
    $(COMPILER_BE)/serialize/$(OBJDIR)/libserialize.a \
    $(COMPILER_BE)/layers/$(OBJDIR)/liblayers.a \
    $(COMPILER_BE)/events/$(OBJDIR)/libevents.a  \
    $(COMPILER_BE)/wave/$(OBJDIR)/libwave.a  \
    $(COMPILER_BE)/arch/$(OBJDIR)/libarch.a \
    $(COMPILER_BE)/utils/$(OBJDIR)/libutils.a \
    $(TCC_LIB)

########################################################
ADDITIONAL_CLEAN_FILES := $(MODULE).exe
########################################################

########################################################

exe : $(MODULE).exe
########################################################

$(MODULE).exe : $(OFILES) $(LIBS)
	@echo LD $@
	$(COMPILER) $(LDOPTS) $^ -o $@

