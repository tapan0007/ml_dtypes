top : exe
########################################################
MODULE:=compiler
#MODS := utils nets layers arch memmgr schedule serialize codegen wave
MODS :=  \
	wavecode	\
	nets		\
	serialize	\
	events		\
	wave		\
	arch		\
	kelf 		\
	dma         \
	utils		\
	compisa


SRCS := compiler

include ../Makefile.common


########################################################

LIBS1 := $(foreach L,$(MODS),$(COMPILER_BE)/$(L)/$(OBJDIR)/lib$(L).a)
LIBS := $(LIBS1) 
########################################################

########################################################
ADDITIONAL_CLEAN_FILES := $(MODULE).exe
########################################################

########################################################

exe : $(MODULE).exe
########################################################

$(MODULE).exe : $(OFILES) $(LIBS)
	@echo LD $@
	@$(COMPILER) $(LDFLAGS) $^ -o $@

