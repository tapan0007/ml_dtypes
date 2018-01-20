top : exe
########################################################
MODULE:=compiler
MODS := utils nets layers arch memmgr schedule serialize codegen
SRCS := compiler

include ../Makefile.common

########################################################
TCC_LIB := $(INKLING_PATH)/tcc/libs/libtcc.a

## LIBS cannot be assigned using ':=' because OBJDIR
## is defined in Makefile.common that is incl
LIBS := \
    ../nets/$(OBJDIR)/libnets.a \
    ../layers/$(OBJDIR)/liblayers.a \
    ../memmgr/$(OBJDIR)/libmemmgr.a \
    ../schedule/$(OBJDIR)/libschedule.a \
    ../serialize/$(OBJDIR)/libserialize.a \
    ../arch/$(OBJDIR)/libarch.a \
    ../utils/$(OBJDIR)/libutils.a \
    ../codegen/$(OBJDIR)/libcodegen.a  \
	$(TCC_LIB)

########################################################
ADDITIONAL_CLEAN_FILES := $(MODULE).exe
########################################################

########################################################

exe : $(MODULE).exe
########################################################

$(MODULE).exe : $(OFILES) $(LIBS)
	@echo LD $@
	@$(CPP) $(LDOPTS) $^ -o $@

