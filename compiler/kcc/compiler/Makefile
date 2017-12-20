
########################################################
TCC_LIB = $(INKLING_PATH)/tcc/libs/libtcc.a

LIBS = \
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
MODULE=compiler
MODS = utils nets layers arch memmgr schedule serialize codegen

########################################################
SRCS = compiler

top : $(MODULE).exe
########################################################
include ../Makefile.common

$(MODULE).exe : $(OFILES) $(LIBS)
	@echo LD $@
	@$(CPP) $(LDOPTS) $^ -o $@

