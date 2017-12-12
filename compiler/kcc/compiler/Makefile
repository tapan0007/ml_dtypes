AR=ar
CPP = g++

COPTS= -W -Wall -Werror -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14
LDOPT = -W -Wall -Werror -pedantic -g -ggdb -Wno-missing-field-initializers -std=c++14

INK_INC_DIR = shared tcc
INK_INC = $(addprefix -I$(INKLING_PATH)/,$(addsuffix /inc,$(INK_INC_DIR)))
TCC_LIB = $(INKLING_PATH)/tcc/libs/libtcc.a

CINC = $(INK_INC) -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

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

CFLAGS=$(COPTS) $(CINC)

LDFLAGS = $(COPTS) $(LIBS)

########################################################
MODULE=compiler
MODS = utils nets layers arch memmgr schedule serialize codegen

SRCDIR = src
OBJDIR = obj

########################################################
SRCS = compiler

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(MODULE).exe

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

########################################################
########################################################
$(MODULE).exe : $(OFILES)
	@echo LD $@
	@$(CPP) $(LDOPTS) $^ $(LIBS) -o $@

clean :
	@echo CLEAN $(MODULE)
	@rm -f $(OFILES)

i : $(MODULE).exe
	./$< --json $@.json

