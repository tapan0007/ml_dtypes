AR=ar
CPP = g++

COPTS= -W -Wall -Werror -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

INK_INC_DIR = shared tcc
INK_INC = $(addprefix -I$(INKLING_PATH)/,$(addsuffix /inc,$(INK_INC_DIR)))

CINC = $(INK_INC) -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

CFLAGS=$(COPTS) $(CINC)

########################################################
MODULE=nets
MODS = $(MODULE) utils arch layers serialize
LIBRARY = $(OBJDIR)/lib$(MODULE).a


SRCDIR = src
OBJDIR = obj
DEPDIR = dep

########################################################
SRCS = network

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(LIBRARY)

-include $(DEPDIR)/all.dep
########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@echo CC $<
	@$(CPP) $(CFLAGS) -c -o $@ $<

########################################################
alldeps:
	@test -d $(DEPDIR) || mkdir $(DEPDIR); \
	for f in $(SRCS); do \
	  $(CPP) $(CFLAGS) -MM -MT $(OBJDIR)/$$f.o $(SRCDIR)/$$f.cpp; \
	done > $(DEPDIR)/all.dep

########################################################
########################################################

$(LIBRARY) : $(OFILES)
	@echo AR $@
	@$(AR) rcs $@ $(OFILES)

clean :
	@echo CLEAN $(MODULE)
	@rm -f $(OFILES)


