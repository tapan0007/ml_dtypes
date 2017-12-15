AR=ar
CPP = g++

COPTS= -W -Wall -Werror -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

MODULE=utils
MODS = $(MODULE) nets layers serialize schedule

INK_INC_DIR = shared tcc
INK_INC = $(addprefix -I$(INKLING_PATH)/,$(addsuffix /inc,$(INK_INC_DIR)))

CINC = $(INK_INC) -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

CFLAGS=$(COPTS) $(CINC)

########################################################
LIBRARY = $(OBJDIR)/lib$(MODULE).a

SRCDIR = src
OBJDIR = obj
DEPDIR = dep

########################################################
SRCS = fmapdesc debug datatype
##printers
## consts

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

$(OBJDIR)/fmapdesc.o : $(SRCDIR)/fmapdesc.cpp
$(OBJDIR)/datatype.o : $(SRCDIR)/datatype.cpp
$(OBJDIR)/printers.o : $(SRCDIR)/printers.cpp
$(OBJDIR)/debug.o : $(SRCDIR)/debug.cpp

$(LIBRARY) : $(OFILES)
	@echo AR $@
	@$(AR) rcs $@ $(OFILES)

clean :
	@echo CLEAN $(MODULE)
	@rm -f $(OFILES)

