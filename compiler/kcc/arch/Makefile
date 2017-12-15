AR=ar
CPP = g++

COPTS= -W -Wall -Werror -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

MODS= utils nets
CINC = -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

CFLAGS=$(COPTS) $(CINC)

########################################################
MODULE=arch
LIBRARY = $(OBJDIR)/lib$(MODULE).a


SRCDIR = src
OBJDIR = obj
DEPDIR = dep

########################################################
SRCS = activationeng arch pearray poolingeng psumbuffer statebuffer

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
########################################################
alldeps:
	@test -d $(DEPDIR) || mkdir $(DEPDIR); \
	for f in $(SRCS); do \
	  $(CPP) $(CFLAGS) -MM -MT $(OBJDIR)/$$f.o $(SRCDIR)/$$f.cpp; \
	done > $(DEPDIR)/all.dep

########################################################
$(DEPDIR)/%.dep : $(SRCDIR)/%.cpp
	@echo DEP $<
	@$(CPP) $(CFLAGS) -MM -MT $(OBJDIR)/$*.o -o $@ $<


$(LIBRARY) : $(OFILES)
	@echo AR $@
	@$(AR) rcs $@ $(OFILES)

clean :
	@echo CLEAN $(MODULE)
	@rm -f $(OFILES)

