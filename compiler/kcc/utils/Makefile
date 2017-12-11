AR=ar
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

CINC = -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

CFLAGS=$(COPTS) $(CINC)

########################################################
MODULE=utils
MODS = $(MODULE) nets layers serialize
LIBRARY = $(OBJDIR)/lib$(MODULE).a

SRCDIR = src
OBJDIR = obj

########################################################
SRCS = fmapdesc consts debug
##printers

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(LIBRARY)

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

########################################################

$(OBJDIR)/fmapdesc.o : $(SRCDIR)/fmapdesc.cpp
$(OBJDIR)/consts.o : $(SRCDIR)/consts.cpp
$(OBJDIR)/printers.o : $(SRCDIR)/printers.cpp
$(OBJDIR)/debug.o : $(SRCDIR)/debug.cpp

$(LIBRARY) : $(OFILES)
	$(AR) rcs $@ $(OFILES)

clean :
	rm -f $(OFILES)

