AR=ar
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

MODS = utils nets layers
CINC = -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

CFLAGS=$(COPTS) $(CINC)

########################################################
MODULE=utils
LIBRARY = $(OBJDIR)/lib$(MODULE).a

SRCDIR = src
OBJDIR = obj

########################################################
SRCS = fmapdesc consts printers

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

$(LIBRARY) : $(OFILES)
	$(AR) rcs $@ $(OFILES)


