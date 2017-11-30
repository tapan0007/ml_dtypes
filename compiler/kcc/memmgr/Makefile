AR=ar
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

MODS = utils nets arch layers
CINC = -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))


CFLAGS=$(COPTS) $(CINC)

########################################################
MODULE=memmgr
LIBRARY = $(OBJDIR)/lib$(MODULE).a


SRCDIR = src
OBJDIR = obj

########################################################
SRCS = statebufmgr

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(LIBRARY)

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

########################################################
########################################################
$(OBJDIR)/statebufmgr.o : $(SRCDIR)/statebufmgr.cpp


$(LIBRARY) : $(OFILES)
	$(AR) rcs $@ $(OFILES)


