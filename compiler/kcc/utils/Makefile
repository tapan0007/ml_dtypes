AR=ar
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14
CINC= -I./inc

CFLAGS=$(COPTS) $(CINC)

########################################################
MODULE=utils
LIBRARY = $(OBJDIR)/lib$(MODULE).a

SRCDIR = src
OBJDIR = obj

########################################################
SRCS = fmapdesc

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(OBJDIR)/$(MODULE).lib

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

########################################################
$(OBJDIR)/fmapdesc.o : $(SRCDIR)/fmapdesc.cpp

$(OBJDIR)/$(MODULE).lib : $(OFILES)
	$(AR) rcs $@ $(OFILES)


