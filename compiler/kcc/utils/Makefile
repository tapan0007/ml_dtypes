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
all : $(LIBRARY)

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

########################################################
$(OBJDIR)/fmapdesc.o : $(SRCDIR)/fmapdesc.cpp

$(LIBRARY) : $(OFILES)
	$(AR) rcs $@ $(OFILES)


