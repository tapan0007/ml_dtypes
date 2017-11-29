CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14
CINC= -I./inc -I../utils/inc -I../nets/inc 

CFLAGS=$(COPTS) $(CINC)

########################################################
SRCDIR = src
OBJDIR = obj

SRCS = layer activlayer relulayer combinelayer
CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))
########################################################
all :  $(OFILES)

########################################################
$(OBJDIR)/activlayer.o : $(SRCDIR)/activlayer.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/relulayer.o : $(SRCDIR)/relulayer.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/layer.o : $(SRCDIR)/layer.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/combinelayer.o : $(SRCDIR)/combinelayer.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

