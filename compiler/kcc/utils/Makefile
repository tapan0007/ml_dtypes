#CPP = g++
CPP = clang++

COPTS= -std=c++11  -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers
CINC= -I./inc

CFLAGS=$(COPTS) $(CINC)

########################################################
SRCDIR = src
OBJDIR = obj

########################################################
$(OBJDIR)/fmapdesc.o : $(SRCDIR)/fmapdesc.cpp
	$(CPP) $(CFLAGS) -o $@ $<

