#CPP = clang++
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14
CINC= -I./inc

CFLAGS=$(COPTS) $(CINC)

########################################################
SRCDIR = src
OBJDIR = obj

########################################################
$(OBJDIR)/fmapdesc.o : $(SRCDIR)/fmapdesc.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

