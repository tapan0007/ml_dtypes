CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14
CINC= -I./inc -I../utils/inc -I../nets/inc

CFLAGS=$(COPTS) $(CINC)

########################################################
SRCDIR = src
OBJDIR = obj

########################################################
$(OBJDIR)/layer.o : $(SRCDIR)/layer.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

