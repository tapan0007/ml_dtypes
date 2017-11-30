AR=ar
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14
CINC= -I./inc -I../utils/inc -I../nets/inc 

CFLAGS=$(COPTS) $(CINC)

########################################################
MODULE=arch
LIBRARY = $(OBJDIR)/lib$(MODULE).a


SRCDIR = src
OBJDIR = obj

########################################################
SRCS = activationeng arch pearray poolingeng psumbuffer statebuffer

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(LIBRARY)

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

########################################################
########################################################
$(OBJDIR)/arch.o : $(SRCDIR)/arch.cpp

$(OBJDIR)/activationeng.o : $(SRCDIR)/activationeng.cpp

$(OBJDIR)/pearray.o : $(SRCDIR)/pearray.cpp

$(OBJDIR)/poolingeng.o : $(SRCDIR)/poolingeng.cpp

$(OBJDIR)/psumbuffer.o : $(SRCDIR)/psumbuffer.cpp

$(OBJDIR)/statebuffer.o : $(SRCDIR)/statebuffer.cpp

$(LIBRARY) : $(OFILES)
	$(AR) rcs $@ $(OFILES)

