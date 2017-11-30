AR=ar
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14
CINC= -I./inc -I../utils/inc -I../nets/inc 

CFLAGS=$(COPTS) $(CINC)

########################################################
MODULE=layers
LIBRARY = $(OBJDIR)/lib$(MODULE).a


SRCDIR = src
OBJDIR = obj

########################################################
SRCS = layer activlayer relulayer combinelayer subsamplelayer convlayer 

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(LIBRARY)

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

########################################################
########################################################
$(OBJDIR)/activlayer.o : $(SRCDIR)/activlayer.cpp

$(OBJDIR)/relulayer.o : $(SRCDIR)/relulayer.cpp

$(OBJDIR)/layer.o : $(SRCDIR)/layer.cpp

$(OBJDIR)/combinelayer.o : $(SRCDIR)/combinelayer.cpp

$(OBJDIR)/convlayer.o : $(SRCDIR)/convlayer.cpp

$(LIBRARY) : $(OFILES)
	$(AR) rcs $@ $(OFILES)


