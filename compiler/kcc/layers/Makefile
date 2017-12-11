AR=ar
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

CINC = -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

CFLAGS=$(COPTS) $(CINC)

########################################################
MODULE=layers
MODS = $(MODULE) utils nets serialize
LIBRARY = $(OBJDIR)/lib$(MODULE).a


SRCDIR = src
OBJDIR = obj

########################################################
SRCS = layer activlayer relulayer tanhlayer combinelayer subsamplelayer convlayer 

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(LIBRARY)

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

########################################################
########################################################

$(LIBRARY) : $(OFILES)
	$(AR) rcs $@ $(OFILES)

clean :
	rm -f $(OFILES)

