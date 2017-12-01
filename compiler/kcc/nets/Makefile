AR=ar
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

MODS = utils nets arch layers
CINC = -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

CFLAGS=$(COPTS) $(CINC)

########################################################
MODULE=nets
LIBRARY = $(OBJDIR)/lib$(MODULE).a


SRCDIR = src
OBJDIR = obj

########################################################
SRCS = network

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


