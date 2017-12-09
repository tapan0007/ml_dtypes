AR=ar
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

MODS = utils nets layers arch memmgr schedule
CINC = -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

LIBS = ../utils/$(OBJDIR)/libutils.a ../layers/$(OBJDIR)/liblayers.a

CFLAGS=$(COPTS) $(CINC)

LDFLAGS = $(COPTS) $(LIBS)

########################################################
MODULE=compiler

SRCDIR = src
OBJDIR = obj

########################################################
SRCS = compiler

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(OBJDIR)/$(MODULE).exe

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

########################################################
########################################################
$(OBJDIR)/$(MODULE).exe : $(OFILES)
	$(CPP) $(COPTS) $^ $(LIBS) -o $@

