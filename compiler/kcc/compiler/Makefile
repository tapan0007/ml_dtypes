AR=ar
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

CINC = -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

LIBS = \
    ../memmgr/$(OBJDIR)/libmemmgr.a \
    ../schedule/$(OBJDIR)/libschedule.a \
    ../serialize/$(OBJDIR)/libserialize.a \
    ../nets/$(OBJDIR)/libnets.a \
    ../layers/$(OBJDIR)/liblayers.a \
    ../arch/$(OBJDIR)/libarch.a \
    ../utils/$(OBJDIR)/libutils.a 

CFLAGS=$(COPTS) $(CINC)

LDFLAGS = $(COPTS) $(LIBS)

########################################################
MODULE=compiler
MODS = utils nets layers arch memmgr schedule serialize

SRCDIR = src
OBJDIR = obj

########################################################
SRCS = compiler

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(MODULE).exe

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

########################################################
########################################################
$(MODULE).exe : $(OFILES)
	$(CPP) $(COPTS) $^ $(LIBS) -o $@

