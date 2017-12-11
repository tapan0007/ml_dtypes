AR=ar
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

CINC = -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

CFLAGS=$(COPTS) $(CINC)

########################################################
MODULE=serialize
MODS = $(MODULE) utils 
LIBRARY = $(OBJDIR)/lib$(MODULE).a


SRCDIR = src
INCDIR = inc
OBJDIR = obj

########################################################
SRCS = serlayer

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(LIBRARY)

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCDIR)/%.hpp
	$(CPP) $(CFLAGS) -c -o $@ $<

########################################################
########################################################

$(LIBRARY) : $(OFILES)
	$(AR) rcs $@ $(OFILES)

clean :
	rm -f $(OFILES)


