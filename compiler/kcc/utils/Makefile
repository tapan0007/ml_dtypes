AR=ar
CPP = g++

COPTS= -W -Wall -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

MODULE=utils
MODS = $(MODULE) nets layers serialize schedule

INK_INC_DIR = shared tcc
INK_INC = $(addprefix -I$(INKLING_PATH)/,$(addsuffix /inc,$(INK_INC_DIR)))

CINC = $(INK_INC) -I./inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

CFLAGS=$(COPTS) $(CINC)

########################################################
LIBRARY = $(OBJDIR)/lib$(MODULE).a

SRCDIR = src
OBJDIR = obj

########################################################
SRCS = fmapdesc debug datatype
##printers
## consts

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(LIBRARY)

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

########################################################

$(OBJDIR)/fmapdesc.o : $(SRCDIR)/fmapdesc.cpp
$(OBJDIR)/datatype.o : $(SRCDIR)/datatype.cpp
$(OBJDIR)/printers.o : $(SRCDIR)/printers.cpp
$(OBJDIR)/debug.o : $(SRCDIR)/debug.cpp

$(LIBRARY) : $(OFILES)
	$(AR) rcs $@ $(OFILES)

clean :
	rm -f $(OFILES)

