AR=ar
CPP = g++

COPTS= -W -Wall -Werror -pedantic -ansi -g -ggdb -Wno-missing-field-initializers -std=c++14

INK_INC_DIR = shared tcc
INK_INC = $(addprefix -I$(INKLING_PATH)/,$(addsuffix /inc,$(INK_INC_DIR)))

CINC =  $(INK_INC) -I../inc  $(addsuffix /inc,$(addprefix -I../,$(MODS)))

CFLAGS=$(COPTS) $(CINC)

########################################################
MODULE=layers
MODS = $(MODULE) utils nets serialize
LIBRARY = $(OBJDIR)/lib$(MODULE).a


SRCDIR = src
OBJDIR = obj

########################################################
SRCS := layer 
SRCS := $(SRCS) activlayer relulayer tanhlayer 
SRCS := $(SRCS) combinelayer subsamplelayer convlayer 
SRCS := $(SRCS) poollayer maxpoollayer avgpoollayer

CFILES = $(addprefix $(SRCDIR)/,$(addsuffix .c,$(SRCS)))
OFILES = $(addprefix $(OBJDIR)/,$(addsuffix .o,$(SRCS)))

########################################################
all : $(LIBRARY)

########################################################
$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	@echo CC $<
	@$(CPP) $(CFLAGS) -c -o $@ $<

########################################################
########################################################

$(LIBRARY) : $(OFILES)
	@echo AR $@
	@$(AR) rcs $@ $(OFILES)

clean :
	@echo CLEAN $(MODULE)
	@rm -f $(OFILES)

