CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
TARGET = verif_test
SRCDIR = .
OBJDIR = .
SIMDIR = ../sim
SHAREDIR = ../shared
SHARE_INCDIR := $(SHAREDIR)/inc
SIM_INCDIR := $(SIMDIR)/src
LIBDIR = $(SIMDIR)/libs
LINKFLAGS = -L $(LIBDIR) -lsim

RM = rm -f
MAKE=make

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h) 
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

.PHONY: clean

all: $(TARGET) 

clean: 
	$(RM) $(TARGET) $(OBJECTS) 


$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCLUDES)  
	 $(CXX) -DVERIF -I$(SHARE_INCDIR) -I$(SIM_INCDIR) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS) 
	$(CXX) $^  -I$(SHARE_INCDIR) -I$(SIM_INCDIR) $(LINKFLAGS) -o $@ 


