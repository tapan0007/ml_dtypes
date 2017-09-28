CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
TARGET = verif_test
SRCDIR = .
OBJDIR = .
SIMDIR = ../sim
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
	 $(CXX) -DVERIF $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS) 
	$(CXX) $^  $(LINKFLAGS) -o $@ 


