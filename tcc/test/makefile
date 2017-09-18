
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
LIBFLAGS = -ltcc
LIBRARY = $(LIBDIR)/compiler_lib.a
SRCDIR = src
OBJDIR = obj
LIBDIR = ../libs
SHAREDIR = ../../shared
INCDIRS = $(SHAREDIR)/src $(SHAREDIR)/build src ../inc
INCLUDES = $(foreach d, $(INCDIRS), -I$d)
TARGET = test_convolve

RM = rm -f
MAKE=make

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

.PHONY: clean


all: $(TARGET) 

clean: 
	$(RM) $(OBJECTS) 

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp 
	 echo $(INCLUDES); \
	 $(CXX) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CXX) $^ $(INCDIR)  $(LIBFLAGS) -L$(LIBDIR) -o $@

