
CXX?= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
LIBFLAGS = -ltcc
LIBRARY = $(LIBDIR)/compiler_lib.a
SRCDIR = src
OBJDIR = obj
LIBDIR = ../../libs
ISA_SHAREDIR = ../../../shared
SHAREDIR = ../shared
INCDIRS = $(ISA_SHAREDIR)/inc $(SHAREDIR) src ../../inc 
INCLUDES = $(foreach d, $(INCDIRS), -I$d)

RM = rm -f
MAKE=make

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
SRC_OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
SHARED_SOURCES  := $(wildcard $(SHAREDIR)/*.cpp)
SHARED_OBJECTS  := $(SHARED_SOURCES:$(SHAREDIR)/%.cpp=$(OBJDIR)/%.o)
OBJECTS := $(SRC_OBJECTS) $(SHARED_OBJECTS)

.PHONY: clean


all: $(TARGET) 

clean: 
	$(RM) $(OBJECTS) $(TARGET)

$(SRC_OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp 
	@echo CXX $(notdir $@)
	@$(CXX) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(SHARED_OBJECTS): $(OBJDIR)/%.o : $(SHAREDIR)/%.cpp 
	@echo CXX $(notdir $@)
	@$(CXX) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJECTS) $(LIBDIR)/libtcc.a
	@echo CXX $(notdir $@)
	@$(CXX) $^ $(INCDIR) -L$(LIBDIR)  $(LIBFLAGS) -o $@ 

