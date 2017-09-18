
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
LIBRARY = $(LIBDIR)/libtcc.a
SRCDIR = src
OBJDIR = obj
LIBDIR = libs
INCDIR = inc
SHARED_DIR = ../shared
SHARED_OBJDIR = $(SHARED_DIR)/build
SHARED_SRCDIR = $(SHARED_DIR)/src
SHARED_INCDIR_LIST = $(SHARED_OBJDIR) $(SHARED_SRCDIR)
SHARED_INCDIR =$(foreach d, $(SHARED_INCDIR_LIST), -I$d)
SCRIPTDIR = scripts

RM = rm -f
MAKE=make

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
SHARED_SOURCES  := $(wildcard $(SHARED_SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(INCDIR)/*.h) 
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
SHARED_OBJECTS  := $(SHARED_SOURCES:$(SHARED_SRCDIR)/%.cpp=$(OBJDIR)/%.o)

.PHONY: clean


all: $(LIBRARY) 

clean: 
	$(RM) $(OBJECTS) $(LIBRARY)

$(LIBRARY): $(OBJECTS) $(SHARED_OBJECTS)
	ar rcs $(LIBRARY) $(OBJECTS) $(SHARED_OBJECTS)

$(SHARED_OBJECTS): $(OBJDIR)/%.o : $(SHARED_SRCDIR)/%.cpp 
	 $(CXX) $(CFLAGS) $(CPPFLAGS) -I $(INCDIR) $(SHARED_INCDIR) -c $< -o $@

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCLUDES)  
	 $(CXX) $(CFLAGS) $(CPPFLAGS) -I $(INCDIR) $(SHARED_INCDIR) -c $< -o $@


