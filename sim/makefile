
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
TARGET = sim
SRCDIR = src
OBJDIR = obj
LIBDIR = libs
LIBRARY = $(LIBDIR)/libsim.a
SHAREDIR = ../shared
SHARE_SRCDIR = $(SHAREDIR)/src
SHARE_INCDIR_LIST = $(SHAREDIR)/src $(SHAREDIR)/build
SHARE_INCDIR = $(foreach d, $(SHARE_INCDIR_LIST), -I$d)

RM = rm -f
MAKE=make

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
SHARE_SOURCES  := $(wildcard $(SHARE_SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h) 
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
NONMAIN_OBJECTS  := $(filter-out $(OBJDIR)/main.o, $(OBJECTS))
SHARE_OBJECTS  := $(SHARE_SOURCES:$(SHARE_SRCDIR)/%.cpp=$(OBJDIR)/%.o)

.PHONY: clean

all: $(TARGET) $(LIBRARY)

libs: $(LIBRARY)

clean: 
	$(RM) $(TARGET) $(OBJECTS)   $(LIBRARY)

$(LIBRARY): 
	 ar rcs $(LIBRARY) $(NONMAIN_OBJECTS) $(SHARE_OBJECTS)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCLUDES)  
	 @$(CXX) $(CFLAGS) $(CPPFLAGS) -I $(SHARE_SRCDIR) $(SHARE_INCDIR) -c $< -o $@

$(SHARE_OBJECTS): $(OBJDIR)/%.o : $(SHARE_SRCDIR)/%.cpp 
	 @$(CXX) $(CFLAGS) $(CPPFLAGS) -I $(SHARE_SRCDIR) $(SHARE_INCDIR) -c $< -o $@

$(TARGET): $(OBJECTS)  $(SHARE_OBJECTS)
	$(CXX) $^  $(LINKFLAGS) -o $@


