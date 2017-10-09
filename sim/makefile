
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
TARGET = sim
SRCDIR = src
OBJDIR = obj
LIBDIR = libs
LIBRARY = $(LIBDIR)/libsim.a
SHAREDIR = ../shared
SHARE_INCDIR = $(SHAREDIR)/inc

RM = rm -f
MAKE=make

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h) 
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
NONMAIN_OBJECTS  := $(filter-out $(OBJDIR)/main.o, $(OBJECTS))

.PHONY: clean

all: $(TARGET) $(LIBRARY)

libs: $(LIBRARY)

clean: 
	$(RM) $(TARGET) $(OBJECTS)   $(LIBRARY)

$(LIBRARY): 
	 ar rcs $(LIBRARY) $(NONMAIN_OBJECTS)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCLUDES)  
	 @$(CXX) $(CFLAGS) $(CPPFLAGS) -I $(SHARE_INCDIR) -c $< -o $@


$(TARGET): $(OBJECTS)  
	$(CXX) $^  $(LINKFLAGS) -o $@


