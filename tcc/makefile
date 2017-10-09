
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
LIBRARY = $(LIBDIR)/libtcc.a
SRCDIR = src
OBJDIR = obj
LIBDIR = libs
INCDIR = inc
SHARED_DIR = ../shared
SHARED_INCDIR = $(SHARED_DIR)/inc
SCRIPTDIR = scripts

RM = rm -f
MAKE=make

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(INCDIR)/*.h) 
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

.PHONY: clean


all: $(LIBRARY) 

clean: 
	$(RM) $(OBJECTS) $(LIBRARY)

$(LIBRARY): $(OBJECTS) $(SHARED_OBJECTS)
	ar rcs $(LIBRARY) $(OBJECTS) $(SHARED_OBJECTS)


$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCLUDES)  
	 $(CXX) $(CFLAGS) $(CPPFLAGS) -I $(INCDIR) -I$(SHARED_INCDIR) -c $< -o $@


