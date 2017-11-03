CXX= g++
CXXFLAGS = -I. -std=c++11 -O1
CFLAGS = -W -Wall -ansi -ggdb -g -Wpacked -Wno-missing-field-initializers
AR=ar
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

$(LIBRARY): $(NONMAIN_OBJECTS)
	@echo AR $(notdir $@)
	@$(AR) rcs $(LIBRARY) $(NONMAIN_OBJECTS)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCLUDES)
	@echo CXX $(notdir $@)
	@$(CXX) $(CFLAGS) $(CXXFLAGS) -I $(SHARE_INCDIR) -c $< -o $@


$(TARGET): $(OBJECTS)
	@echo LINK $(notdir $@)
	@$(CXX) $^  $(LINKFLAGS) -o $@


