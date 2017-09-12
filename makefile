
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
TARGET = test
SRCDIR = src
OBJDIR = obj
SCRIPTS = scripts

RM = rm  -f
MAKE=make

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h) 
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
ISA      := $(SRCDIR)/isa.h

.PHONY: all clean

all: $(TARGET)

clean: 
	$(RM) $(TARGET) $(OBJECTS)

$(ISA) : $(SRCDIR)/isa.json
	$(SCRIPTS)/create_h.py $(SRCDIR)/isa.json $(SRCDIR)/isa.h

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCLUDES) $(ISA)
	 @$(CXX) $(CFLAGS) $(CPPFLAGS) $(INCDIR) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CXX) $^ $(INCDIR) $(LIBDIR) $(LIB)  $(LINKFLAGS) -o $@

$(CNPYLIB):
	cd $(CNPYBUILD); \
	cmake -DCMAKE_INSTALL_PREFIX=$(CNPYINSTALL) $(CNPYSRC); \
	make; \
	make install ; \
	cd ../

