
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
ISA_JSON := $(SRCDIR)/isa.json 
ISA_H    := $(SRCDIR)/isa.h 
ISA_PDF  := isa.pdf

.PHONY: all clean

all: $(TARGET) $(ISA_PDF)

clean: 
	$(RM) $(TARGET) $(OBJECTS)

$(ISA_H) : $(ISA_JSON)
	$(SCRIPTS)/create_h.py $(SRCDIR)/isa.json $(SRCDIR)/isa.h

$(ISA_PDF) : $(ISA_JSON)
	$(SCRIPTS)/create_doc.py $(SRCDIR)/isa.json isa.pdf

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCLUDES)  $(ISA_H)
	 @$(CXX) $(CFLAGS) $(CPPFLAGS) $(INCDIR) -c $< -o $@

$(TARGET): $(OBJECTS) 
	$(CXX) $^ $(INCDIR) $(LIBDIR) $(LIB)  $(LINKFLAGS) -o $@

$(CNPYLIB):
	cd $(CNPYBUILD); \
	cmake -DCMAKE_INSTALL_PREFIX=$(CNPYINSTALL) $(CNPYSRC); \
	make; \
	make install ; \
	cd ../

