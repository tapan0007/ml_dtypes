
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
TARGET = test
SRCDIR = src
OBJDIR = obj
SCRIPTS = scripts

RM = rm  
MAKE=make

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h) 
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
ISA_JSON := $(SRCDIR)/isa.json 
ISA_H    := $(SRCDIR)/isa.h 
ISA_PDF  := isa.pdf

.PHONY: clean

all: $(TARGET) $(DOCS)

clean: 
	$(RM) -f $(TARGET) $(OBJECTS) $(SCRIPTS)/*.dtx $(SCRIPTS)/*log $(SCRIPTS)/*sty
	$(RM) -f $(SRCDIR)/isa.h

docs: $(ISA_PDF)  $(ISA_JSON)

$(ISA_H) : $(ISA_JSON) 
	$(SCRIPTS)/create_h.py $(SRCDIR)/isa.json $(SRCDIR)/isa.h

$(ISA_PDF) : $(ISA_JSON)
	cd $(SCRIPTS) && \
	rm bytefield.sty && \
	tex bytefield.ins && \
	./create_tex.py ../$(SRCDIR)/isa.json ../isa.tex && \
	pdflatex --output-directory ../ ../isa.tex

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

