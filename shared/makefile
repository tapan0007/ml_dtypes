
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
SRCDIR = src
OBJDIR = build
DOCDIR = $(OBJDIR)
SCRIPTDIR = scripts

RM = rm -f
MAKE=make

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h) 
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
ISA_JSON := $(SRCDIR)/isa.json 
ISA_H    := $(SRCDIR)/isa.h 
ISA_PDF  := isa.pdf

.PHONY: clean

all: $(OBJECTS) $(ISA_PDF) $(ISA_H)

clean: 
	$(RM) $(OBJECTS) $(SCRIPTDIR)/*log $(SCRIPTDIR)/*sty 
	$(RM) $(SRCDIR)/isa.h isa.aux isa.log isa.tex isa.pdf

docs: $(ISA_PDF)

$(ISA_H) : $(ISA_JSON)  $(SCRIPTDIR)/create_h.py
	$(SCRIPTDIR)/create_h.py $(SRCDIR)/isa.json $(OBJDIR)/isa.h

$(ISA_PDF) : $(ISA_JSON)
	cd $(SCRIPTDIR) && \
	$(RM)  bytefield.sty && \
	tex bytefield.ins && \
	./create_tex.py ../$(SRCDIR)/isa.json ../$(OBJDIR)/isa.tex && \
	pdflatex --output-directory ../$(DOCDIR) ../$(OBJDIR)/isa.tex

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCLUDES)  $(ISA_H)
	 @$(CXX) $(CFLAGS) $(CPPFLAGS) $(INCDIR) -c $< -o $@



