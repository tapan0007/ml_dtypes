
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
TARGET = objdump
GEN_SRC = objdump.cpp
SRCDIR = .
OBJDIR = .
SHAREDIR = ../shared
SHARE_SRCDIR = $(SHAREDIR)/src
SHARE_INCDIR_LIST = $(SHAREDIR)/src $(SHAREDIR)/build
SHARE_INCDIR = $(foreach d, $(SHARE_INCDIR_LIST), -I$d)

RM = rm -f
MAKE=make


.PHONY: clean

all: $(TARGET) 

clean: 
	$(RM) $(TARGET)  $(GEN_SRC)


$(GEN_SRC) : $(SHARE_SRCDIR)/isa.json objdump.py
	./objdump.py $(SHARE_SRCDIR)/isa.json $(GEN_SRC)

$(TARGET): $(GEN_SRC)
	$(CXX) $(CFLAGS) $(GEN_SRC) $(SHARE_INCDIR) -o $@


