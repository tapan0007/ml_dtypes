
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -ggdb -g -Wno-missing-field-initializers
TARGET = objdump
SRC = objdump.cpp
SRCDIR = .
OBJDIR = .
SHAREDIR = ../shared
SHARE_JSONDIR = $(SHAREDIR)/json
SHARE_INCDIR = $(SHAREDIR)/inc

RM = rm -f
MAKE=make


.PHONY: clean

all: $(TARGET) 

clean: 
	$(RM) $(TARGET)  


$(TARGET): $(SRC)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(SRC) -I$(SHARE_INCDIR) -o $@


