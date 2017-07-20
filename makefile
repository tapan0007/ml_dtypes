
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -pedantic
TARGET = test
RM = rm
SRCDIR = src
OBJDIR = obj

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

.PHONY: all clean

all: $(TARGET)

clean: 
	$(RM) $(TARGET) $(OBJECTS)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCLUDES)
	 @$(CXX) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CXX) $^ -o $@

