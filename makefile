
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -pedantic -ggdb
TARGET = test
RM = rm  -f
SRCDIR = src
OBJDIR = obj
CNPYLIB = cnpy-build/lib/libcnpy.a
MAKE=make

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

.PHONY: all clean

all: $(TARGET) $(CNPYLIB)

clean: 
	$(RM) $(TARGET) $(OBJECTS) && \
	$(MAKE) -C cnpy/cnpy-build clean;  

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCLUDES)
	 @$(CXX) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CXX) $^ -o $@

$(CNPYLIB):
	echo "hi hi hi hi";
	cd cnpy/cnpy-build; \
	cmake -DCMAKE_INSTALL_PREFIX=../cnpy-install ../cnpy-master; \
	make; \
	make install ; \
	cd ../

