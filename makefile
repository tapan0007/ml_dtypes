
CXX= g++
CPPFLAGS = -I. -std=c++11
CFLAGS = -W -Wall -ansi -pedantic -ggdb
TARGET = test
SRCDIR = src
OBJDIR = obj

CNPYSRC = $(PWD)/cnpy/cnpy-master/
CNPYBUILD = $(PWD)/cnpy/cnpy-build/
CNPYINSTALL = $(PWD)cnpy/cnpy-install/
CNPYLIB = $(CNPYINSTALL)/lib/libcnpy.a
INCDIR = -I $(PWD)/cnpy/cnpy-install/include/
LIBDIR = #-L $(PWD)/cnpy/cnpy-install/lib/
#LIB    = -lcnpy
LIB    = $(PWD)/cnpy/cnpy-install/lib/libcnpy.a
RM = rm  -f
MAKE=make

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h) 
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

.PHONY: all clean

all: $(CNPYLIB) $(TARGET)

clean: 
	$(RM) $(TARGET) $(OBJECTS) && \
	$(RM) -rf $(CNPYINSTALL)/* && \
	$(MAKE) -C $(CNPYBUILD) clean;  

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp $(INCLUDES)
	 @$(CXX) $(CFLAGS) $(CPPFLAGS) $(INCDIR) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CXX) $^ $(INCDIR) $(LIBDIR) $(LIB)  $(LINKFLAGS) -o $@

$(CNPYLIB):
	cd $(CNPYBUILD); \
	cmake -DCMAKE_INSTALL_PREFIX=$(CNPYINSTALL) $(CNPYSRC); \
	make; \
	make install ; \
	cd ../

#-L/path/to/install/dir -lcnpy

