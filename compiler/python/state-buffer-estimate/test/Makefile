FLAGS= -W -Wall -ggdb -g 
CFLAGS = $(FLAGS) -I. -Wno-missing-field-initializers
CPPFLAGS = $(CFLAGS) -std=c++11
LFLAGS = $(FLAGS)

INKLING = ./Inkling

LIBDIRS = -L$(INKLING)/tcc/libs
LIBS = -ltcc

run : code
	./code o

code : code.o
	g++ $(LFLAGS) -o code code.o $(LIBDIRS) $(LIBS)

code.cpp :
	cd .. ; make resnet

code.o : code.cpp
	g++ $(CPPFLAGS) -c code.cpp 2>LOG


clean :
	rm code code.o

link :
	ln -s Inkling/tcc/test/convolve/src/cnpy.h
	ln -s Inkling/shared/inc/isa_common.h
	ln -s Inkling/tcc/inc/tcc.h
	ln -s Inkling/shared/inc/tpb_isa.h
	ln -s Inkling/shared/inc/tpb_isa_ldweights.h
	ln -s Inkling/shared/inc/tpb_isa_matmul.h
	ln -s Inkling/shared/inc/tpb_isa_pool.h
	ln -s Inkling/shared/inc/tpb_isa_simrdfilter.h
	ln -s Inkling/shared/inc/tpb_isa_simrdifmap.h
	ln -s Inkling/shared/inc/tpb_isa_simwrofmap.h
	ln -s Inkling/shared/inc/uarch_cfg.h

