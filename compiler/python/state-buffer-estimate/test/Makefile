FLAGS= -W -Wall -Werror -ggdb -g 
CFLAGS = $(FLAGS) -I. -Wno-missing-field-initializers
CPPFLAGS = $(CFLAGS) -std=c++11
LFLAGS = $(FLAGS)

INKLING = ./Inkling

LIBDIR1 = $(INKLING)/tcc/libs
LIBDIR_FLAGS= -L$(LIBDIR1)

TCC=tcc
LIB_FLAGS = -l$(TCC)
TCCAR = $(LIBDIR1)/lib$(TCC).a

run : code
	./code o

code : code.o $(TCCAR)
	g++ $(LFLAGS) -o code code.o $(LIBDIR_FLAGS) $(LIB_FLAGS)

code.cpp :
	cd .. ; make resnet

code.o : code.cpp
	g++ $(CPPFLAGS) -c code.cpp 2>&1 | tee LOG


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

