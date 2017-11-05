FLAGS= -W -Wall -Werror -ggdb -g 
INC_FLAGS = -IInkling/shared/inc -IInkling/tcc/inc
#-IInkling/tcc/test/convolve/src

CFLAGS = $(FLAGS) -I. $(INC_FLAGS) -Wno-missing-field-initializers

CPPFLAGS = $(CFLAGS) -std=c++11
LFLAGS = $(FLAGS)
#################################################
INKLING=./Inkling
LIBDIR1 = $(INKLING)/tcc/libs
LIBDIR_FLAGS= -L$(LIBDIR1)

TCC=tcc
LIB_FLAGS = -l$(TCC)
TCCAR = $(LIBDIR1)/lib$(TCC).a
NET=trivnet
CPP=$(NET).cpp
EXE=$(NET)-exe
OBJ=$(NET).o
TPB=$(NET).tpb
ASM=$(NET).asm
SIMRES=$(NET).simres

GOLDEN=golden
FINAL=$(NET)-out

#################################################
top : all
#################################################
$(TPB) : $(EXE)
	./$(EXE) $(TPB)

$(ASM) : $(TPB)
	./objdump $(TPB) > $@

simulate : $(TPB)
	./sim $(TPB) > tee $(SIMRES)

$(EXE) : $(OBJ) $(TCCAR)
	g++ $(LFLAGS) -o $@ $(OBJ) $(LIBDIR_FLAGS) $(LIB_FLAGS)

$(CPP) :
	cd .. ; make $(NET)

$(OBJ) : $(CPP)
	g++ $(CPPFLAGS) -c $(CPP) 2>&1 | tee LOG

npy2txt :
	for fmt in int16 float16; do                              \
	    for file in $(GOLDEN) $(FINAL); do                    \
	        npy=$$file.npy;  txt=$$file-$$fmt.txt;            \
	        dumpnpy --$$fmt $$npy > $$txt;                    \
	        echo $$txt;                                       \
	        cat $$txt;                                        \
	    done;                                                 \
	done

all : $(EXE) $(ASM) simulate npy2txt

clean :
	rm -f $(EXE) $(OBJ) *-out.npy simulation.log $(TPB) $(ASM) $(SIMRES) *.txt

link :
	ln -s $$HOME/kaena/CodeCommit/Inkling .
	ln -s ../nets/trivnet .
#################################################

