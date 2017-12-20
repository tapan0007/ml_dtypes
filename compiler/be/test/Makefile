FLAGS:= -W -Wall -Werror -ggdb -g 
INC_FLAGS := -I$(INKLING)/shared/inc -I$(INKLING)/tcc/inc
#-I$(INKLING)/tcc/test/convolve/src

CFLAGS := $(FLAGS) -I. $(INC_FLAGS) -Wno-missing-field-initializers

CPPFLAGS := $(CFLAGS) -std=c++11
LFLAGS := $(FLAGS)
#################################################
LIBDIR1 := $(INKLING)/tcc/libs
LIBDIR_FLAGS:= -L$(LIBDIR1)
RESULTS:=./results

TCC:=tcc
LIB_FLAGS := -l$(TCC)
TCCAR := $(LIBDIR1)/lib$(TCC).a
NET:=trivnet
CPP:=$(NET).cpp
EXE:=$(NET)-exe
OBJ:=$(NET).o
TPB:=$(RESULTS)/$(NET).tpb
ASM:=$(RESULTS)/$(NET).asm
SIMRES:=$(RESULTS)/$(NET).simres.$(N)

GOLDEN:=golden
FINAL:=$(NET)-out

#################################################
top : all
#################################################
$(TPB) : $(EXE)
	./$(EXE) $(TPB)

$(ASM) : $(TPB)
	./objdump $(TPB) > $@

simulate : $(TPB)
	./sim $(TPB)

$(EXE) : $(OBJ) $(TCCAR)
	g++ $(LFLAGS) -o $@ $(OBJ) $(LIBDIR_FLAGS) $(LIB_FLAGS)

$(CPP) :
	cd .. ; make $(NET)

$(OBJ) : $(CPP)
	g++ $(CPPFLAGS) -c $(CPP) 2>&1 | tee LOG

FMTS := int16 float16
FMTS := float16

npy2txt :
	for fmt in $(FMTS); do                                    \
	    for file in $(GOLDEN) $(FINAL); do                    \
	        npy=$$file.npy;  txt=$(RESULTS)/$$file-$$fmt.txt;            \
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

