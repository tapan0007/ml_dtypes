#! /bin/bash

set -e
SCRIPT=$0

##############################################################
Error () {
    echo 1>&2 "$@"
}
Fatal () {
    Error '>>>' "$@"
    exit 1
}
FatalUsage () {
    Fatal Usage: $SCRIPT TGZ-FILE
}

##############################################################
if ! test -d "$INKLING_PATH"; then
    Fatal Inkling path directory does not exist: "INKLING_PATH = $INKLING_PATH"
fi

if ! test -d "$KAENA_PATH"; then
    Fatal Kaena path directory does not exist: "KAENA_PATH = $KAENA_PATH"
fi


export COMPILER=$KAENA_PATH/compiler/python/state-buffer-estimate
export TEST=$COMPILER/test
export npy_diff=$KAENA_PATH/compiler/util/npy_diff_files
export DUMPNPY=$KAENA_PATH/compiler/util/npy2txt
export INKLING=$INKLING_PATH
SIM=$INKLING/sim/sim

## CODEGEN_TOP=$KAENA_PATH/compiler/codegen
CODEGEN_TOP=$INKLING

OBJDUMP=$CODEGEN_TOP/objdump/objdump

export PYTHONPATH=$PYTHONPATH:$COMPILER
##############################################################
test -d "$INKLING"  || Fatal Inkling directory "$INKLING" missing
test -d "$CODEGEN_TOP"  || Fatal Codegen directory "$CODEGEN_TOP" missing
test -d "$COMPILER" || Fatal Compiler directory "$COMPILER" missing

##############################################################

RunCmd () {
   #sleep 3
   echo "$@"
   "$@"
}

##############################################################
case $# in
(0) FatalUsage;;
esac

JSON=false

while let "$# != 0"
do
    case "x$1" in
    (x-json|x--json) JSON=true ;;
    (x-*) Fatal Wrong option "$1" ;;
    (*) break;;
    esac
done

case "x$1" in
(x*.tgz)
    TGZ=$1; 
    x=${TGZ%.tgz}
    Name="${x#.*/}"
    ;;
(*) Name=$1; TGZ=$Name.tgz;;
esac

##############################################################

##############################################################
tar xvfz $TGZ
#PYTHONPATH="$PYTHONPATH:$PWD"
F=./net_json_params.sh
test -r $F || Fatal missing file $F
. $F

sed1='s/.*": *"//'; sed2='s/"[,]* *//'

InputNpy=$( egrep '"ref_file":' $JsonFile | head -n 1 | sed -e "$sed1" -e "$sed2" )
NetName=$( egrep '"net_name":' $JsonFile | sed -e "$sed1" -e "$sed2" )
LastLayerName=$( egrep '"layer_name":' $JsonFile | tail -n 1| sed -e "$sed1" -e "$sed2" )

NetName=$(echo $NetName | tr A-Z a-z)
NET=$NetName

CPP=$NET.cpp
OBJ=$NET.o
EXE=$NET-exe

RESULTS=./results/$Name
rm -fR $RESULTS; mkdir -p $RESULTS || Fatal Cannot mkdir dir $RESULTS

ASM=$RESULTS/$NET.asm
TPB=$RESULTS/$NET.tpb

SIMRES=$RESULTS/$NET.simres
SIMLOG=$RESULTS/simulation.log
LOG=$RESULTS/LOG
##############################################################


##############################################################
## First make NET.py file
#touch __init__.py
cmd="python3 $COMPILER/compiler.py --json $JsonFile"
RunCmd $cmd
cp -p $CPP $RESULTS/.

##############################################################
## compile C++
FLAGS="-W -Wall -Werror -ggdb -g"

INC_FLAGS="-I$CODEGEN_TOP/shared/inc -I$CODEGEN_TOP/tcc/inc"
CFLAGS="$FLAGS -I. $INC_FLAGS -Wno-missing-field-initializers"
CPPFLAGS="$CFLAGS -std=c++11"
LDFLAGS="$FLAGS -ltcc"
LIBDIR1="$CODEGEN_TOP/tcc/libs"
LIBDIR_FLAGS="-L$LIBDIR1"

CXX=clang++
CXX=g++
RunCmd $CXX $CPPFLAGS -c $CPP
RunCmd $CXX -o $EXE $OBJ $LIBDIR_FLAGS $LIB_FLAGS $LDFLAGS 

##############################################################
RunCmd ./$EXE $TPB 

##############################################################
RunCmd $OBJDUMP $TPB > $ASM 
RunCmd shasum $TPB

##############################################################
echo $SIM $TPB
$SIM $TPB >$SIMRES || Fatal Sim failed on $TPB

SimOutputNpy="$NetName-$LastLayerName-simout.npy"
SimOutputNpy="$(echo $SimOutputNpy | sed -e 's@/@-@g')"
##############################################################
echo Out npy: $OutputNpy

NpyFiles="
    $OutputNpy
    $SimOutputNpy
"
    # int16

FMTS="
    float16
"

for f in $Files; do
    cp $f $RESULTS/.
done

##############################################################
for fmt in $FMTS; do
    for npyFile in $NpyFiles; do
        txt=$RESULTS/$npyFile-$fmt.txt
        echo $DUMPNPY --$fmt $npyFile '>' $txt
        $DUMPNPY --$fmt $npyFile > $txt
        shasum  $txt
        #cat $txt
    done
done

#echo GOLDEN from framework '(TF)':
#cat $OutputNpy-float16.txt
#echo FROM SIM:
#cat $SimOututNpy-float16.txt

##############################################################
cmd="$npy_diff --gold $OutputNpy --new $SimOutputNpy --verbose 2"
echo $cmd '| tee ' $RESULTS/$NET.diff
$cmd 2>&1 | tee $RESULTS/$NET.diff

diffStatus="${PIPESTATUS[0]}"    ### collect status of npy_diff process
## echo status: $?, diffStatus $diffStatus

case $diffStatus in
(0) Error Finished successfully ;;
(*) Error Failed comparison of golden and simulation results;;
esac
exit $diffStatus

