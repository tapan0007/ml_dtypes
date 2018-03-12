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


export COMPILER=$KAENA_PATH/compiler/be
export TEST=$COMPILER/test
export npy_diff=$KAENA_PATH/compiler/util/npy_diff_files
export DUMPNPY=$KAENA_PATH/compiler/util/npy2txt
export INKLING=$INKLING_PATH
SIM=$INKLING/sim/sim

##############################################################
## CODEGEN_TOP=$KAENA_PATH/compiler/be/codegen
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


##############################################################

##############################################################
# Use local directory instead of fixed tgz file since the NNE flow
# needs to generate the local inputs by translation of the previous
# subgraph outputs
#tar xvfz $TGZ
#PYTHONPATH="$PYTHONPATH:$PWD"
F=./net_json_params.sh
test -r $F || Fatal missing file $F
. $F

sed1='s/.*": *"//'; sed2='s/"[,]* *//'
NetName=$( egrep '"net_name":' $JsonFile | sed -e "$sed1" -e "$sed2" )
NET=$NetName

RESULTS=./results/$Name
rm -fR $RESULTS; mkdir -p $RESULTS || Fatal Cannot mkdir dir $RESULTS

Engines="pe pool act sp"


##############################################################
Parallel=false
Parallel=true


inputGraphArgs="--json $JsonFile"
# Wave scheduler flow
if [[ $JsonFile = *"wavegraph.json" ]]; then
  inputGraphArgs="--wavegraph $JsonFile"
fi

if $Parallel; then
    cmd="$COMPILER/compiler/compiler.exe --parallel $inputGraphArgs"
else
    cmd="$COMPILER/compiler/compiler.exe $inputGraphArgs"
fi

RunCmd $cmd || Fatal Compiler failed

##############################################################
if $Parallel; then
    for f in $Engines; do
        TPB=$NET-$f.tpb
        ASM=$RESULTS/$NET-$f.asm
        RunCmd $OBJDUMP $TPB > $ASM
        RunCmd shasum $TPB
    done
else 
    TPB=$NET.tpb
    ASM=$RESULTS/$NET.asm
    RunCmd $OBJDUMP $TPB > $ASM 
    RunCmd shasum $TPB
fi

##############################################################

