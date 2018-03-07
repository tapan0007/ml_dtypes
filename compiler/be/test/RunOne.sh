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

#case "x$1" in
#(x*.tgz)
#    TGZ=$1; 
#    x=${TGZ%.tgz}
#    Name="${x##.*/}"
#    ;;
#(*) Name=$1; TGZ=$Name.tgz;;
#esac
Name=nn

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

InputNpy=$( egrep '"ref_file":' $JsonFile | head -n 1 | sed -e "$sed1" -e "$sed2" )
NetName=$( egrep '"net_name":' $JsonFile | sed -e "$sed1" -e "$sed2" )
LastLayerName=$( egrep '"layer_name":' $JsonFile | tail -n 1| sed -e "$sed1" -e "$sed2" )

#NetName=$(echo $NetName | tr A-Z a-z)
NET=$NetName

RESULTS=./results/$Name
rm -fR $RESULTS; mkdir -p $RESULTS || Fatal Cannot mkdir dir $RESULTS

ASM=$RESULTS/$NET.asm

SIMRES=$RESULTS/$NET.simres
SIMLOG=$RESULTS/simulation.log
LOG=$RESULTS/LOG
##############################################################


TPB=$NET.tpb
inputGraphArgs="--json $JsonFile"
# Wave scheduler flow
if [[ $JsonFile = *"wavegraph.json" ]]; then
  inputGraphArgs="--wavegraph $JsonFile"
fi
cmd="$COMPILER/compiler/compiler.exe $inputGraphArgs"
RunCmd $cmd || Fatal Compiler failed

##############################################################
RunCmd $OBJDUMP $TPB > $ASM 
RunCmd shasum $TPB

##############################################################

