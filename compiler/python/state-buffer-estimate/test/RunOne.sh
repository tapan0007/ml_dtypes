SCRIPT=$0

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
case "x$INKLING_PATH" in
(x) INKLING_PATH=$HOME/code/CodeCommit/Inkling ;;
esac
case "x$KAENA_PATH" in
(x) KAENA_PATH=$HOME/code/GitFarm/Kaena ;;
esac


export COMPILER=$KAENA_PATH/compiler/python/state-buffer-estimate
export TEST=$COMPILER/test
export npy_diff=$KAENA_PATH/compiler/util/npy_diff_files
export DUMPNPY=$KAENA_PATH/compiler/util/dumpnpy
export INKLING=$INKLING_PATH

export PYTHONPATH=$PYTHONPATH:$COMPILER
##############################################################
test -d "$INKLING"  || Fatal Inkling directory "$INKLING" missing
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

case "x$1" in
(x*.tgz)
    TGZ=$1; 
    x=${TGZ%.tgz}
    Name="${x#.*/}"
    ;;
(*) Name=$1; TGZ=$Name.tgz;;
esac

DIR=./results
##############################################################
SIM=$INKLING/sim/sim
OBJDUMP=$INKLING/objdump/objdump

##############################################################
tar xvfz $TGZ
#PYTHONPATH="$PYTHONPATH:$PWD"
. ./files.sh 
sed1='s/.*":  *"//'; sed2='s/",.*//'

InputNpy=$( egrep '"input_file":' $JsonFile | sed -e "$sed1" -e "$sed2" )
NetName=$( egrep '"net_name":' $JsonFile | sed -e "$sed1" -e "$sed2" )
LastLayerName=$( egrep '"layer_name":' $JsonFile | tail -n 1| sed -e "$sed1" -e "$sed2" )

NET=$NetName

CPP=$NET.cpp
OBJ=$NET.o
EXE=$NET-exe

RESULTS=./results/$Name
rm -fR $RESULTS; mkdir -p $RESULTS || Fatal Cannot mkdir dir $DIR

ASM=$RESULTS/$NET.asm
TPB=$RESULTS/$NET.tpb

SIMRES=$RESULTS/$NET.simres
SIMLOG=$RESULTS/simulation.log
LOG=$RESULTS/LOG

##############################################################
{


##############################################################
## First make NET.py file
#touch __init__.py
RunCmd python3 $COMPILER/main.py --json $JsonFile
cp -p $CPP $RESULTS/.

##############################################################
## compile C++
FLAGS="-W -Wall -Werror -ggdb -g"
INC_FLAGS="-I$INKLING/shared/inc -I$INKLING/tcc/inc"
CFLAGS="$FLAGS -I. $INC_FLAGS -Wno-missing-field-initializers"
CPPFLAGS="$CFLAGS -std=c++11"
LDFLAGS="$FLAGS -ltcc"
LIBDIR1="$INKLING/tcc/libs"
LIBDIR_FLAGS="-L$LIBDIR1"

g++ $CPPFLAGS -c "$CPP" || Fatal Failed to compile $CPP
g++ $LDFLAGS -o $EXE $OBJ $LIBDIR_FLAGS $LIB_FLAGS || Fatal Failed to link $OBJ

##############################################################
RunCmd ./$EXE $TPB || Fatal Failed ./$EXE $TPB

##############################################################
RunCmd $OBJDUMP $TPB > $ASM || Fatal Failed to create $ASM
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

echo $npy_diff $OutputNpy $SimOutputNpy '>' $RESULTS/$NET.diff
$npy_diff $OutputNpy $SimOutputNpy 2>&1 | tee $RESULTS/$NET.diff


for fmt in $FMTS; do
    for npyFile in $NpyFiles; do
        txt=$RESULTS/$npyFile-$fmt.txt
        echo $DUMPNPY --$fmt $npyFile '>' $txt
        $DUMPNPY --$fmt $npyFile > $txt
        shasum  $txt
        #cat $txt
    done
done

#echo GOLDEN:
#cat $OutputNpy-float16.txt
#echo FROM SIM:
#cat $SimOututNpy-float16.txt

} 2>&1 | tee $LOG

