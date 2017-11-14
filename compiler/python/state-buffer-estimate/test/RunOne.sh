Error () {
    echo 1>&2 "$@"
}
Fatal () {
    Error '>>>' "$@"
    exit 1
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
(0) Fatal Usage $0 RESULT;;
esac

case "x$1" in
(x*.tgz) TGZ=$1; Name=${TGZ%.tgz};;
(*) Name=$1; TGZ=$Name.tgz;;
esac

DIR=./results
##############################################################
SIM=$INKLING/sim/sim
OBJDUMP=$INKLING/objdump/objdump

NET=trivnet

CPP=$NET.cpp
OBJ=$NET.o
EXE=$NET-exe

RESULTS=./results/$Name
rm -fR $RESULTS; mkdir $RESULTS || Fatal Cannot mkdir dir $DIR

ASM=$RESULTS/$NET.asm
TPB=$RESULTS/$NET.tpb

SIMRES=$RESULTS/$NET.simres
SIMLOG=$RESULTS/simulation.log
LOG=$RESULTS/LOG

##############################################################
{
tar xvfz $TGZ
PYTHONPATH="$PYTHONPATH:$PWD"

##############################################################
## First make NET.py file
touch __init__.py
RunCmd python3 $COMPILER/main.py --trivnet
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

g++ $CPPFLAGS -c "$CPP" || Fail Failed to compile $CPP
g++ $LDFLAGS -o $EXE $OBJ $LIBDIR_FLAGS $LIB_FLAGS || Fail Failed to link $OBJ

##############################################################
RunCmd ./$EXE $TPB || Fatal Failed $EXE $TPB

##############################################################
RunCmd $OBJDUMP $TPB > $ASM || Fatal Failed to create $ASM
RunCmd shasum $TPB

##############################################################
echo $SIM $TPB
$SIM $TPB >$SIMRES || Fatal Sim failed on $TPB

##############################################################
outNpy=$( egrep -A 1 compile_write_ofmap $CPP |
            tail -n 1 | sed -e 's/^ *"//' -e 's/.npy".*//' )
echo Out npy: $outNpy

Files="
    output
    $outNpy
"
    # int16

FMTS="
    float16
"

for f in $Files; do
    cp $f.npy $RESULTS/.
done

echo $npy_diff output.npy $outNpy.npy '>' $RESULTS/$NET.diff
$npy_diff output.npy $outNpy.npy 2>&1 | tee $RESULTS/$NET.diff

for fmt in $FMTS; do
    for file in $Files; do
        npy=$file.npy
        txt=$RESULTS/$file-$fmt.txt
        echo dumpnpy --$fmt $npy '>' $txt
        dumpnpy --$fmt $npy > $txt
        shasum  $txt
        #cat $txt
    done
done

#echo GOLDEN:
#cat output-float16.txt
#echo FROM SIM:
#cat $NET-out-float16.txt

} 2>&1 | tee $LOG
