build:
	cd ${INKLING_PATH}/sim; make
	cd ${KAENA_PATH}/compiler/be; make
check:
	cd ${KAENA_PATH}/compiler/tffe/test; ./RunAll

clean:  build_clean check_clean

check_clean:
	cd ${KAENA_PATH}/compiler/tffe/test; /bin/rm -rf [0-9]*
build_clean:
	cd ${INKLING_PATH}; make clean
	cd ${KAENA_PATH}/compiler/be; make clean

