build:
	cd ${INKLING_PATH}; make
	cd ${KAENA_PATH}/compiler/be; make

check:
	cd ${KAENA_PATH}/compiler/tffe/test; ./RunAll
check_clean:
	cd ${KAENA_PATH}/compiler/tffe/test; /bin/rm -rf [0-9]*
