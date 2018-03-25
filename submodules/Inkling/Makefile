
SHELL := /bin/bash

build:
	make -C ${INKLING_PATH}
check:

clean:
	make -C ${INKLING_PATH} clean

# default install prefix
PREFIX=/tmp/
install: build
	mkdir -p $(DESTDIR)$(PREFIX)/bin
	cp ${INKLING_PATH}/sim/sim $(DESTDIR)$(PREFIX)/bin/sim
uninstall:
	rm $(DESTDIR)$(PREFIX)/bin/sim
