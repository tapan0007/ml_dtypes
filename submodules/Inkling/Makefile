
SHELL := /bin/bash
# use g++48 to build inkling if available.

G++48_EXISTS := $(shell command -v g++48 2> /dev/null)

INKLING_CXX = g++
ifdef G++48_EXISTS
INKLING_CXX = g++48
endif

build:
	$(MAKE) -C ${INKLING_PATH} CXX=${INKLING_CXX}
check:

clean:
	$(MAKE) -C ${INKLING_PATH} clean

# default install prefix
PREFIX=/tmp/
install: build
	mkdir -p $(DESTDIR)$(PREFIX)/bin
	cp ${INKLING_PATH}/sim/sim $(DESTDIR)$(PREFIX)/bin/sim
uninstall:
	rm $(DESTDIR)$(PREFIX)/bin/sim
