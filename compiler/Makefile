
SHELL := /bin/bash

PYTHON := $(PYTHON)


.PHONY: build
build:
	$(PYTHON) setup.py build || exit 1
	$(MAKE) -j -C ./be top || exit 1
check:

clean:
	$(PYTHON) setup.py clean --all
	$(MAKE) -C ./be clean

# default install prefix
PREFIX=/tmp/
RECORD_FILE=install_files_rec.txt
.PHONY: install
install: build
	$(PYTHON) setup.py install -O1  --prefix=$(DESTDIR)$(PREFIX) --record $(RECORD_FILE)
	$(MAKE) -C ./be install PREFIX=${PREFIX}
.PHONY: uninstall
uninstall:
	set -x
	if [[ -a $(RECORD_FILE) ]]; then \
	    echo "record file found" && \
	    for f in `cat $(RECORD_FILE)`; do rm $(DESTDIR)$(PREFIX)/$$f; done && \
	    rm $(RECORD_FILE); \
	fi
	$(MAKE) -C ./be uninstall PREFIX=${PREFIX} DESTDIR=$(DESTDIR)

