SHELL := /bin/bash

PYTHON := python3
build:
	$(PYTHON) setup.py build
check:

clean:
	$(PYTHON) setup.py clean --all

# default install prefix
PREFIX=/tmp/
RECORD_FILE=install_files_rec.txt
install:
	$(PYTHON) setup.py install -O1  --prefix=$(DESTDIR)$(PREFIX) --record $(RECORD_FILE)
uninstall:
	if [[ -a $(RECORD_FILE) ]]; then \
	    echo "record file found" && \
	    for f in `cat $(RECORD_FILE)`; do rm $(DESTDIR)$(PREFIX)/$$f; done && \
	    rm $(RECORD_FILE); \
	fi
