SHELL := /bin/bash
ifndef KAENA_PATH
$(error KAENA_PATH is not set)
endif
ifndef ARCH_ISA_PATH
$(error ARCH_ISA_PATH is not set)
endif
ifndef ARCH_ARTIFACTS_PATH
$(error ARCH_ARTIFACTS_PATH is not set)
endif
ifndef INKLING_PATH
$(error INKLING_PATH is not set)
endif
ifndef KAENA_EXT_PATH
$(error KAENA_EXT_PATH is not set)
endif


PYTHON=python3
SUBDIRS ?= \
    runtime \
    compiler

.PHONY: build
build:
	@for subdir in $(SUBDIRS); \
		do $(MAKE) -j -C $$subdir PREFIX=${PREFIX} DESTDIR=$(DESTDIR) PYTHON=$(PYTHON) || exit -1; \
	done
	
PREFIX=/tmp
.PHONY: install
install:
	@for subdir in $(SUBDIRS); do \
		($(MAKE) -C $$subdir install PREFIX=${PREFIX} DESTDIR=$(DESTDIR) PYTHON=$(PYTHON)) || exit -1; \
	done

.PHONY: uninstall
uninstall:
	@for subdir in $(SUBDIRS); do \
		($(MAKE) -C $$subdir uninstall PREFIX=${PREFIX} DESTDIR=$(DESTDIR)) || exit -1; \
	done


check:
	$(MAKE) check -C ${KAENA_PATH}/test/e2e

check_emu:
	$(MAKE) check_emu -C ${KAENA_PATH}/test/e2e

clean:  build_clean repo_clean


build_clean:
	@for subdir in $(SUBDIRS); do \
		($(MAKE) -C $$subdir clean  PYTHON=$(PYTHON)) \
	done


