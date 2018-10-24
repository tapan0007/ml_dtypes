
## instructions
ifndef ARCH_ISA_PATH
$(error ARCH_ISA_PATH is not set)
endif
## compiler
ifndef KAENA_PATH
$(error KAENA_PATH is not set)
endif
## JSON
ifndef KAENA_EXT_PATH
$(error KAENA_EXT_PATH is not set)
endif

# codegen compiler
#egrep -v --line-buffered 'pedantic|ing director|rm |ar rc'

MODS := arch layers memmgr nets utils schedule serialize events wavecode wave kelf dma compisa
MAINMOD := compiler
COMPILER_BE := $(shell pwd)

top : alldeps
	@for mod in $(MODS) $(MAINMOD); do \
	    (if cd $$mod; then \
			for D in obj; do test -d $$D || mkdir $$D; done; \
			$(MAKE) -k COMPILER_BE=$(COMPILER_BE); \
		fi) \
	done 2>&1 | tee e; \
	echo; echo '========================'; \
	if egrep -wqi 'warning|error' e; then \
		egrep -i 'warning|error' e; false; \
	else true; \
	fi

#	| egrep -v --line-buffered 'pedantic|ing director|rm |ar rc'

clean :
	rm -f */obj/*.o */obj/lib*.a *.exe */dep/*

deps : alldeps

alldeps :
	@for mod in $(MODS) $(MAINMOD); do \
		(cd $$mod; \
		for D in dep; do test -d $$D || mkdir $$D; done; \
		$(MAKE) alldeps) \
	done

superclean : clean alldeps top

install : top
	@for mod in $(MAINMOD); do \
		(cd $$mod; \
		$(MAKE) install) \
	done

uninstall:
	@for mod in $(MAINMOD); do \
		(cd $$mod; \
		$(MAKE) uninstall) \
	done
libs :
	@for mod in $(MODS) $(MAINMOD); do \
	    (if cd $$mod; then \
			$(MAKE) -k lib; \
		fi) \
	done

