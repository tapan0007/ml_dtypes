# codegen compiler
#egrep -v --line-buffered 'pedantic|ing director|rm |ar rc' |

MODS := arch layers memmgr nets utils schedule serialize codegen
MAINMOD := compiler 

top :
	@for mod in $(MODS) $(MAINMOD); do \
	    (if cd $$mod; then \
			for D in obj; do test -d $$D || mkdir $$D; done; \
			$(MAKE) -k; \
		fi) \
	done 2>&1 | tee e; \
	echo; echo '========================'; \
	if egrep -wqi 'warning|error' e; then \
		egrep -i 'warning|error' e; false; \
	else true; \
	fi

#	| egrep -v --line-buffered 'pedantic|ing director|rm |ar rc' 

clean :
	rm -f */obj/*.o */obj/lib*.a *.exe

deps : alldeps

alldeps :
	@for mod in $(MODS) $(MAINMOD); do \
		(cd $$mod; \
		for D in dep; do test -d $$D || mkdir $$D; done; \
		$(MAKE) alldeps) \
	done
