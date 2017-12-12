# codegen compiler
	#egrep -v --line-buffered 'pedantic|ing director|rm |ar rc' | 

MODS = arch layers memmgr nets utils schedule serialize codegen
MAINMOD= compiler 

top :
	for mod in $(MODS) $(MAINMOD); do \
	    (if cd $$mod; then \
		    make clean; make -k; \
		fi) \
	done 2>&1 | tee e; \
	echo; echo '========================'; \
	if egrep -qi 'warn|error' e; then \
		egrep -i 'warn|error' e; false; \
	else true; \
	fi

#	| egrep -v --line-buffered 'pedantic|ing director|rm |ar rc' 

clean :
	rm -f */obj/*.o */obj/lib*.a

