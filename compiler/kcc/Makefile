# codegen compiler

MODS = arch layers memmgr nets utils schedule

top :
	for mod in $(MODS); do (cd $$mod; make) done 2>&1 | tee e


clean :
	rm -f */obj/*.o */obj/lib*.a

