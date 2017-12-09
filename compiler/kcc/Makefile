# codegen compiler

MODS = arch layers memmgr nets utils schedule serialize compiler

top :
	for mod in $(MODS); do (if cd $$mod; then make clean; make; fi) done


clean :
	rm -f */obj/*.o */obj/lib*.a

