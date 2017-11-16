RM = rm -f
MAKE=make
CXX:=g++

.PHONY: clean

all:
	@$(MAKE) CXX=$(CXX) -C tcc
	@$(MAKE) CXX=$(CXX) -C tcc/test/convolve
	@$(MAKE) CXX=$(CXX) -C tcc/test/pool
	@$(MAKE) CXX=$(CXX) -C tcc/test/activation
	@$(MAKE) CXX=$(CXX) -C sim
	@$(MAKE) CXX=$(CXX) -C objdump
	@$(MAKE) libs CXX=$(CXX) -C sim
	@$(MAKE) CXX=$(CXX) -C verif

clean: 
	@$(MAKE) clean -C tcc
	@$(MAKE) clean -C tcc/test/convolve
	@$(MAKE) clean -C tcc/test/pool
	@$(MAKE) clean -C tcc/test/activation
	@$(MAKE) clean -C sim
	@$(MAKE) clean -C objdump
	@$(MAKE) clean -C verif


