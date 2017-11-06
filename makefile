RM = rm -f
MAKE=make

.PHONY: clean

all:
	@$(MAKE) -C tcc
	@$(MAKE) -C tcc/test/convolve
	@$(MAKE) -C tcc/test/pool
	@$(MAKE) -C sim
	@$(MAKE) -C objdump
	@$(MAKE) libs -C sim
	@$(MAKE) -C verif

clean: 
	@$(MAKE) clean -C tcc
	@$(MAKE) clean -C tcc/test/convolve
	@$(MAKE) clean -C tcc/test/pool
	@$(MAKE) clean -C sim
	@$(MAKE) clean -C objdump
	@$(MAKE) clean -C verif


