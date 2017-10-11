RM = rm -f
MAKE=make

.PHONY: clean

all:
	@$(MAKE) -C tcc
	@$(MAKE) -C tcc/test/convolve
	@$(MAKE) -C sim
	#@$(MAKE) -C objdump
	@$(MAKE) libs -C sim

clean: 
	@$(MAKE) clean -C tcc
	@$(MAKE) clean -C tcc/test/convolve
	@$(MAKE) clean -C sim
	#@$(MAKE) clean -C objdump


