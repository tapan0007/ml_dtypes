RM = rm -f
MAKE=make

.PHONY: clean

all:
	@$(MAKE) -C shared
	@$(MAKE) -C tcc
	@$(MAKE) -C tcc/test
	@$(MAKE) -C sim

clean: 
	@$(MAKE) clean -C shared
	@$(MAKE) clean -C tcc
	@$(MAKE) clean -C tcc/test
	@$(MAKE) clean -C sim


