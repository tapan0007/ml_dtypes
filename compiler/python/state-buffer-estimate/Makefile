PYTHON=python3

ALL_OPTS := 
ALL_OPTS := --print-sched --print-layers --print-levels 

DENSE169 = DenseNet-169
RES50 = ResNet-50

COMPILER_PY = compiler.py

top : res

resb :resnet-batch

resnet-residue :
	@  $(PYTHON) $(COMPILER_PY) --resnet $(ALL_OPTS)

resnet-relu :
	@  $(PYTHON) $(COMPILER_PY) --resnet --relu $(ALL_OPTS)

resnet-batch-relu :
	@  $(PYTHON) $(COMPILER_PY) --resnet --batch --relu $(ALL_OPTS)

densenet :
	@  $(PYTHON) $(COMPILER_PY) --densenet $(ALL_OPTS)

densenet-json :
	@  $(PYTHON) $(COMPILER_PY) --json densenet-169.json $(ALL_OPTS)

resnet :
	$(PYTHON) $(COMPILER_PY) --resnet $(ALL_OPTS)

resnet-batch :
	@  $(PYTHON) $(COMPILER_PY) --json resnet50.json  --batch $(ALL_OPTS)

resnet-json : resnet-batch

trivnet :
	$(PYTHON) $(COMPILER_PY) --trivnet $(ALL_OPTS)

trivnet :
	$(PYTHON) main.py --trivnet $(ALL_OPTS)


dbgdense :
	py -mpdb $(COMPILER_PY) --densenet $(ALL_OPTS)

dbgres :
	py -mpdb $(COMPILER_PY) --resnet $(ALL_OPTS)


clean :
	rm -f $$(find . -name '*.pyc')

dense : densenet

res : resnet

triv : trivnet

densedot :
	dot -Tps $(DENSE169).dot > $(DENSE169).ps; ps2pdf $(DENSE169).ps

resdot :
	dot -Tps $(RES50).dot > $(RES50).ps; ps2pdf $(RES50).ps

pylint :
	for mod in arch codegen layers memmgr nets schedule utils; do \
	    pylint $$mod > $$mod/$$mod.lint;  \
	done

lint : pylint
