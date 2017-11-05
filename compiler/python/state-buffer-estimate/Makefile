PYTHON=python3

ALL_OPTS := --print-sched --print-layers --print-levels 
ALL_OPTS := 

DENSE169 = DenseNet-169
RES50 = ResNet-50

top : res

resb :resnet-batch

resnet-residue :
	@  $(PYTHON) main.py --resnet $(ALL_OPTS)

resnet-relu :
	@  $(PYTHON) main.py --resnet --relu $(ALL_OPTS)

resnet-batch-relu :
	@  $(PYTHON) main.py --resnet --batch --relu $(ALL_OPTS)

densenet :
	@  $(PYTHON) main.py --densenet $(ALL_OPTS)

resnet :
	$(PYTHON) main.py --resnet $(ALL_OPTS)

resnet-batch :
	@  $(PYTHON) main.py --resnet --batch $(ALL_OPTS)


trivnet :
	$(PYTHON) main.py --trivnet $(ALL_OPTS)


dbgdense :
	py -mpdb main.py --densenet $(ALL_OPTS)

dbgres :
	py -mpdb main.py --resnet $(ALL_OPTS)


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
