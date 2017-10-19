ALL_OPTS := --print-layers  --batch
ALL_OPTS := --print-layers 
ALL_OPTS := 
ALL_OPTS := --print-layers --print-levels --batch
ALL_OPTS := --print-layers --print-levels 

DENSE169 = DenseNet-169
RES50 = ResNet-50

top : res

resb :resnet-batch

densenet :
	@  python main.py --densenet $(ALL_OPTS)

resnet :
	@  python main.py --resnet $(ALL_OPTS)

resnet-batch :
	@  python main.py --resnet $(ALL_OPTS) --batch

dbgdense :
	py -mpdb main.py --densenet $(ALL_OPTS)

dbgres :
	py -mpdb main.py --resnet $(ALL_OPTS)


clean :
	rm -f $$(find . -name '*.pyc')

dense : densenet

res : resnet

densedot :
	dot -Tps $(DENSE169).dot > $(DENSE169).ps; ps2pdf $(DENSE169).ps

resdot :
	dot -Tps $(RES50).dot > $(RES50).ps; ps2pdf $(RES50).ps

