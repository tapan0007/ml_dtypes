
DENSE169 = DenseNet-169
RES50 = ResNet-50

top : res

densenet :
	@  python main.py --densenet

resnet :
	@  python main.py --resnet

dbgdense :
	py -mpdb main.py --densenet

dbgres :
	py -mpdb main.py --resnet


clean :
	rm -f $$(find . -name '*.pyc')

dense : densenet

res : resnet

densedot :
	dot -Tps $(DENSE169).dot > $(DENSE169).ps; ps2pdf $(DENSE169).ps

resdot :
	dot -Tps $(RES50).dot > $(RES50).ps; ps2pdf $(RES50).ps

