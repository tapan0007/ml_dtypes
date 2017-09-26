## SHELL=rc

DENSE169 = DenseNet-169
RES50 = ResNet-50

dense : densenet

res : resnet

densenet :
	@  python main.py --densenet
	@# PYTHONPATH=$$PYTHONPATH:. python main.py

resnet :
	@  python main.py --resnet

dbgdense :
	python -mpdb main.py --densenet

dbgres :
	python -mpdb main.py --resnet


clean :
	rm -f *.pyc


densedot :
	dot -Tps $(DENSE169).dot > $(DENSE169).ps; ps2pdf $(DENSE169).ps

resdot :
	dot -Tps $(RES50).dot > $(RES50).ps; ps2pdf $(RES50).ps

