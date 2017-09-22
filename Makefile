## SHELL=rc

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


