SHELL := /bin/bash

check: 
	make clean
	python -m unittest discover

clean:
	rm -rf input/random/*.npy 
	rm -rf output/*.npy 
	rm -rf binary/*.bin 
	rm -rf stdout/*.txt 
