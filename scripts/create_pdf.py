#!/usr/bin/python

import create_tex
import sys
import os

json_name = sys.argv[1]
o_name = sys.argv[2]
create_tex.json_to_tex(json_name, '/tmp/out.tex')
os.system("pdflatex /tmp/out.tex "+ o_name)
