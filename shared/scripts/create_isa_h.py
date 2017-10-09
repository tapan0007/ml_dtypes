#!/usr/bin/python
import json
import sys
from collections import OrderedDict

indent = 0
tab = 4

def pp(o, s):
    if isinstance(s, basestring):
        s = [s]
    for line in s:
        o.write("{}{}\n".format(" "*indent, line))

def write_header_footer(o, h):
    pp(o, h)


def write_instruction(o, insn, desc):
    global indent
    pp(o, "struct {} {{".format(insn))
    indent += tab
    for tnc in desc["fields"]:
        for (i,x) in enumerate(tnc):
            while tnc[i] in macros:
                tnc[i] = macros[tnc[i]]
        (t,n) = tnc[0:2]
        c = "" if len(tnc)==2 else "// " + tnc[2]
        pp(o, "{:12}    {:12}   {}".format(t, n + ";", c))
    if "constructor" in desc:
        c = desc["constructor"]
        pp(o, c[0])
        if len(c) > 1:
            indent += tab
            pp(o, c[1:-2])
            indent -= tab
            if len(c) > 2:
                pp(o, c[-1])
    indent -= tab
    pp(o, "} TONGA_PACKED;\n\n")


json_name = sys.argv[1]
o_name = open(sys.argv[2], "wb")
with open(json_name) as json_file:
    try:
        j = json.load(json_file, object_pairs_hook=OrderedDict)
    except Exception,e:
        print("Unexpected error:", e)
        sys.exit(255)

macros = j["macros"]

#header
write_header_footer(o_name, j["header"])

#instructions
for (insn, desc) in j["instructions"].iteritems():
    write_instruction(o_name, insn, desc)

#footer
write_header_footer(o_name, j["footer"])


