#!/usr/bin/python
import json
import sys

def write_header(o):
    o.write("#ifndef ISA_H\n")
    o.write("#define ISA_H\n\n")
    o.write("#include \"stdint.h\"\n\n\n")

def write_instruction(o, insn, desc):
    o.write("typedef struct __attribute__ ((__packed__))  {} {{\n".format(\
            desc["template"]))
    for (t,k,v) in desc["fields"]:
        if v in defines:
            v = defines[v]
        v = int(v)
        if t == "bitfield":
            o.write("   uint64_t    {:12}          : {};\n".format(k, v))
        elif t == "bytearray":
            o.write("   char        {}[{}] ;\n".format(k, v)) 
        else:
            assert(false and "unrecognized type in field")

    o.write("}} {};\n\n".format(desc["template"]))

def write_footer(o):
    o.write("#endif\n")


json_name = sys.argv[1]
o_name = open(sys.argv[2], "wb")
with open(json_name) as json_file:
    try:
        j = json.load(json_file)
    except Exception,e:
        print("Unexpected error:", e)
        sys.exit(255)

defines = j["defines"]

#header
write_header(o_name)

#instructions
for (insn, desc) in j["instructions"].iteritems():
    write_instruction(o_name, insn, desc)

#footer
write_footer(o_name)


