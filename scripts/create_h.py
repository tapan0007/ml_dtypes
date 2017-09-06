#!/usr/bin/python
import json
import sys

def write_header(o):
    o.write("#ifndef ISA_H\n")
    o.write("#define ISA_H\n\n\n")

def write_instruction(o, insn, desc):
    o.write("typedef struct {} {{\n".format(desc["template"]))
    for (k,v) in desc["fields"]:
        if v in defines:
            v = defines[v]
        v = int(v)
        o.write("   uint64_t    {:12}          : {};\n".format(k, v))
    o.write("}} {};\n".format(desc["template"]))

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


write_header(o_name)

defines = j["defines"]
opcodes = []
for (insn, desc) in j["instructions"].iteritems():
    opcodes.append(desc["opcode"])
    write_instruction(o_name, insn, desc)

write_footer(o_name)


