#!/usr/bin/python
import json
import sys

def write_header(o):
    o.write("#ifndef ISA_H\n")
    o.write("#define ISA_H\n\n\n")

def write_opcodes(o, opcodes):
    o.write("enum OPCODES {\n")
    for (k,v) in opcodes:
        o.write("{} = {},\n".format(k, v))
    o.write("};\n\n")

def write_defines(o, defines):
    for (k,v) in defines.iteritems():
        o.write("#define {} {}\n".format(k, v))
    o.write("\n\n")

def write_instruction(o, insn, desc):
    o.write("typedef struct {} {{\n".format(desc["template"]))
    for (k,v) in desc["fields"]:
        if v in defines:
            v = defines[v]
        v = int(v)
        o.write("   uint64_t    {:12}          : {};\n".format(k, v))
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

#defines
write_defines(o_name, defines)

#opcodes
opcode_map = [(ins, desc["opcode"]) for (ins, desc) in j["instructions"].iteritems()]
write_opcodes(o_name, opcode_map)

#instructions
for (insn, desc) in j["instructions"].iteritems():
    desc["fields"] = [("opcode", defines["OPCODE_BITS"])] + desc["fields"]
    write_instruction(o_name, insn, desc)

#footer
write_footer(o_name)


