#!/usr/bin/python
import json
import sys

def write_header(o):
    o.write("#ifndef ISADEF_H\n")
    o.write("#define ISADEF_H\n\n")
    o.write("#include \"stdint.h\"\n\n\n")

def write_opcodes(o, opcodes):
    o.write("enum OPCODES {\n")
    for (k,v) in opcodes:
        o.write("{} = {},\n".format(k, v))
    o.write("};\n\n")

def write_defines(o, defines):
    for (k,v) in defines.iteritems():
        o.write("#define {} {}\n".format(k, v))
    o.write("\n\n")

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

#footer
write_footer(o_name)


