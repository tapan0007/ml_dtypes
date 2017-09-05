#!/usr/bin/python
import json
import sys

def get_info(fields):
    curr_bit = 0
    bitheaders = []
    bitboxes = []
    for (k,v) in fields:
        bitheaders.append(curr_bit)
        if v in defines:
            v = defines[v]
        v = int(v)
        k = k.replace("_", "\_")
        curr_bit += v
        bitheaders.append(curr_bit-1)
        bitboxes.append((k,v))
    bitheaders = [-(b - curr_bit)-1 for b in bitheaders[::-1]]
    return (bitheaders, bitboxes)


def write_header(o):
    o.write("\documentclass{article}\n")
    o.write("\usepackage[endianness=big]{bytefield}\n")
    o.write("\usepackage[margin=0.25in]{geometry}\n")
    o.write("\\begin{document}\n")

def write_instruction(o, opcode, bitheaders, bitboxes):
    box_len = bitheaders[-1] + 1
    o.write(opcode + "\\\\ \n")
    o.write("\\begin{{bytefield}}[bitwidth=1.5em]{{{}}}\n".format(box_len))
    o.write("\\bitheader{{{}}}\\\\\n".format(str(bitheaders)[1:-1]))
    for (text, width) in bitboxes:
        o.write("\\bitbox{{{}}}{{{}}}\n".format(width, text))
    o.write("\end{bytefield}\n")

def write_footer(o):
    o.write("\end{document}\n")


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
for (insn, desc) in j["instructions"].iteritems():
    opcode = desc["opcode"]
    opcode_field = ["opcode={}".format(opcode), "OPCODE_BITS"]
    (bitheaders, bitboxes) = get_info([opcode_field] + desc["fields"])
    write_instruction(o_name, insn, bitheaders, bitboxes)

write_footer(o_name)


