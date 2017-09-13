#!/usr/bin/python
import json
import sys
import time

HEADERS=0
BOXES=1
STEP = 32

# each element of info is a tuple, the tuple holds two lists: the list of
# bitheaders that go with the row and the list of bitboxes that go with the row
def get_info(defines, fields):
    fields.reverse()
    infos = [([],[])]
    base_bit = 0
    curr_bit = 0
    HEADERS = 0
    BOXES = 1
    for (k,v) in fields:
        i = infos[-1]
        i[HEADERS].append(curr_bit)
        if v in defines:
            v = defines[v]
        v = int(v)
        k = k.replace("_", "\_")
        curr_bit += v
        # if we go over, start new row
        if curr_bit >= base_bit + STEP:
            remainder = curr_bit - (base_bit + STEP)
            i[HEADERS].append(base_bit + STEP - 1)
            i[BOXES].append((k, v - remainder))
            i[BOXES].reverse() #put it in the right local order
            base_bit = base_bit + STEP
            infos.append(([base_bit, curr_bit-1],[(k, remainder)]))
        else:
            i[HEADERS].append(curr_bit-1)
            i[BOXES].append((k,v))
    infos[-1][HEADERS].append(base_bit + STEP - 1)
    infos[-1][BOXES].append(("unused", base_bit + STEP - curr_bit))
    infos[-1][BOXES].reverse()
    #put it in the right global order
    infos.reverse()
    return infos


def write_header(o):
    o.write("\documentclass{article}\n")
    o.write("\usepackage[endianness=big]{bytefield}\n")
    o.write("\usepackage[margin=0.50in]{geometry}\n")
    o.write("\\begin{document}\n")
    o.write("\\title{TONGA ISA} \n")
    ## dd/mm/yyyy format
    date = time.strftime("%m/%d/%Y")
    o.write("\\date{" + date + "} \n")
    o.write("\\maketitle\n")


def write_instruction(o, opcode, fields, comments):
    o.write(opcode + "\\\\\\\\ \n")
    o.write("\\begin{{bytefield}}[bitwidth=1.5em]{{{}}}\n".format(STEP))
    for (bitheaders, bitboxes) in fields:
        o.write("\\bitheader[lsb={}]{{{}}}\\\\\n".format(bitheaders[0], str(bitheaders)[1:-1]))
        for (text, width) in bitboxes:
            o.write("\\bitbox{{{}}}{{{}}}\n".format(width, text))
        o.write("\\\\")
        o.write("\n")
    o.write("\end{bytefield}\n")
    if comments:
        o.write("\\\\")
        o.write(" ".join(comments))
    o.write("\\\\\\\\ \n")

def write_footer(o):
    o.write("\end{document}\n")


def json_to_tex(json_name, o_name):
    o = open(o_name, "wb")
    with open(json_name) as json_file:
        try:
            j = json.load(json_file)
        except Exception,e:
            print("Unexpected error:", e)
            sys.exit(255)


    write_header(o)

    defines = j["defines"]
    for (insn, desc) in j["instructions"].iteritems():
        comments = desc.get("comments")
        opcode = desc.get("opcode")
        opcode_field = ["opcode={}".format(opcode), "OPCODE_BITS"]
        fields = get_info(defines, [opcode_field] + desc.get("fields"))
        write_instruction(o, insn, fields, comments)

    write_footer(o)

if __name__ == "__main__":
    json_name = sys.argv[1]
    o_name = sys.argv[2]
    json_to_tex(json_name, o_name)


