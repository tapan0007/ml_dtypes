#!/usr/bin/python
import json
import sys
import time
from collections import OrderedDict

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
    for (t,k,v) in fields:
        i = infos[-1]
        # mark start of this field
        i[HEADERS].append(curr_bit)

        # adjust size of field, based on type
        if v in defines:
            v = defines[v]
        if t == "bytearray":
            v = 8 * int(v)
        elif t == "bitfield":
            v = int(v)
        else:
            assert(False and "unrecognized field type")
        k = k.replace("_", "\_")

        # update where we are in the field
        curr_bit += v

        # if we go over, start new row
        if curr_bit >= base_bit + STEP:
            # how much of this field is inside vs outside of the row
            outside = curr_bit - (base_bit + STEP)
            inside = v - outside
            # finish this row
            if inside:
                i[HEADERS].append(base_bit + STEP - 1)
                new_k = "$\\dots$" + k if outside >= STEP else k
                i[BOXES].append((new_k, inside))
                i[BOXES].reverse() #put it in the right local order
            infos.append(([],[]))
            base_bit = base_bit + STEP
            # new row
            i = infos[-1]
            if outside >= STEP:
                while curr_bit >= base_bit + STEP:
                    base_bit += STEP
                i[HEADERS].append(base_bit - STEP)
                i[HEADERS].append(base_bit - 1)
                i[BOXES].append((" $\\dots$", STEP))
                infos.append(([],[]))
                i = infos[-1]
            # if it went past the next row, start skipping rows
            # OK, now ready to start new row and write the outside part
            if (curr_bit != base_bit):
                i[HEADERS].append(base_bit)
                i[HEADERS].append(curr_bit - 1)
                i[BOXES].append((k, curr_bit - base_bit))
        else: # we are still in the same row, sweeet
            i[HEADERS].append(curr_bit-1)
            i[BOXES].append((k,v))
    # no duplicate writing
    if infos[-1][HEADERS][-1] != curr_bit - 1:
        infos[-1][HEADERS].append(curr_bit-1)
    infos[-1][HEADERS].append(curr_bit)
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


def write_instruction(o, insn, fields, comments):
    o.write("\\verb|" + insn + "| \\\\\\\\ \n")
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
            j = json.load(json_file, object_pairs_hook=OrderedDict)
        except Exception,e:
            print("Unexpected error:", e)
            sys.exit(255)


    write_header(o)

    defines = j["defines"]
    for (insn, desc) in j["instructions"].iteritems():
        comments = desc.get("comments")
        opcode = desc.get("opcode")
        opcode_field = ["bitfield", "opcode={}".format(opcode), "OPCODE_BITS"]
        fields = get_info(defines, [opcode_field] + desc.get("fields")[1:])
        write_instruction(o, insn, fields, comments)

    write_footer(o)

if __name__ == "__main__":
    json_name = sys.argv[1]
    o_name = sys.argv[2]
    json_to_tex(json_name, o_name)


