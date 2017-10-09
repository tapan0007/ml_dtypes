#!/usr/bin/python
import json
import sys
import time
import re
from collections import OrderedDict

HEADERS=0
BOXES=1
STEP = 32

sizeof = {
    "char"  : 1,
    "uint8_t" : 1,
    "int8_t"  : 1,
    "int16_t" : 2,
    "uint16_t" : 2,
    "uint32_t":4,
    "int32_t" : 4,
    "uint64_t" : 8,
    "int64_t" : 8
}

def vector_len(s):
    l = 0 
    if (re.search(r"\[([0-9]+)\]", s)):
        l = int(m.group(1))
    return l


# each element of info is a tuple, the tuple holds two lists: the list of
# bitheaders that go with the row and the list of bitboxes that go with the row
def get_info(macros, fields):
    fields.reverse()
    infos = [([],[])]
    base_bit = 0
    curr_bit = 0
    HEADERS = 0
    BOXES = 1
    for tnc in fields:
        if (len(tnc) == 2):
            (t,n) = tnc
            c = ""
        else:
            (t,n,c) = tnc
        i = infos[-1]
        # mark start of this field
        i[HEADERS].append(curr_bit)

        n = n.replace("_", "\_")
        # adjust size of field, based on type
        if t in macros:
           t  = macros[t]
        v = 8 * sizeof[t] * vector_len(t)
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
                new_n = "$\\dots$" + n if outside >= STEP else n
                i[BOXES].append((new_n, inside))
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
                i[BOXES].append((n, curr_bit - base_bit))
        else: # we are still in the same row, sweeet
            i[HEADERS].append(curr_bit-1)
            i[BOXES].append((n,v))
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

    macros = j["macros"]
    for (insn, desc) in j["instructions"].iteritems():
        comments = desc.get("comments")
        opcode = desc.get("opcode")
        fields = get_info(macros, desc.get("fields"))
        write_instruction(o, insn, fields, comments)

    write_footer(o)

if __name__ == "__main__":
    json_name = sys.argv[1]
    o_name = sys.argv[2]
    json_to_tex(json_name, o_name)


