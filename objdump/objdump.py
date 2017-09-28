#!/usr/bin/python
import sys
import json

def print_prologue(o):
    o.write(
'''
#include "string.h"
#include "stdio.h"
#include "isa.h"
#include "isadef.h"
#include "dtypes.h"
#include "assert.h"
#include "stdarg.h"
#include <string>

#define PF(FIELD)  \
   print_field("%-20s: 0x%lx \\n", #FIELD, args->FIELD);

#define PF_ARRAY(FIELD)  \
   print_field("%-20s: 0x%s \\n", #FIELD, args->FIELD);

void
print_name_header(std::string name, FILE *fp)
{
   printf("0x%x @ %s: \\n", (unsigned int)ftell(fp) - INSTRUCTION_NBYTES, name.c_str());
}

void
print_field(std::string format, ...)
{
    va_list args;
    va_start(args, format);
    printf("  ");
    vprintf(format.c_str(), args);
    va_end(args);
}

int
main(int argc, char **argv)
{
    FILE *fptr;
    char buffer[INSTRUCTION_NBYTES];
    uint64_t mask = (1 << OPCODE_BITS) - 1;
    if (argc < 2) {
        printf("Usage is %s [object file]", argv[0]);
        return 0;
    }

    fptr = fopen(argv[1], "r");
    while (fread(buffer, INSTRUCTION_NBYTES, 1, fptr)) {
        switch (*((uint64_t *)buffer) & mask) {
'''
)

def print_epilogue(o):
    o.write('''
        }
    }
}
'''
)

def print_switch(o, j):
    for (insn, desc) in j["instructions"].iteritems():
        o.writelines([
            '            case {}:\n'.format(insn),
            '                {\n',
            '                    {} *args = ({} *)buffer;\n'.format(
                desc["template"], desc["template"]),
            '                    print_name_header("{}", fptr);\n'.format(insn)])
        for (t,k,v) in desc["fields"]:
            if t == 'bytearray':
                o.write('                    PF_ARRAY({})\n'.format(k))
            else:
                o.write('                    PF({})\n'.format(k))
        o.write('                }\n')
        o.write('                break;\n')
    o.write('            default:\n')
    o.write('                assert(0);\n')

if len(sys.argv) < 3:
    print "Usage {} [JSON_FILE] [OUT_C_FILE]\n".format(sys.argv[0])
    sys.exit(1)

json_name = sys.argv[1]
o = open(sys.argv[2], "wb")
with open(json_name) as json_file:
    try:
        j = json.load(json_file)
    except Exception,e:
        print("Unexpected error:", e)
        sys.exit(255)


print_prologue(o)
print_switch(o, j)
print_epilogue(o)



