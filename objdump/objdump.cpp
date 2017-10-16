
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <string>
#include "tpb_isa.h"

#define PRINT_TEMPLATE(X)     "%-20s: 0x%#X  \n"


#define PF(FIELD, FORMAT_SPECIFIER)     print_field(PRINT_TEMPLATE(FORMAT_SPECIFIER), #FIELD, args->FIELD);

void
print_name_header(std::string name, FILE *fp)
{
   printf("0x%x @ %s: \n", (unsigned int)ftell(fp) - INSTRUCTION_NBYTES, name.c_str());
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
    if (argc < 2) {
        printf("Usage is %s [object file]", argv[0]);
        return 0;
    }

    fptr = fopen(argv[1], "r");
    while (fread(buffer, INSTRUCTION_NBYTES, 1, fptr)) {
        switch (TPB_OPCODE(*((uint64_t *)buffer))) {
            case SIM_RDFILTER_OPC:
                {
                    SIM_RDFILTER *args = (SIM_RDFILTER *)buffer;
                    print_name_header("SIM_RDFILTER", fptr);
                    PF(opcode, "%x")
                    PF(address, "%lx")
                    PF(fname, "%s")
                }
                break;
            case SIM_RDIFMAP_OPC:
                {
                    SIM_RDIFMAP *args = (SIM_RDIFMAP *)buffer;
                    print_name_header("SIM_RDIFMAP", fptr);
                    PF(opcode, "%x")
                    PF(address, "%lx")
                    PF(fname, "%s")
                }
                break;
            case MATMUL_OPC:
                {
                    MATMUL *args = (MATMUL *)buffer;
                    print_name_header("MATMUL", fptr);
                    PF(opcode, "%x")
                    PF(dequant_table_idx, "%x")
                    PF(quant_data_size, "%x")
                    PF(dequant_data_size, "%x")
                    PF(start_tensor_calc, "%x")
                    PF(stop_tensor_calc, "%x")
                    PF(reserved, "%x")
                    PF(dtype, "%x")
                    PF(fmap_start_addr, "%lx")
                    PF(fmap_x_step, "%lx")
                    PF(fmap_x_num, "%x")
                    PF(fmap_y_step, "%lx")
                    PF(fmap_y_num, "%x")
                    PF(last_row, "%x")
                    PF(n_pad, "%x")
                    PF(w_pad, "%x")
                    PF(e_pad, "%x")
                    PF(s_pad, "%x")
                    PF(psum_start_addr, "%lx")
                    PF(last_col, "%x")
                    PF(psum_step, "%x")
                    PF(event_func, "%x")
                    PF(event_id, "%x")
                }
                break;
            case SIM_WROFMAP_OPC:
                {
                    SIM_WROFMAP *args = (SIM_WROFMAP *)buffer;
                    print_name_header("SIM_WROFMAP", fptr);
                    PF(opcode, "%x")
                    PF(fname, "%s")
                    PF(address, "%lx")
                    PF(dims[0], "%lx")
                    PF(dims[1], "%lx")
                    PF(dims[2], "%lx")
                    PF(dims[3], "%lx")
                    PF(word_size, "%x")
                }
                break;
            case LDWEIGHTS_OPC:
                {
                    LDWEIGHTS *args = (LDWEIGHTS *)buffer;
                    print_name_header("LDWEIGHTS", fptr);
                    PF(opcode, "%x")
                    PF(address, "%lx")
                    PF(dtype, "%x")
                    PF(x_step, "%lx")
                    PF(x_num, "%x")
                    PF(y_step, "%lx")
                    PF(y_num, "%x")
                    PF(last_row, "%x")
                }
                break;
            case POOL_OPC:
                {
                    POOL *args = (POOL *)buffer;
                    print_name_header("POOL", fptr);
                    PF(opcode, "%x")
                    PF(pool_func, "%x")
                    PF(in_dtype, "%x")
                    PF(out_dtype, "%x")
                    PF(src_start_addr, "%lx")
                    PF(src_x_step, "%lx")
                    PF(src_x_num, "%lx")
                    PF(src_y_step, "%lx")
                    PF(src_y_num, "%lx")
                    PF(dst_start_addr, "%lx")
                    PF(dst_x_step, "%lx")
                    PF(dst_x_num, "%lx")
                    PF(dst_y_step, "%lx")
                    PF(dst_y_num, "%lx")
                    PF(str_x_step, "%lx")
                    PF(str_x_num, "%lx")
                    PF(str_y_step, "%lx")
                    PF(str_y_num, "%lx")
                    PF(max_partition, "%lx")
                    PF(event_func, "%x")
                    PF(event_id, "%x")
                }
                break;
            default:
                assert(0);

        }
    }
}
