
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <string>
#include "tpb_isa.h"

#define PRINT_TEMPLATE(X)     "%-20s: "X"  \n"

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
                    PF(opcode, "0x%x")
                    PF(address, "0x%lx")
                    PF(fname, "%s")
                }
                break;
            case SIM_RDIFMAP_OPC:
                {
                    SIM_RDIFMAP *args = (SIM_RDIFMAP *)buffer;
                    print_name_header("SIM_RDIFMAP", fptr);
                    PF(opcode, "0x%x")
                    PF(address, "0x%lx")
                    PF(fname, "%s")
                }
                break;
            case MATMUL_OPC:
                {
                    MATMUL *args = (MATMUL *)buffer;
                    print_name_header("MATMUL", fptr);
                    PF(opcode, "0x%x")
                    PF(dequant_table_idx, "0x%x")
                    PF(quant_data_size, "0x%x")
                    PF(dequant_data_size, "0x%x")
                    PF(start_tensor_calc, "0x%x")
                    PF(stop_tensor_calc, "0x%x")
                    PF(reserved, "0x%x")
                    PF(dtype, "0x%x")
                    PF(fmap_start_addr, "0x%lx")
                    PF(fmap_x_step, "0x%lx")
                    PF(fmap_x_num, "0x%x")
                    PF(fmap_y_step, "0x%lx")
                    PF(fmap_y_num, "0x%x")
                    PF(last_row, "0x%x")
                    PF(n_pad, "0x%x")
                    PF(w_pad, "0x%x")
                    PF(e_pad, "0x%x")
                    PF(s_pad, "0x%x")
                    PF(psum_start_addr, "0x%lx")
                    PF(last_col, "0x%x")
                    PF(psum_step, "0x%x")
                    PF(event_func, "0x%x")
                    PF(event_id, "0x%x")
                }
                break;
            case SIM_WROFMAP_OPC:
                {
                    SIM_WROFMAP *args = (SIM_WROFMAP *)buffer;
                    print_name_header("SIM_WROFMAP", fptr);
                    PF(opcode, "0x%x")
                    PF(fname, "%s")
                    PF(address, "0x%lx")
                    PF(dims[0], "0x%lx")
                    PF(dims[1], "0x%lx")
                    PF(dims[2], "0x%lx")
                    PF(dims[3], "0x%lx")
                    PF(dtype, "0x%x")
                }
                break;
            case LDWEIGHTS_OPC:
                {
                    LDWEIGHTS *args = (LDWEIGHTS *)buffer;
                    print_name_header("LDWEIGHTS", fptr);
                    PF(opcode, "0x%x")
                    PF(address, "0x%lx")
                    PF(dtype, "0x%x")
                    PF(x_step, "0x%lx")
                    PF(x_num, "0x%x")
                    PF(y_step, "0x%lx")
                    PF(y_num, "0x%x")
                    PF(last_row, "0x%x")
                }
                break;
            case POOL_OPC:
                {
                    POOL *args = (POOL *)buffer;
                    print_name_header("POOL", fptr);
                    PF(opcode, "0x%x")
                    PF(pool_func, "0x%x")
                    PF(in_dtype, "0x%x")
                    PF(out_dtype, "0x%x")
                    PF(src_start_addr, "0x%lx")
                    PF(src_x_step, "0x%lx")
                    PF(src_x_num, "0x%lx")
                    PF(src_y_step, "0x%lx")
                    PF(src_y_num, "0x%lx")
                    PF(dst_start_addr, "0x%lx")
                    PF(dst_x_step, "0x%lx")
                    PF(dst_x_num, "0x%lx")
                    PF(dst_y_step, "0x%lx")
                    PF(dst_y_num, "0x%lx")
                    PF(str_x_step, "0x%lx")
                    PF(str_x_num, "0x%lx")
                    PF(str_y_step, "0x%lx")
                    PF(str_y_num, "0x%lx")
                    PF(max_partition, "0x%lx")
                    PF(event_func, "0x%x")
                    PF(event_id, "0x%x")
                }
                break;
            default:
                assert(0);

        }
    }
}
