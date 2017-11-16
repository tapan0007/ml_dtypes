#include <string.h>
#include "cnpy.h"
#include "tpb_isa.h"
#include "uarch_cfg.h"
#include "tcc.h"
#include "utils.h"

int 
main(int argc, char **argv) 
{
    addr_t ifmap_full_addr  = 0;
    addr_t ofmap_full_addr  = 0;
    uint64_t i_dims[4], o_dims[4];
    unsigned int word_size, tot_size;
    ARBPRECTYPE dtype = INVALID_ARBPRECTYPE;
    char *i_name, *o_name, *binary_name;
    int i = 1;
    int act_func = INVALID_ACTIVATIONFUNC;
    FILE *fptr;

    if (argc < 3) {
        printf("Usage is %s [-a ACT_FUNC_AS_INT] IFMAP_NPY OUTPUT_NPY OUTPUT_BINARY\n", argv[0]);
        return 1;
    }
    if (!strcmp(argv[i], "-a")) {
        i++;
        act_func = atoi(argv[i++]);
    }
    i_name = argv[i++];
    o_name = argv[i++];
    binary_name = argv[i++];

    if (!(fptr = fopen(binary_name, "w"))) {
        fprintf(stderr, "File did not open");
        return 1;
    }


    get_dims(i_name, i_dims, word_size, tot_size, dtype);
    ofmap_full_addr = ifmap_full_addr + tot_size;

    compile_read_ifmap(fptr, ifmap_full_addr, i_name, "NCHW");
    compile_activation(fptr, ifmap_full_addr, i_dims, 
            ofmap_full_addr, o_dims, dtype, dtype, (ACTIVATIONFUNC)act_func);
    compile_write_ofmap(fptr, o_name, ofmap_full_addr,
            o_dims, dtype);



    return 0;
}

