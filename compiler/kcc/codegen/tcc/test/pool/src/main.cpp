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
    uint64_t i_dims[4], o_dims[4], s_dims[4], k_dims[4];
    unsigned int word_size, tot_size;
    ARBPRECTYPE dtype = INVALID_ARBPRECTYPE;
    char *i_name, *o_name, *binary_name;
    int i = 1;
    int pool_func = MAX_POOL;
    FILE *fptr;

    if (argc < 3) {
        printf("Usage is %s [-p POOL_FUNC_AS_INT] IFMAP_NPY K_DIMS[4] S_DIMS[4] OUTPUT_NPY OUTPUT_BINARY\n", argv[0]);
        return 1;
    }
    if (!strcmp(argv[i], "-p")) {
        i++;
        pool_func = atoi(argv[i++]);
    }
    i_name = argv[i++];
    for (int j = 0; j < 4; j++) {
        k_dims[j] = atoi(argv[i++]);
    }
    for (int j = 0; j < 4; j++) {
        s_dims[j] = atoi(argv[i++]);
    }
    o_name = argv[i++];
    binary_name = argv[i++];

    if (!(fptr = fopen(binary_name, "w"))) {
        fprintf(stderr, "File did not open");
        return 1;
    }


    get_dims(i_name, i_dims, word_size, tot_size, dtype);

    compile_read_ifmap(fptr, ifmap_full_addr, i_name, "NCHW");
    compile_pool(fptr, ifmap_full_addr, i_dims, k_dims, 
            ofmap_full_addr, o_dims, s_dims, dtype, (POOLFUNC)pool_func);
    compile_write_ofmap(fptr, o_name, ofmap_full_addr,
            o_dims, dtype);



    return 0;
}

