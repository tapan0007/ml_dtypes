#include <string.h>
#include "cnpy.h"
#include "tpb_isa.h"
#include "uarch_cfg.h"
#include "tcc.h"



void
get_dims(char *fname, uint64_t *shape, size_t *word_size) {
    cnpy::NpyArray arr = cnpy::npy_load(fname);
    for (int i = 0; i < 4; i++) {
        shape[i] = (uint64_t)arr.shape[i];
    }
    *word_size = arr.word_size;
    free(arr.data);
}


#define SB_STEP 13

int 
main(int argc, char **argv) 
{
    addr_t ifmap_full_addr  = 0 * (1 << SB_STEP);
    addr_t ofmap_full_addr  = 2 * (1 << SB_STEP);
    uint64_t i_dims[4], o_dims[4], s_dims[4], k_dims[4];
    size_t word_size;
    ARBPRECTYPE dtype = UINT8;
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


    get_dims(i_name, i_dims, &word_size);

    compile_read_ifmap(fptr, ifmap_full_addr, i_name, "NCHW");
    compile_pool(fptr, ifmap_full_addr, i_dims, k_dims, 
            ofmap_full_addr, o_dims, s_dims, dtype, (POOLFUNC)pool_func);
    compile_write_ofmap(fptr, o_name, ofmap_full_addr,
            o_dims, word_size);



    return 0;
}

