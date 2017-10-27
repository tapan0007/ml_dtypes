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


int 
main(int argc, char **argv) 
{
    addr_t ifmap_base = 0 * (1 << BANK_BITS);
    addr_t filter_base = 1 * (1 << BANK_BITS);
    addr_t ofmap_addr  = 2 * (1 << BANK_BITS);
    addr_t ifmap_addr[2] = {ifmap_base,0};
    addr_t filter_addr[2] = {filter_base,0};
    uint64_t i_dims[4], f_dims[4];
    size_t word_size;
    size_t oword_size = 4; /* FIXME, no pinning! */
    ARBPRECTYPE dtype = UINT8; /* FIXME, no pinning! */
    uint8_t padding[2] = {0};
    uint8_t stride[] = {1,1};
    uint8_t dilate[] = {0,0};
    uint64_t o_dims[4] = {0,0,0,0};
    char *i_names[2], *f_names[2], *o_name, *binary_name;
    int i = 1;
    int num_inames = 0;
    int num_fnames = 0;
    FILE *fptr;

    if (argc < 3) {
        printf("Usage is %s [-p PAD] [-s STRIDE] [-i IFMAP_NPY]* FILTER_NPY OUTPUT_NPY OUTPUT_BINARY\n", argv[0]);
        return 1;
    }
    if (!strcmp(argv[i], "-p")) {
        padding[1] = atoi(argv[++i]);
        padding[0] = atoi(argv[++i]);
        i++;
    }
    if (!strcmp(argv[i], "-s")) {
        stride[1] = atoi(argv[++i]);
        stride[0] = atoi(argv[++i]);
        i++;
    }
    while (!strcmp(argv[i], "-i")) {
        i_names[num_inames++] = argv[++i];
        i++;
    }
    while (!strcmp(argv[i], "-f")) {
        f_names[num_fnames++] = argv[++i];
        i++;
    }
    o_name = argv[i++];
    binary_name = argv[i++];

    if (!(fptr = fopen(binary_name, "w"))) {
        fprintf(stderr, "File did not open");
        return 1;
    }



    assert(num_inames <= 2);
    uint64_t i_ch = 0;
    for (int j = 0; j < num_inames; j++) {
        get_dims(i_names[j], i_dims, &word_size);
        compile_read_ifmap(fptr, ifmap_addr[j], i_names[j], "NCHW");
        if (j + 1 < num_inames) {
            ifmap_addr[j+1] = ifmap_addr[j] + word_size * i_dims[3] * i_dims[2];
        }
        i_ch += i_dims[1];
    }
    /* sum of channels! */
    i_dims[1] = i_ch;

    uint64_t f_ch = 0;
    for (int j = 0; j < num_fnames; j++) {
        get_dims(f_names[j], f_dims, &word_size);
        compile_read_filter(fptr, filter_addr[j], f_names[j], "MCRS");
        if (j + 1 < num_fnames) {
            filter_addr[j+1] = filter_addr[j] + 
                word_size * f_dims[3] * f_dims[2] * f_dims[0];
        }
        f_ch += f_dims[1];
    }
    f_dims[1] = f_ch;


    compile_convolve(fptr, 
            ifmap_addr, i_dims,
            filter_addr, f_dims,
            ofmap_addr, o_dims,
            dtype, padding, stride, dilate);
    compile_write_ofmap(fptr, o_name, ofmap_addr, o_dims, oword_size);



    return 0;
}

