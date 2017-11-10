#include <string.h>
#include "cnpy.h"
#include "tpb_isa.h"
#include "uarch_cfg.h"
#include "tcc.h"


void
get_dims(char *fname, unsigned int *&shape, unsigned int &word_size, ARBPRECTYPE &dtype) {
    bool fortran_order;
    unsigned int ndims;
    char ctype;
    FILE *fp = fopen(fname, "r");
    cnpy::parse_npy_header(fp, word_size, shape, ndims, fortran_order, ctype);

    switch (word_size) {
        case 1:
            switch (ctype) {
                case 'u':
                    dtype = UINT8;
                    break;
                case 'i':
                    dtype = INT8;
                    break;
                default:
                    assert(0);
            }
            break;
        case 2:
            switch (ctype) {
                case 'u':
                    dtype = UINT8;
                    break;
                case 'i':
                    dtype = INT8;
                    break;
                case 'f':
                    dtype = FP16;
                    break;
                default:
                    assert(0);
            }
            break;
        default:
            assert(0);
    }
}



#define SB_STEP 13

int 
main(int argc, char **argv) 
{
    addr_t ifmap_full_addr  = 0 * (1 << SB_STEP);
    addr_t ofmap_full_addr  = 2 * (1 << SB_STEP);
    unsigned int *i_dims;
    uint64_t i64_dims[4], o_dims[4];
    unsigned int word_size;
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


    get_dims(i_name, i_dims, word_size, dtype);
    for (size_t j = 0; j < 4; j++) {
        i64_dims[j] = i_dims[j]; /* type conversion */
    }

    compile_read_ifmap(fptr, ifmap_full_addr, i_name, "NCHW");
    compile_activation(fptr, ifmap_full_addr, i64_dims, 
            ofmap_full_addr, o_dims, dtype, dtype, (ACTIVATIONFUNC)act_func);
    compile_write_ofmap(fptr, o_name, ofmap_full_addr,
            o_dims, dtype);



    return 0;
}

