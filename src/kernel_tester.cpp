#include "kernel_tester.h"
#include "isa.h"
#include "cnpy.h"
#include "compiler.h"

KernelTester::KernelTester() {
    ibuf_start = (char *)calloc(1, 64*1024);
    ibuf_end = ibuf_start;
}

KernelTester::~KernelTester() {
    free(ibuf_end);
}

void
KernelTester::get_dims(char *fname, uint64_t *shape, size_t *word_size) {
    cnpy::NpyArray arr = cnpy::npy_load(fname);
    for (int i = 0; i < 4; i++) {
        shape[i] = (uint64_t)arr.shape[i];
    }
    *word_size = arr.word_size;
    free(arr.data);
}


void 
KernelTester::convolve(char *i_name, char *f_name,
                char *o_name, uint8_t padding[2])
{
    addr_t ifmap_full_addr  = 0 * (1 << BANK_BITS);
    addr_t filter_full_addr = 1 * (1 << BANK_BITS);
    addr_t ofmap_full_addr  = 2 * (1 << BANK_BITS);
    uint64_t o_rows, o_cols;
    uint64_t i_dims[4], f_dims[4];
    size_t word_size;
    size_t oword_size = 4; /* FIXME, no pinning! */
    ARBPRECTYPE dtype = UINT8; /* FIXME, no pinning! */
    uint8_t stride[] = {1,1};
    uint8_t dilate[] = {0,0};
    size_t ibuf_size = 0;

    get_dims(i_name, i_dims, &word_size);
    get_dims(f_name, f_dims, &word_size);

    compile_read_ifmap(&ibuf_end, ibuf_size, ifmap_full_addr, i_name);
    compile_read_filter(&ibuf_end, ibuf_size, filter_full_addr, f_name);
    compile_convolve(&ibuf_end, ibuf_size, o_rows, o_cols,
            ofmap_full_addr,
            ifmap_full_addr, i_dims,
            filter_full_addr, f_dims,
            dtype, padding, stride, dilate);
    compile_write_ofmap(&ibuf_end, ibuf_size, o_name, ofmap_full_addr,
            i_dims[0], f_dims[1], f_dims[0], o_rows, o_cols, oword_size);
    ibuf.setup(ibuf_start, ibuf_end);
}
