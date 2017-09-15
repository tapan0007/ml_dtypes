#include "compiler.h"
#include "isa.h"
#include "internal_isa.h"
#include <string>
#include "string.h"
#include <assert.h>

#define IBUF_SIZE 64*1024

void compile(char *i_name, char *f_name, char *o_name) {
    char c_dest[IBUF_SIZE];
    void *dest = (void *)(c_dest);
    size_t dest_size;
    uint64_t o_rows, o_cols;
    uint64_t ofmap_full_addr;
    uint64_t ifmap_full_addr;
    uint64_t filter_full_addr;
    uint64_t i_nchw[] = {1,3,224,224};
    uint64_t w_mcrs[] = {64,3,7,7};
    uint8_t pad[] = {3,3};
    uint8_t stride[] = {1,1};
    uint8_t dilate[] = {0,0};
    uint8_t out_word_size = 4;

    /* do convolution, read from bank 0, write to bank 1 */
    ifmap_full_addr  = 0 * (1 << BANK_BITS); 
    filter_full_addr = 1 * (1 << BANK_BITS);
    ofmap_full_addr  = 2 * (1 << BANK_BITS);

    compile_read_ifmap(&dest, dest_size, ifmap_full_addr, i_name);
    compile_read_filter(&dest, dest_size, filter_full_addr, f_name);
    compile_convolve(&dest, dest_size, 
            o_rows, o_cols, 
            ofmap_full_addr,
            ifmap_full_addr, i_nchw,
            filter_full_addr, w_mcrs,
            UINT8,
            pad, stride, dilate);
    compile_write_ofmap(&dest, dest_size, o_name, ofmap_full_addr, i_nchw[3], w_mcrs[2], w_mcrs[3], o_rows, o_cols, out_word_size);


}


#if 0
int 
main(int argc, char **argv)
{
    std::string i_name, f_name, o_name;
    if (argc < 4) {
        i_name = "/home/ec2-user/InklingTest/input/ifmaps/i_uint8_1x3x2x2_rand.npy";
        f_name = "/home/ec2-user/InklingTest/input/filters/f_uint8_2x3x1x1_rand.npy";
        o_name = "/home/ec2-user/InklingUT/ofmap.npy";
    } else {
        int i = 1;
        i_name = argv[i++];
        f_name = argv[i++];
        o_name = argv[i++];
    }
    compile(i_name, f_name, o_name);
    return 0;
}
#endif
