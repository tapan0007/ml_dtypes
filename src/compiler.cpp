#include "compiler.h"
#include "isa.h"
#include "internal_isa.h"
#include <assert.h>
#include <string>
#include "string.h"
#include <assert.h>


uint8_t
get_tile_type(uint8_t row, uint8_t col, 
        uint8_t n_rows, uint8_t n_cols) {
    uint8_t tt = 0;
    if (row == 0) {
        tt |= N_FLAG;
    }
    if (row == n_rows - 1) {
        tt |= S_FLAG;
    }
    if (col == 0) {
        tt |= W_FLAG;
    }
    if (col == n_cols - 1) {
        tt |= E_FLAG;
    }
    return tt;
}

#define UNUSED(X) (void)(X)

#define IBUF_SIZE 64*1024

#define PUSH(DEST, DEST_SIZE, INS) \
    assert(sizeof(INS) <= INSTRUCTION_NBYTES); \
    memcpy(*DEST, &INS, sizeof(INS)); \
    *DEST += INSTRUCTION_NBYTES; \
    DEST_SIZE += INSTRUCTION_NBYTES;

void compile_read_ifmap(void **v_dest, size_t &dest_size, 
        addr_t ifmap_full_addr, char *i_name)
{
    RdIfmapArgs args;
    char **dest = (char **)(v_dest);

    args.opcode = INTERNAL_OPCODE_READ_IFMAP;
    args.address = ifmap_full_addr;
    strcpy(args.fname, i_name);
    
    PUSH(dest, dest_size, args);
}

void compile_read_filter(void **v_dest, size_t &dest_size, 
        addr_t filter_full_addr, char *i_name)
{
    RdFilterArgs args;
    char **dest = (char **)(v_dest);

    args.opcode = INTERNAL_OPCODE_READ_FILTER;
    args.address = filter_full_addr;
    strcpy(args.fname, i_name);
    
    PUSH(dest, dest_size, args);
}

void compile_write_ofmap(void **v_dest, size_t &dest_size, 
        char *o_name, addr_t addr, uint64_t i_n, 
        uint64_t w_c,  uint64_t w_m,
        uint64_t o_rows, uint64_t o_cols, 
        size_t word_size)
{
    char **dest = (char **)(v_dest);
    WrOfmapArgs args;
    args.opcode = INTERNAL_OPCODE_WRITE_OFMAP;
    args.address = addr;
    args.i_n = i_n;
    args.w_c = w_c;
    args.w_m = w_m;
    args.o_rows = o_rows;
    args.o_cols = o_cols;
    args.word_size = word_size;
    strcpy(args.fname, o_name);
    
    PUSH(dest, dest_size, args);
}


void
compile_convolve(void **v_dest, size_t &dest_size, 
        uint64_t &o_rows, uint64_t &o_cols,
        uint64_t ofmap_full_addr,
        uint64_t ifmap_full_addr, uint64_t idim[4],
        uint64_t filter_full_addr, uint64_t wdim[4],
        ARBPRECTYPE dtype,
        uint8_t padding[2], uint8_t stride[2], uint8_t dilate[2])
{
    UNUSED(stride);
    UNUSED(dilate);
    char **dest = (char **)(v_dest);
    uint8_t f_rows = wdim[2];
    uint8_t f_cols = wdim[3];
    uint8_t i_rows = idim[2];
    uint8_t i_cols = idim[3];
    uint8_t p_rows = padding[0];
    uint8_t p_cols = padding[1];
    uint8_t d_rows = 0; // TODO: implement
    uint8_t d_cols = 0; // TODO: implement
    uint8_t s_rows = 1; // TODO: implement
    uint8_t s_cols = 1; // TODO: implement
    uint8_t f_rows_dilated = f_rows + (f_rows - 1) * d_rows;
    uint8_t f_cols_dilated = f_cols + (f_cols - 1) * d_cols;
    /* for output dim derivation, see https://arxiv.org/pdf/1603.07285.pdf */
    o_rows = (i_rows - f_rows_dilated + 2 * p_rows) / s_rows + 1;
    o_cols = (i_cols - f_cols_dilated + 2 * p_cols) / s_cols + 1;
    uint8_t num_cols = wdim[1];
    uint8_t num_rows = idim[1];
    uint8_t tile_rows = ceil((float) o_rows / Constants::tile_size);
    uint8_t tile_cols = ceil((float) o_rows / Constants::tile_size);
    addr_t dsize = sizeofArbPrecType(dtype);
    addr_t weight_step;
    LdWeightsArgs weight_args;
    MatMulArgs    matmul_args;
    PoolArgs      pool_args;

    /* weight args */
    weight_args.opcode = LDWEIGHTS;
    weight_args.dtype = dtype;
    weight_args.num = num_rows;
    weight_args.x_step = dsize;
    weight_args.x_num = wdim[0];
    weight_args.y_step = dsize * wdim[0];
    weight_args.y_num = 1;
    weight_args.address = filter_full_addr;
    weight_step = weight_args.y_num * weight_args.y_step * dsize;

    /* matmul args */
    matmul_args.opcode = MATMUL;
    matmul_args.fmap_x_step = 1;
    matmul_args.fmap_y_step = i_cols;
    matmul_args.dtype = dtype;
    matmul_args.psum_start_addr = 0; /* b/c we specify padding as arg */
    matmul_args.num_rows = num_rows;
    matmul_args.num_cols = num_cols;

    /* pool args */
    ARBPRECTYPE pool_dtype = get_upcast(dtype);
    addr_t pool_dsize = sizeofArbPrecType(pool_dtype);
    pool_args.opcode = POOL;
    pool_args.pool_func = IDENTITY_POOL;
    pool_args.in_dtype     = pool_dtype;
    pool_args.src_start_addr = MMAP_PSUM_BASE;
    pool_args.src_x_step= 1;
    pool_args.src_y_step= 1;
    pool_args.src_z_step= 1;
    pool_args.src_y_num = 1;
    pool_args.src_z_num = 1;
    pool_args.dst_x_step = 1;
    pool_args.dst_y_step = o_cols;
    pool_args.dst_start_addr = ofmap_full_addr;
    pool_args.dst_num = wdim[0];

    /* tile args */
    size_t tile_x_dim = Constants::tile_size;
    size_t tile_y_dim = Constants::tile_size;
    size_t tile_x_whole = o_cols > tile_x_dim ? tile_x_dim : o_cols;
    size_t tile_y_whole = o_rows > tile_y_dim ? tile_y_dim : o_rows;
    size_t tile_x_partial = 
        o_cols % tile_x_dim ? o_cols % tile_x_dim : tile_x_whole;
    size_t tile_y_partial = 
        o_rows % tile_y_dim ? o_rows % tile_y_dim : tile_y_whole;

    uint8_t row_offset, col_offset;
    size_t tile_sz_x, tile_sz_y;
    pool_args.dst_start_addr = ofmap_full_addr;
    uint8_t curr_weight = 0;
    uint8_t tt;
    uint8_t r_adj, s_adj;

    /* go through each tile */
    for (uint8_t i = 0; i < tile_rows; i++) {
        for (uint8_t j = 0; j < tile_cols; j++) {
            curr_weight = 0;

            /* load weights ahead of first convolution for first filter! */
            PUSH(dest, dest_size, weight_args);

            /* go through each weight in the filter and apply it to the ofmap
             * pixels it operates on */
            tt = get_tile_type(i, j, tile_rows, tile_cols);
            tile_sz_x = tt & E_FLAG ? tile_x_partial : tile_x_whole;
            tile_sz_y = tt & S_FLAG ? tile_y_partial : tile_y_whole;

            matmul_args.w_pad = tt & W_FLAG ? p_cols : 0;
            matmul_args.e_pad = tt & E_FLAG ? p_cols : 0;
            matmul_args.n_pad = tt & N_FLAG ? p_rows : 0;
            matmul_args.s_pad = tt & S_FLAG ? p_rows : 0;
            matmul_args.fmap_x_num = tile_sz_x - matmul_args.w_pad -
                matmul_args.e_pad;
            matmul_args.fmap_y_num = tile_sz_y - matmul_args.n_pad - 
                matmul_args.s_pad;
            for (uint8_t r = 0; r <  f_rows; r++) {
                for (uint8_t s = 0; s < f_cols; s++, curr_weight++) {
                    /* matmul arguments and PUSH */
                    r_adj = (r >= p_rows) * (r - p_rows);
                    s_adj = (s >= p_cols) * (s - p_cols);
                    row_offset = (r_adj + i * tile_y_dim); 
                    col_offset = (s_adj + j * tile_x_dim);
                    matmul_args.fmap_start_addr = ifmap_full_addr +
                        (row_offset * i_cols + col_offset) * dsize;

                    matmul_args.start_tensor_calc = (r == 0 && s == 0);
                    matmul_args.stop_tensor_calc = (curr_weight == 
                            (f_rows * f_cols - 1));
                    PUSH(dest, dest_size, matmul_args);

                    /* adjust weight address and LdWeights if not at end*/
                    if ((r == f_rows - 1) && (s == f_cols - 1)) {
                        weight_args.address = filter_full_addr;
                    } else {
                        weight_args.address += weight_step;
                        PUSH(dest, dest_size, weight_args);
                    }
                }
            }
            /* Pool  */
            pool_args.src_x_num = tile_sz_x * tile_sz_y;
            pool_args.dst_x_num = tile_sz_x;
            pool_args.dst_y_num = tile_sz_y;
            PUSH(dest, dest_size, pool_args);
            if (j < (tile_cols - 1)) { /* non-edge tile */
                pool_args.dst_start_addr += tile_sz_x * pool_dsize;
            } else { /* edge tile */
                pool_args.dst_start_addr = ofmap_full_addr +
                    (j * tile_x_whole * tile_y_whole + 
                     tile_sz_x * tile_sz_y) *  pool_dsize;
            }
        }
    }

}

