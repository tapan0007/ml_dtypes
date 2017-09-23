#include "tcc.h"
#include "isa.h"
#include <assert.h>
#include <string>
#include "string.h"
#include "math.h"


#define N_FLAG 1
#define S_FLAG 1 << 1
#define E_FLAG 1 << 2
#define W_FLAG 1 << 3

enum NSEW {N=0, S, E, W, NUM_NSEW};


#define UNUSED(X) (void)(X)

#define PUSH(FPTR, INS) \
    assert(sizeof(INS) <= INSTRUCTION_NBYTES); \
    fwrite(&INS, sizeof(INS), 1, FPTR); \
    fseek(FPTR, INSTRUCTION_NBYTES - sizeof(INS) - 1, SEEK_CUR); \
    fputc('\0', FPTR);

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

void
_convolve_tile(FILE *fptr,
        uint64_t ifmap_full_addr, uint64_t idim[4],
        uint64_t filter_full_addr, uint64_t wdim[4],
        uint64_t psum_addr,
        ARBPRECTYPE dtype,
        uint8_t fmap_num[2],
        uint8_t pads[NUM_NSEW])
{
    /* bounds checking */
    for (int i = 0; i < 4; i++) {
        assert((idim[i] < 256) && "TOO BIG!");
        assert((wdim[i] < 256) && "TOO BIG!");
    }
    uint8_t f_rows = wdim[2];
    uint8_t f_cols = wdim[3];
    uint8_t i_cols = idim[3];
    uint8_t o_channels = wdim[0]; 

    assert(wdim[1] == idim[1]);
    /* for output dim derivation, see https://arxiv.org/pdf/1603.07285.pdf */
    uint64_t num_cols = wdim[0];
    uint64_t num_rows = idim[1];
    assert(num_rows <= (1 << ROW_BITS));
    assert(num_cols <= (1 << COLUMN_BITS));
    assert(num_rows && num_cols);

    addr_t dsize = sizeofArbPrecType(dtype);
    addr_t weight_step;
    LdWeightsArgs weight_args = {0};
    MatMulArgs    matmul_args = {0};

    /* weight args */
    weight_args.opcode = LDWEIGHTS;
    weight_args.dtype = dtype;
    weight_args.max = num_rows - 1;
    weight_args.x_step = dsize;
    weight_args.x_num = o_channels;
    weight_args.y_step = dsize * o_channels;
    weight_args.y_num = 1;
    weight_args.address = filter_full_addr;
    weight_step = weight_args.y_num * weight_args.y_step * dsize;

    /* matmul args */
    matmul_args.opcode = MATMUL;
    matmul_args.fmap_x_step = 1;
    matmul_args.fmap_y_step = i_cols;
    matmul_args.dtype = dtype;
    matmul_args.psum_start_addr = psum_addr; /* b/c we specify padding as arg */
    matmul_args.max_row = num_rows - 1;
    matmul_args.max_col = num_cols - 1;



    /* load weights ahead of first convolution for first filter! */
    PUSH(fptr, weight_args);

    /* go through each weight in the filter and apply it to the ofmap
     * pixels it operates on */
    matmul_args.w_pad = pads[W];
    matmul_args.e_pad = pads[E];
    matmul_args.n_pad = pads[N];
    matmul_args.s_pad = pads[S];
    matmul_args.fmap_x_num = fmap_num[0];
    matmul_args.fmap_y_num = fmap_num[1];

    uint8_t curr_weight = 0;
    uint8_t r_adj, s_adj;
    for (uint8_t r = 0; r <  f_rows; r++) {
        for (uint8_t s = 0; s < f_cols; s++, curr_weight++) {
            /* matmul arguments and PUSH */
            r_adj = (r >= pads[N]) * (r - pads[N]);
            s_adj = (s >= pads[W]) * (s - pads[W]);
            matmul_args.fmap_start_addr = ifmap_full_addr + 
                (r_adj * i_cols + s_adj) * dsize;

            matmul_args.start_tensor_calc = (r == 0 && s == 0);
            matmul_args.stop_tensor_calc = (curr_weight == 
                    (f_rows * f_cols - 1));
            PUSH(fptr, matmul_args);

            /* adjust weight address and LdWeights if not at end*/
            if ((r == f_rows - 1) && (s == f_cols - 1)) {
                weight_args.address = filter_full_addr;
            } else {
                weight_args.address += weight_step;
                PUSH(fptr, weight_args);
            }
        }
    }
}

void
_pool_tile(FILE *fptr,
        uint64_t src_addr,
        uint64_t dst_addr,
        size_t tile_sz_x, size_t tile_sz_y,
        uint64_t o_cols,
        ARBPRECTYPE pool_dtype)
{
    PoolArgs      pool_args = {0};

    /* pool args */
    pool_args.opcode = POOL;
    pool_args.pool_func = IDENTITY_POOL;
    pool_args.in_dtype     = pool_dtype;
    pool_args.src_start_addr = src_addr;
    pool_args.src_x_step= 1;
    pool_args.src_y_step= 1;
    pool_args.src_z_step= 1;
    pool_args.src_y_num = 1;
    pool_args.src_z_num = 1;
    pool_args.dst_x_step = 1;
    pool_args.dst_y_step = o_cols;
    pool_args.dst_start_addr = dst_addr;
    pool_args.dst_num = o_cols;

    /* Pool  */
    pool_args.src_x_num = tile_sz_x * tile_sz_y;
    pool_args.dst_x_num = tile_sz_x;
    pool_args.dst_y_num = tile_sz_y;
    pool_args.dst_start_addr = dst_addr;
    PUSH(fptr, pool_args);

}

void compile_read_ifmap(FILE *fptr,
        addr_t ifmap_full_addr, const char *i_name)
{
    RdIfmapArgs args = {0};

    args.opcode = SIM_RDIFMAP;
    args.address = ifmap_full_addr;
    strcpy(args.fname, i_name);
    
    PUSH(fptr, args);
}

void compile_read_filter(FILE *fptr,
        addr_t filter_full_addr, const char *i_name)
{
    RdFilterArgs args = {0};

    args.opcode = SIM_RDFILTER;
    args.address = filter_full_addr;
    strcpy(args.fname, i_name);
    
    PUSH(fptr, args);
}

void compile_write_ofmap(FILE *fptr,
        const char *o_name, addr_t addr, uint64_t i_n, 
        uint64_t w_c,  uint64_t w_m,
        uint64_t o_rows, uint64_t o_cols, 
        size_t word_size)
{
    WrOfmapArgs args = {0};
    args.opcode = SIM_WROFMAP;
    args.address = addr;
    args.i_n = i_n;
    args.w_c = w_c;
    args.w_m = w_m;
    args.o_rows = o_rows;
    args.o_cols = o_cols;
    args.word_size = word_size;
    strcpy(args.fname, o_name);
    
    PUSH(fptr, args);
}

void
compile_convolve(FILE *fptr,
        uint64_t &o_rows, uint64_t &o_cols,
        uint64_t ofmap_full_addr,
        uint64_t ifmap_full_addr, uint64_t idim[4],
        uint64_t filter_full_addr, uint64_t wdim[4],
        ARBPRECTYPE dtype,
        uint8_t padding[2], uint8_t stride[2], uint8_t dilate[2])
{
    UNUSED(stride);
    UNUSED(dilate);
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
    uint8_t tile_rows = ceil((float) o_rows / TILE_SIZE);
    uint8_t tile_cols = ceil((float) o_rows / TILE_SIZE);
    addr_t dsize = sizeofArbPrecType(dtype);

    /* tile args */
    size_t tile_x_dim = TILE_SIZE;
    size_t tile_y_dim = TILE_SIZE;
    size_t tile_x_whole = o_cols > tile_x_dim ? tile_x_dim : o_cols;
    size_t tile_y_whole = o_rows > tile_y_dim ? tile_y_dim : o_rows;
    size_t tile_x_partial = 
        o_cols % tile_x_dim ? o_cols % tile_x_dim : tile_x_whole;
    size_t tile_y_partial = 
        o_rows % tile_y_dim ? o_rows % tile_y_dim : tile_y_whole;

    uint8_t row_offset, col_offset;
    size_t tile_sz_x, tile_sz_y;
    uint8_t tt;
    uint8_t pads[NUM_NSEW];
    uint8_t fmap_num[2];
    addr_t matmul_addr;
    ARBPRECTYPE pool_dtype = get_upcast(dtype);
    addr_t pool_dst_addr = ofmap_full_addr;
    size_t pool_dsize = sizeofArbPrecType(pool_dtype);
    addr_t psum_addr = MMAP_PSUM_BASE;

    /* go through each tile */
    for (uint8_t i = 0; i < tile_rows; i++) {
        for (uint8_t j = 0; j < tile_cols; j++) {
            /* go through each weight in the filter and apply it to the ofmap
             * pixels it operates on */
            tt = get_tile_type(i, j, tile_rows, tile_cols);
            tile_sz_x = tt & E_FLAG ? tile_x_partial : tile_x_whole;
            tile_sz_y = tt & S_FLAG ? tile_y_partial : tile_y_whole;
            row_offset = (i * tile_y_dim); 
            col_offset = (j * tile_x_dim);
            matmul_addr = ifmap_full_addr + (row_offset * i_cols + col_offset) *
                dsize;
            pads[W] = tt & W_FLAG ? p_cols : 0;
            pads[E] = tt & E_FLAG ? p_cols : 0;
            pads[N] = tt & N_FLAG ? p_rows : 0;
            pads[S] = tt & S_FLAG ? p_rows : 0;
            fmap_num[0] = tile_sz_x - pads[W] - pads[E];
            fmap_num[1] = tile_sz_y - pads[N] - pads[S];
    


            _convolve_tile(fptr,
                    matmul_addr, idim,
                    filter_full_addr, wdim,
                    psum_addr,
                    dtype, fmap_num, pads);
            _pool_tile(fptr,
                    psum_addr, pool_dst_addr,
                    tile_sz_x, tile_sz_y,
                    o_cols, pool_dtype);
            if (j < (tile_cols - 1)) { /* non-edge tile */
                pool_dst_addr += tile_sz_x * pool_dsize;
            } else { /* edge tile */
                pool_dst_addr = ofmap_full_addr +
                    (j * tile_x_whole * tile_y_whole + 
                     tile_sz_x * tile_sz_y) *  pool_dsize;
            }
            psum_addr += (1 << BANK_BITS);
            if (psum_addr >= (1 << BANK_BITS) * 
                    (1 << BANKS_PER_PARTITION_BITS)) {
                psum_addr = MMAP_PSUM_BASE;
            }
        }
    }

}



