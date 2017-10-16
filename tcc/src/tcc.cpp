#include "tcc.h"
#include "tpb_isa.h"
#include "uarch_cfg.h"
#include <assert.h>
#include <string>
#include "string.h"
#include "math.h"


#define N_FLAG 1
#define S_FLAG 1 << 1
#define E_FLAG 1 << 2
#define W_FLAG 1 << 3

#define TILE_SIZE 16

enum NSEW {N=0, S, E, W, NUM_NSEW};


#define UNUSED(X) (void)(X)

#define PUSH(FPTR, INS) \
    assert(sizeof(INS) <= INSTRUCTION_NBYTES); \
    fwrite(&INS, sizeof(INS), 1, FPTR); \
    fseek(FPTR, INSTRUCTION_NBYTES - sizeof(INS) - 1, SEEK_CUR); \
    fputc('\0', FPTR);

uint8_t
get_tile_type(const uint8_t row, const uint8_t col, 
        const uint8_t n_rows, const uint8_t n_cols) {
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
        const addr_t ifmap_full_addr, const uint64_t idim[4],
        const addr_t filter_full_addr, const uint64_t wdim[4],
        const addr_t psum_addr,
        const ARBPRECTYPE dtype,
        const uint8_t fmap_num[2],
        const uint8_t pads[NUM_NSEW], 
        const uint8_t strides[2])
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
    assert(num_rows && num_cols);

    addr_t dsize = sizeofArbPrecType(dtype);
    addr_t weight_step;
    LDWEIGHTS weight_args = {0};
    MATMUL    matmul_args = {0};

    /* weight args */
    weight_args.opcode = LDWEIGHTS_OPC;
    weight_args.dtype = dtype;
    weight_args.last_row = num_rows - 1;
    weight_args.x_step = dsize;
    weight_args.x_num = o_channels;
    weight_args.y_step = dsize * o_channels;
    weight_args.y_num = 1;
    weight_args.address = filter_full_addr;
    weight_step = weight_args.y_num * weight_args.y_step * dsize;

    /* matmul args */
    matmul_args.opcode = MATMUL_OPC;
    matmul_args.fmap_x_step = strides[0];
    matmul_args.fmap_y_step = strides[1] * i_cols;
    matmul_args.dtype = dtype;
    matmul_args.psum_start_addr = psum_addr; /* b/c we specify padding as arg */
    matmul_args.last_row = num_rows - 1;
    matmul_args.last_col = num_cols - 1;



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
        const addr_t src_addr,
        const addr_t dst_addr,
        const size_t tile_sz_x, const size_t tile_sz_y,
        const uint64_t o_dims[4],
        const ARBPRECTYPE pool_dtype)
{
    POOL      pool_args = {0};
    uint64_t o_cols = o_dims[3];
    uint64_t o_ch    = o_dims[1];

    /* pool args */
    pool_args.opcode = POOL_OPC;
    pool_args.pool_func = IDENTITY_POOL;
    pool_args.in_dtype     = pool_dtype;

    pool_args.src_x_step= 1;
    pool_args.src_y_step= 1;
    pool_args.src_x_num = tile_sz_x * tile_sz_y;
    pool_args.src_y_num = 1;
    pool_args.src_start_addr = src_addr;

    pool_args.dst_x_step = 1;
    pool_args.dst_y_step = o_cols;
    pool_args.dst_x_num = tile_sz_x;
    pool_args.dst_y_num = tile_sz_y;
    pool_args.dst_start_addr = dst_addr;
    pool_args.max_partition = o_ch - 1;


    pool_args.str_x_step = 1;
    pool_args.str_y_step = 1;
    pool_args.str_x_num = 1;
    pool_args.str_y_num = 1;
    PUSH(fptr, pool_args);

}

void compile_read_ifmap(FILE *fptr,
        addr_t ifmap_full_addr, const char *i_name, const char *numpy_layout)
{
    SIM_RDIFMAP args = {0};

    assert(!strcmp(numpy_layout, "NCHW") && "only NCHW currently supported");

    args.opcode = SIM_RDIFMAP_OPC;
    args.address = ifmap_full_addr;
    strcpy(args.fname, i_name);
    
    PUSH(fptr, args);
}

void compile_read_filter(FILE *fptr,
        addr_t filter_full_addr, const char *i_name, const char *numpy_layout)
{
    SIM_RDFILTER args = {0};

    assert(!strcmp(numpy_layout, "MCRS") && "only MCRS currently supported");

    args.opcode = SIM_RDFILTER_OPC;
    args.address = filter_full_addr;
    strcpy(args.fname, i_name);
    
    PUSH(fptr, args);
}

void compile_write_ofmap(FILE *fptr,
        const char *o_name, const addr_t addr,
        const uint64_t o_dims[4],
        const size_t word_size)
{
    SIM_WROFMAP args = {0};
    args.opcode = SIM_WROFMAP_OPC;
    args.address = addr;
    for (int i = 0; i < 4; i++) {
        args.dims[i] = o_dims[i];
    }
    args.word_size = word_size;
    strcpy(args.fname, o_name);
    
    PUSH(fptr, args);
}

struct Tile_Dims {
    uint8_t rows;
    uint8_t cols;
    size_t  x_whole;
    size_t  y_whole;
    size_t  x_partial;
    size_t  y_partial;
    Tile_Dims(uint8_t o_rows, uint8_t o_cols) {
        rows = ceil((float) o_rows / TILE_SIZE);
        cols = ceil((float) o_cols / TILE_SIZE);

        /* tile args */
        x_whole = o_cols > TILE_SIZE ? TILE_SIZE : o_cols;
        y_whole = o_rows > TILE_SIZE ? TILE_SIZE : o_rows;
        x_partial = cols % TILE_SIZE ? o_cols % TILE_SIZE : x_whole;
        y_partial = o_rows % TILE_SIZE ? o_rows % TILE_SIZE : y_whole;
    };
    size_t area() {
        return x_whole * y_whole;
    }
    void get_info(int i, int j, uint8_t *tt,
            uint8_t *row_offset, uint8_t *col_offset,
            size_t *tile_sz_x, size_t *tile_sz_y) {
        *tt = get_tile_type(i, j, rows, cols);
        *row_offset = (i * TILE_SIZE); 
        *col_offset = (j * TILE_SIZE);
        *tile_sz_x = *tt & E_FLAG ? x_partial : x_whole;
        *tile_sz_y = *tt & S_FLAG ? y_partial : y_whole;

    }
    addr_t addr(unsigned int i, unsigned int j) {
        addr_t lin_addr = 0;
        uint8_t row_offset, col_offset;
        size_t  tile_sz_x, tile_sz_y;
        uint8_t tt = get_tile_type(i, j, rows, cols);
        /* there is easier ways to do this */
        for (unsigned int ii = 0; ii < i; ii++) {
            for (unsigned int jj = 0; jj < j; jj++) {
                get_info(ii, jj, &tt, &row_offset, &col_offset, 
                        &tile_sz_x, &tile_sz_y);
                lin_addr += tile_sz_x * tile_sz_y;
            }
        }
        return lin_addr;
    }

};

void
compile_convolve(FILE *fptr,
        const addr_t ifmap_full_addr, const uint64_t idim[4],
        const addr_t filter_full_addr, const uint64_t wdim[4],
        const addr_t ofmap_full_addr, uint64_t o_dims[4],
        const ARBPRECTYPE dtype,
        const uint8_t padding[2], const uint8_t striding[2], 
        const uint8_t dilate[2])
{
    UNUSED(dilate);
    uint8_t f_rows = wdim[2];
    uint8_t f_cols = wdim[3];
    uint8_t i_rows = idim[2];
    uint8_t i_cols = idim[3];
    uint8_t p_rows = padding[0];
    uint8_t p_cols = padding[1];
    uint8_t d_rows = 0; // TODO: implement
    uint8_t d_cols = 0; // TODO: implement
    uint8_t s_rows = striding[1];
    uint8_t s_cols = striding[0];
    uint8_t f_rows_dilated = f_rows + (f_rows - 1) * d_rows;
    uint8_t f_cols_dilated = f_cols + (f_cols - 1) * d_cols;
    /* for output dim derivation, see https://arxiv.org/pdf/1603.07285.pdf */
    o_dims[0] = idim[0];
    o_dims[1] = wdim[0];
    o_dims[2] = (i_rows - f_rows_dilated + 2 * p_rows) / s_rows + 1;
    o_dims[3] = (i_cols - f_cols_dilated + 2 * p_cols) / s_cols + 1;
    uint8_t o_rows = o_dims[2];
    uint8_t o_cols = o_dims[3];
    addr_t dsize = sizeofArbPrecType(dtype);
    Tile_Dims tile_dims = Tile_Dims(o_rows, o_cols);

    /* tile args */

    uint8_t pads[NUM_NSEW];
    uint8_t fmap_num[2];
    addr_t matmul_addr;
    ARBPRECTYPE pool_dtype = get_upcast(dtype);
    addr_t pool_dst_addr = ofmap_full_addr;
    size_t pool_dsize = sizeofArbPrecType(pool_dtype);
    //  FIXME: assuming psum_base == sb_size
    addr_t psum_addr = MMAP_PSUM_BASE;
    uint8_t row_offset, col_offset;
    uint8_t tt;
    size_t tile_sz_x, tile_sz_y;

    /* go through each tile */
    /* write tile iterators? */
    for (uint8_t i = 0; i < tile_dims.rows; i++) {
        for (uint8_t j = 0; j < tile_dims.cols; j++) {
            /* go through each weight in the filter and apply it to the ofmap
             * pixels it operates on */
            tile_dims.get_info(i, j, &tt, &row_offset, &col_offset, 
                    &tile_sz_x, &tile_sz_y);
            /* adjust input tile size for pads that we don't feed */
            pads[W] = tt & W_FLAG ? p_cols : 0;
            pads[E] = tt & E_FLAG ? p_cols : 0;
            pads[N] = tt & N_FLAG ? p_rows : 0;
            pads[S] = tt & S_FLAG ? p_rows : 0;
            fmap_num[0] = tile_sz_x - pads[W] - pads[E];
            fmap_num[1] = tile_sz_y - pads[N] - pads[S];
    
            matmul_addr = ifmap_full_addr + (row_offset * i_cols + col_offset) *
                dsize;

            _convolve_tile(fptr,
                    matmul_addr, idim,
                    filter_full_addr, wdim,
                    psum_addr,
                    dtype, fmap_num, pads, striding);
            _pool_tile(fptr,
                    psum_addr, pool_dst_addr,
                    tile_sz_x, tile_sz_y,
                    o_dims, pool_dtype);

            pool_dst_addr = ofmap_full_addr + tile_dims.addr(i, j) * pool_dsize;
            psum_addr += (1 << BANK_BITS);
            if (psum_addr >= (1 << BANK_BITS) * 
                    (1 << BANKS_PER_PARTITION_BITS)) {
                psum_addr = MMAP_PSUM_BASE; 
            }
        }
    }

}



void
compile_pool(FILE *fptr,
        const addr_t ifmap_addr, const uint64_t ifmap_dims[4],
        const uint64_t kernel_dims[4],
        const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output */
        const uint64_t stride_dims[4],
		ARBPRECTYPE dtype,
        POOLFUNC pool_func)
{
	uint64_t s_cols = stride_dims[3];
	uint64_t s_rows = stride_dims[2];
	uint64_t s_ch   = stride_dims[1];
	uint64_t s_n    = stride_dims[0];
	uint64_t k_cols = kernel_dims[3];
	uint64_t k_rows = kernel_dims[2];
	uint64_t k_ch   = kernel_dims[1];
	uint64_t k_n    = kernel_dims[0];
	uint64_t i_cols = ifmap_dims[3];
	//uint64_t i_rows = ifmap_dims[2];
	//uint64_t i_ch   = ifmap_dims[1];
	uint64_t i_n    = ifmap_dims[0];
	uint64_t &o_cols = ofmap_dims[3];
	uint64_t &o_rows = ofmap_dims[2];
	uint64_t &o_ch   = ofmap_dims[1];
	//uint64_t &o_n    = ofmap_dims[0];
    POOL      pool_args = {0};
	addr_t        src_addr = ifmap_addr;
	addr_t        dst_addr = ofmap_addr;
    addr_t dsize = sizeofArbPrecType(dtype);
    for (int i = 0; i < 4; i++) {
        ofmap_dims[i] = floor((ifmap_dims[i] - kernel_dims[i])/stride_dims[i]) +
            1;
    }
	
	assert(s_n == 1 && "TBD: pooling across channels/batches");
	assert(i_n == 1 && "TBD: batches");
	assert(k_ch == 1 && "TBD: Pooling across channels");
	assert(k_n == 1 && "TBD: Pooling across inputs");
	assert(o_ch <= 64 && "Only 64 pooling engines, supported");
            
	for (unsigned int i = 0; i < s_n; i++) {
        for (unsigned int j = 0; j < s_ch; j++) {
            /* pool args */
            pool_args.opcode = POOL_OPC;
            pool_args.pool_func = pool_func;
            pool_args.in_dtype     = dtype;
            pool_args.src_start_addr = src_addr;
            pool_args.src_x_step = 1;
            pool_args.src_y_step = i_cols * dsize;
            pool_args.src_x_num = k_cols;
            pool_args.src_y_num = k_rows;
            pool_args.dst_start_addr = dst_addr;
            pool_args.str_x_step = s_cols;
            pool_args.str_y_step = i_cols * s_rows;
            pool_args.str_x_num = o_cols;
            pool_args.str_y_num = o_rows;
            pool_args.max_partition = o_ch - 1;

            /* Pool  */
            pool_args.dst_x_step = 1;
            pool_args.dst_y_step = o_cols;
            pool_args.dst_x_num = o_cols;
            pool_args.dst_y_num = o_rows;
            pool_args.dst_start_addr = dst_addr;
            PUSH(fptr, pool_args);
        }
    }

}

#if 0
void
compile_resadd(FILE *out_binary,
        const addr_t lhs_addr, 
        const addr_t rhs_addr, 
        const uint64_t dims[4],
        const ARBPRECTYPE dtype)
{
	uint64_t cols = dims[3];
	uint64_t rows = dims[2];
	uint64_t ch   = dims[1];
	uint64_t n    = dims[0];
    uint8_t tile_rows,  tile_cols;
    size_t  tile_x_whole, tile_y_whole;
    size_t  tile_x_partial, tile_y_partial;
    uint8_t tt;

    Tile_Dims tile_dims = Tile_Dims(rows, cols)

    /* go through each tile */
    for (uint8_t i = 0; i < tile_rows; i++) {
        for (uint8_t j = 0; j < tile_cols; j++) {
        }
    }


}
#endif

	
