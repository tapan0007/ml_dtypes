#ifndef _TILE_H
#define _TILE_H

#include <math.h>

#define N_FLAG 1
#define S_FLAG 1 << 1
#define E_FLAG 1 << 2
#define W_FLAG 1 << 3

#define TILE_SIZE 16

enum NSEW {N=0, S, E, W, NUM_NSEW};


class Tile_Dims {
    public:
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
            x_partial = o_cols % TILE_SIZE ? o_cols % TILE_SIZE : x_whole;
            y_partial = o_rows % TILE_SIZE ? o_rows % TILE_SIZE : y_whole;
        };
        void get_info(int i, int j, uint8_t *tt,
                uint8_t *row_offset, uint8_t *col_offset,
                size_t *tile_sz_x, size_t *tile_sz_y) {
            *tt = get_tile_type(i, j, rows, cols);
            *row_offset = (i * TILE_SIZE); 
            *col_offset = (j * TILE_SIZE);
            *tile_sz_x = *tt & E_FLAG ? x_partial : x_whole;
            *tile_sz_y = *tt & S_FLAG ? y_partial : y_whole;

        }
        addr_t flatten_coord(unsigned int i, unsigned int j) {
            addr_t lin_addr = 0;
            uint8_t row_offset, col_offset;
            size_t  tile_sz_x, tile_sz_y;
            uint8_t tt = get_tile_type(i, j, rows, cols);
            /* there is easier ways to do this */
            for (unsigned int ii = 0; ii < i; ii++) {
                for (unsigned int jj = 0; jj < cols; jj++) {
                    get_info(ii, jj, &tt, &row_offset, &col_offset, 
                            &tile_sz_x, &tile_sz_y);
                    lin_addr += tile_sz_x * tile_sz_y;
                }
            }
            for (unsigned int jj = 0; jj < j; jj++) {
                get_info(i, jj, &tt, &row_offset, &col_offset, 
                        &tile_sz_x, &tile_sz_y);
                lin_addr += tile_sz_x;
            }
            return lin_addr;
        }
    private:
        uint8_t get_tile_type(const uint8_t row, const uint8_t col, 
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


};

#endif


	
