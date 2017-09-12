#ifndef ISA_H
#define ISA_H


typedef struct LdWeightsArgs {
   uint64_t    address               : 32;
   uint64_t    dtype                 : 8;
   uint64_t    x_step                : 12;
   uint64_t    x_num                 : 9;
   uint64_t    y_step                : 12;
   uint64_t    y_num                 : 9;
   uint64_t    num_rows              : 7;
} LdWeightsArgs;
typedef struct MatMulArgs {
   uint64_t    dequant_table_idx          : 8;
   uint64_t    quant_data_size          : 4;
   uint64_t    dequant_data_size          : 8;
   uint64_t    start_tensor_calc          : 1;
   uint64_t    stop_tensor_calc          : 1;
   uint64_t    reserved              : 3;
   uint64_t    dtype                 : 8;
   uint64_t    fmap_start_addr          : 32;
   uint64_t    fmap_x_step           : 12;
   uint64_t    fmap_x_num            : 9;
   uint64_t    fmap_y_step           : 12;
   uint64_t    fmap_y_num            : 9;
   uint64_t    num_rows              : 7;
   uint64_t    n_pad                 : 4;
   uint64_t    w_pad                 : 4;
   uint64_t    e_pad                 : 4;
   uint64_t    s_pad                 : 4;
   uint64_t    psum_start_addr          : 32;
   uint64_t    num_cols              : 6;
   uint64_t    psum_step             : 8;
   uint64_t    event_func            : 4;
   uint64_t    event_id              : 8;
} MatMulArgs;
#endif
