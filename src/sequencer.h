#ifndef SEQUENCER_H
#define SEQUENCER_H

#include "sigint.h"
#include <queue>

typedef struct ConvolveArgs{
    ADDR_UNION(ifmap);
    ADDR_UNION(filter);
    ARBPRECTYPE weight_dtype;
    int i_n, i_c, i_h, i_w;
    int w_c, w_m, w_r, w_s;

} ConvolveArgs;

class Sequencer;
class Instruction {
    public:
        Instruction() {}
        ~Instruction() {}
        virtual void execute(Sequencer *seq) = 0;
        virtual void dump(bool header) {(void)header; std::cout << "not implemented";}
};

template <class T>
class DynamicInstruction : public Instruction {
     public:
        DynamicInstruction(T _args) : args(_args) {};
        ~DynamicInstruction() {} ;
        void execute(Sequencer *seq);
        void        dump(bool header) {(void)header; std::cout << "not implemented";}
     private:
        T args;
};

typedef struct LdWeightsArgs {
    ADDR_UNION(weight);
    uint64_t    dtype          : Constants::type_bits;
    addr_t      x_step         : Constants::bank_bits - 1;
    uint64_t    x_num_elements : 8;
    addr_t      y_step         : Constants::bank_bits - 1;
    uint64_t    y_num_elements : 8;
    uint64_t    num_rows : Constants::row_bits;
    uint64_t    num_cols : Constants::column_bits;
} LdWeightsArgs;

typedef struct MatMulArgs {
    uint64_t    psum_start  : 1;
    uint64_t    psum_end    : 1;
    uint64_t    dtype       : Constants::type_bits;
    ADDR_UNION(ifmap);
    addr_t      x_step         : Constants::bank_bits - 1;
    uint64_t    x_num_elements : 8;
    addr_t      y_step         : Constants::bank_bits - 1;
    uint64_t    y_num_elements : 8;
    uint64_t    psum_dtype : Constants::type_bits;
    uint64_t    num_rows   : Constants::row_bits;
    uint64_t    num_cols   : Constants::column_bits;
} MatMulArgs;


typedef struct PoolArgs {
    uint64_t    pool_func   : 8;
    uint64_t    dtype       : Constants::type_bits;
    ADDR_UNION(src);
    addr_t      src_x_step     : Constants::bank_bits - 1;
    uint64_t    src_x_num_elements : 8;
    addr_t      src_y_step         : Constants::bank_bits - 1;
    uint64_t    src_y_num_elements : 8;
    addr_t      src_z_step         : Constants::bank_bits - 1;
    uint64_t    src_z_num_elements : 8;
    ADDR_UNION(dst);
    addr_t      dst_x_step     : Constants::bank_bits - 1;
    uint64_t    dst_x_num_elements : 8;
    addr_t      dst_y_step         : Constants::bank_bits - 1;
    uint64_t    dst_y_num_elements : 8;
    uint64_t    num_partitions     : Constants::row_bits;
} PoolArgs;

class Pool : public Instruction {
    public:
        Pool(const PoolArgs &args);
        ~Pool();
        void execute(Sequencer *seq);
    private:
        PoolArgs args;
};

class Sequencer : public EdgeInterface  {
    public:
        Sequencer();
        ~Sequencer();
        void step();
        EdgeSignals pull_edge();
        void convolve_static(const ConvolveArgs &args);
        void convolve_dynamic(const ConvolveArgs &args);
        bool synch();
        bool done();
        void dump();

        /* internal state */
        EdgeSignals es;

        /* weight */
        size_t      weight_x_step;
        size_t      weight_y_step;
        uint8_t     weight_x_num;
        uint8_t     weight_y_num;
        uint8_t     weight_x_cnt;
        uint8_t     weight_y_cnt;
        uint8_t     weight_clamp_countdown;

        /* matmul */
        addr_t      ifmap_base;
        size_t      ifmap_x_step;
        size_t      ifmap_y_step;
        uint8_t     ifmap_x_num;
        uint8_t     ifmap_y_num;
        uint8_t     ifmap_x_cnt;
        uint8_t     ifmap_y_cnt;

        /*  misc */
        bool        raw_signal;


    private:
        tick_t   clock;
        std::queue<Instruction *> uop;
        void dump_es(const EdgeSignals &es, bool header);

};

#endif  
