#ifndef SEQUENCER_H
#define SEQUENCER_H

#include "sigint.h"
#include <queue>

typedef struct ConvolveArgs{
    addr_t ifmap_addr, filter_addr, ofmap_addr;
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


class EdgeSignalsInstruction : public Instruction {
     public:
        EdgeSignalsInstruction(EdgeSignals _es) : es(_es) {};
        ~EdgeSignalsInstruction() {} ;
        void execute(Sequencer *seq);
        void        dump(bool header);
     private:
        EdgeSignals es;
};

typedef struct LdWeightsArgs {
    addr_t  weight_addr;
    addr_t  weight_stride;
    addr_t  weight_step;
    uint8_t weight_columns;
    uint8_t weight_rows;
    ARBPRECTYPE weight_dtype;
} LdWeightsArgs;

class LdWeights : public Instruction {
    public:
        LdWeights(const LdWeightsArgs &args);
        ~LdWeights();
        void execute(Sequencer *seq);
    private:
        LdWeightsArgs args;
};

typedef struct MatMulArgs {
    addr_t  ifmap_addr;
    addr_t  ifmap_stride;
    size_t  ifmap_step;
    addr_t  ifmap_box_width;
    uint8_t ifmap_box_height;
    uint8_t ifmap_box_stride;
    ARBPRECTYPE ifmap_dtype;
    addr_t ofmap_addr;
    size_t ofmap_step;
    addr_t ofmap_stride;
    ARBPRECTYPE psum_dtype;
    int         num_ifmaps;
    int num_ofmaps;
    bool psum_start;
} MatMulArgs;

class MatMul : public Instruction {
    public:
        MatMul(const MatMulArgs &args);
        ~MatMul();
        void execute(Sequencer *seq);
    private:
        MatMulArgs args;
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
        size_t      weight_step;
        uint8_t     weight_columns;

        /* matmul */
        addr_t      ifmap_base;
        size_t      ifmap_step;
        size_t      ifmap_eol_stride;
        uint8_t     ifmap_x_num;
        uint8_t     ifmap_y_num;
        uint8_t     ifmap_x_cnt;
        uint8_t     ifmap_y_cnt;
        size_t      ofmap_step;

        /*  misc */
        bool        raw_signal;


    private:
        tick_t   clock;
        std::queue<Instruction *> uop;
        void dump_es(const EdgeSignals &es, bool header);

};

#endif  
