#ifndef SEQUENCER_H
#define SEQUENCER_H

#include "sigint.h"
#include <queue>

class Instruction {
    public:
        virtual ~Instruction()  {};
        virtual void execute(void *seq) = 0;
        virtual void dump(bool header) {(void)header; std::cout << "not implemented";}
};

template <class T>
class DynamicInstruction : public Instruction {
     public:
        DynamicInstruction(T _args) : args(_args) {};
        ~DynamicInstruction() {} ;
        void execute(void *seq);
        void        dump(bool header) {(void)header; std::cout << "not implemented";}
     private:
        T args;
};

enum NSEW {N=0, S, E, W, NUM_NSEW};

class Memory;

class Sequencer : public EdgeInterface, public PoolInterface  {
    public:
        Sequencer(Memory *_mem) : mem(_mem) {}
        void connect_uopfeed(UopFeedInterface *feed);
        void step();
        EdgeSignals pull_edge();
        PoolSignals pull_pool();
        bool pad_valid(uint8_t, uint8_t);
        bool synch();
        bool done();
        void dump();

        /* internal state */
        EdgeSignals es = {0};
        PoolSignals ps = {0};

        /* weight */
        addr_t      weight_base   = 0x0;
        size_t      weight_x_step = 0;
        size_t      weight_y_step = 0;
        uint8_t     weight_x_num  = 0;
        uint8_t     weight_y_num  = 0;
        uint8_t     weight_x_cnt  = 0;
        uint8_t     weight_y_cnt  = 0;
        uint8_t     weight_clamp_countdown = 0;

        /* matmul */
        addr_t      ifmap_base   = 0x0;
        size_t      ifmap_x_step = 0;
        size_t      ifmap_y_step = 0;
        uint8_t     ifmap_x_num = 0;
        uint8_t     ifmap_y_num = 0;
        uint8_t     ifmap_x_cnt = 0;
        uint8_t     ifmap_y_cnt = 0;
        /* pifmaps are padded ifmaps */
        uint8_t     pifmap_x_num = 0;
        uint8_t     pifmap_y_num = 0;
        uint8_t     pifmap_x_cnt = 0;
        uint8_t     pifmap_y_cnt = 0;
        uint8_t     pad[NUM_NSEW] = {0};
        uint8_t     pad_cnt[NUM_NSEW] = {0};
        uint8_t     pad_run[NUM_NSEW] = {0};
        uint8_t     pad_num[NUM_NSEW] = {0};
        enum NSEW   pad_dir;

        /* pool */
        bool        pool_eopools = false;
        addr_t      pool_src_base = 0;
        size_t      pool_str_x_step = 0;
        size_t      pool_str_y_step = 0;
        size_t      pool_str_x_cnt = 0;
        size_t      pool_str_y_cnt = 0;
        size_t      pool_str_x_num = 0;
        size_t      pool_str_y_num = 0;

        size_t      pool_src_x_step = 0;
        size_t      pool_src_y_step = 0;
        size_t      pool_src_x_cnt = 0;
        size_t      pool_src_y_cnt = 0;
        size_t      pool_src_x_num = 0;
        size_t      pool_src_y_num = 0;

        addr_t      pool_dst_base = 0x0;
        size_t      pool_dst_x_step = 0;
        size_t      pool_dst_y_step = 0;
        size_t      pool_dst_x_cnt = 0;
        size_t      pool_dst_y_cnt = 0;
        size_t      pool_dst_x_num = 0;
        size_t      pool_dst_y_num = 0;

        /*  misc */
        bool        raw_signal = false;

        Memory *mem = nullptr;

    private:
        uint64_t   clock = 0;
        UopFeedInterface *feed = nullptr;
        void step_edgesignal();
        void step_poolsignal();
        void dump_es(const EdgeSignals &es, bool header);
        void increment_and_rollover(uint8_t &cnt, uint8_t num,
                uint8_t &rollover);


};

#endif
