#ifndef SEQUENCER_H
#define SEQUENCER_H

#include "sigint.h"
#include <queue>

class Instruction {
    public:
        Instruction() {}
        ~Instruction() {}
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


class Sequencer : public EdgeInterface, public PoolInterface  {
    public:
        void connect_uopfeed(UopFeedInterface *feed);
        void step();
        EdgeSignals pull_edge();
        PoolSignals pull_pool();
        bool pad_valid(uint8_t, uint8_t);
        bool synch();
        bool done();
        void dump();

        /* internal state */
        EdgeSignals es;
        PoolSignals ps;

        /* weight */
        addr_t      weight_base;
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
        uint8_t     pad[NUM_NSEW];
        uint8_t     pad_cnt[NUM_NSEW];
        uint8_t     pad_run[NUM_NSEW];
        uint8_t     pad_num[NUM_NSEW];
        enum NSEW   pad_dir;

        /* pool */
        /* pifmaps are padded ifmaps */
        uint8_t     pifmap_x_num;
        uint8_t     pifmap_y_num;
        uint8_t     pifmap_x_cnt;
        uint8_t     pifmap_y_cnt;
        bool        pool_valid;
        uint64_t    pool_timer;
        addr_t      pool_src_base;
        size_t      pool_src_x_step;
        size_t      pool_src_y_step;
        size_t      pool_src_z_step;
        size_t      pool_src_x_cnt;
        size_t      pool_src_y_cnt;
        size_t      pool_src_z_cnt;
        size_t      pool_src_x_num;
        size_t      pool_src_y_num;
        size_t      pool_src_z_num;

        addr_t      pool_dst_base;
        size_t      pool_dst_x_step;
        size_t      pool_dst_y_step;
        size_t      pool_dst_x_cnt;
        size_t      pool_dst_y_cnt;
        size_t      pool_dst_x_num;
        size_t      pool_dst_y_num;

        /*  misc */
        bool        raw_signal;


    private:
        tick_t   clock;
        UopFeedInterface *feed;
        void step_edgesignal();
        void step_poolsignal();
        void dump_es(const EdgeSignals &es, bool header);
        void increment_and_rollover(uint8_t &cnt, uint8_t num, 
                uint8_t &rollover);


};

#endif  
