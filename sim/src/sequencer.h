#ifndef SEQUENCER_H
#define SEQUENCER_H

#include "sigint.h"
#include <queue>
#include <limits>
#include <cstddef>

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

class PatchIterator {
    public:
        PatchIterator(size_t _ndims) : ndims(_ndims) {
            steps.resize(ndims);
            cnts.resize(ndims + 1);
            nums.resize(ndims + 1);
            cnts[ndims] = 1;
        }
        void init(int *dims, int *step) {
            for (size_t i = 0; i < ndims; i++) {
                nums[i] = dims[i];
                steps[i] = step[i];
            }
            cnts[ndims] = 0;
            nums[ndims] = std::numeric_limits<std::size_t>::max();
        }
        void reset() {
            for (size_t i = 0; i <= ndims; i++) {
                cnts[i] = 0;
            }
        }
        void increment() {
            bool rollover = true;
            for (size_t i = 0; (i <= ndims) && rollover; i++) {
                cnts[i]++;
                rollover = (cnts[i] >= nums[i]);
                if (rollover) {
                    cnts[i] = 0;
                }
            }
        }
        size_t coordinates() {
            size_t coord = 0;
            for (size_t i = 0; i < ndims; i++) {
                coord += cnts[i] * steps[i];
            }
            return coord;
        }
        bool eop() {
            return (cnts[ndims] > 0);
        }
        bool last() {
            bool last = true;
            for (size_t i = 0; i < ndims && last; i++) {
                last &= (cnts[i] == (nums[i] - 1));
            }
            last &= !cnts[ndims];
            return last;
        }
        bool in_range(int *r) {
            bool in_range = true;
            for (size_t i = 0; (i < ndims) && in_range; i++) {
                in_range &= (r[i] >= 0) && ((size_t )r[i] < nums[i]);
            }
            return in_range;
        }
        size_t& operator[](int index) { return cnts[index]; }
    private:
        std::vector<size_t> nums;
        std::vector<size_t> cnts;
        std::vector<int>    steps;
        size_t ndims = {0};
};

enum NSEW {N=0, S, E, W, NUM_NSEW};

class Memory;

class Sequencer : 
    public EdgeInterface, public PoolInterface, public ActivateInterface {
    public:
        Sequencer(Memory *_mem) : weight_pit(1), ifmap_pit(2), pifmap_pit(2),
        pool_src_pit(2), pool_dst_pit(2), pool_str_pit(2), mem(_mem) {}
        void connect_uopfeed(UopFeedInterface *feed);
        void step();
        EdgeSignals pull_edge();
        PoolSignals pull_pool();
        ActivateSignals pull_activate();
        bool pad_valid();
        bool synch();
        bool done();
        void dump();

        /* internal state */
        EdgeSignals es = {0};
        PoolSignals ps = {0};
        ActivateSignals as = {0};

        /* weight */
        addr_t      weight_base   = 0x0;
        PatchIterator weight_pit;
        uint8_t     weight_clamp_countdown = 0;

        /* matmul */
        addr_t        ifmap_base   = 0x0;
        PatchIterator ifmap_pit;
        size_t      ifmap_x_step = 0;
        size_t      ifmap_y_step = 0;

        /* pifmaps are padded ifmaps */
        PatchIterator pifmap_pit;
        uint8_t     pad[NUM_NSEW] = {0};
        uint8_t     pad_cnt[NUM_NSEW] = {0};
        uint8_t     pad_run[NUM_NSEW] = {0};
        uint8_t     pad_num[NUM_NSEW] = {0};
        enum NSEW   pad_dir;

        /* pool */
        addr_t      pool_src_base = 0;
        addr_t      pool_dst_base = 0x0;
        PatchIterator pool_src_pit;
        PatchIterator pool_dst_pit;
        PatchIterator pool_str_pit;;

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
