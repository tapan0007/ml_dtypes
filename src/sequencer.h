#ifndef SEQUENCER_H
#define SEQUENCER_H

#include "sigint.h"

class Sequencer : public EdgeInterface  {
    public:
        Sequencer();
        ~Sequencer();
        void step();
        EdgeSignals pull_edge();
        void set_clamp(bool truth);
        void set_ifmap_valid(bool truth);
        void set_ifmap_addr(addr_t addr);
        void set_ifmap_stride(addr_t addr);
        void set_weight_addr(addr_t addr);
        void set_weight_stride(addr_t addr);
        void set_weight_valid(bool truth);
        void set_weight_dtype(ARBPRECTYPE type);
        void set_toggle_weight(bool truth);
        void set_psum_start(bool truth);
        void set_psum_end(bool truth);
        void set_psum_id(int id);
        void set_psum_dtype(ARBPRECTYPE type);
        void set_row_countdown(uint8_t count_down);
        void set_column_countdown(uint8_t count_down);
    private:
        uint8_t   row_countdown;
        uint8_t   column_countdown;
        int       psum_id;
        ARBPRECTYPE   psum_dtype;
        bool     psum_start;
        bool     psum_end;
        bool     ifmap_valid;
        addr_t   ifmap_addr;
        addr_t   ifmap_stride;
        bool     weight_valid;
        addr_t   weight_addr;
        addr_t   weight_stride;
        ARBPRECTYPE   weight_dtype;
        bool     toggle_weight;
        bool     clamp;
        tick_t   clock;
};

#endif  
