#ifndef ACTIVATE_H
#define ACTIVATE_H

#include "sigint.h"
#include "types.h"
#include <vector>

class Activate {
    public:
        Activate();
        ~Activate();
        void connect_north(PSumActivateInterface *);
        void step();
    private:
        PSumActivateInterface   *north;
        PSumActivateSignals      ps;

};

class ActivateArray {
    public:
        ActivateArray(int n_cols = 64);
        ~ActivateArray();
        void connect_north(int col_id, PSumActivateInterface *);
        void step();
    private:
        std::vector<Activate>     buffer;

};
#endif 
