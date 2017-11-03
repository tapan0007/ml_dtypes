#ifndef ACTIVATE_H
#define ACTIVATE_H

#include "sigint.h"
#include <vector>

class Activate : public ActivateSbInterface {
    public:
        Activate();
        ~Activate();
        ActivateSbSignals pull_activate();
        void connect_psum(PSumActivateInterface *);
        void step();
    private:
        PSumActivateInterface   *psum_connect;
        PSumActivateSignals      ps;

};

class ActivateArray {
    public:
        ActivateArray(int n_cols = 64);
        ~ActivateArray();
        Activate& operator[](int index);
        void connect_psum(int col_id, PSumActivateInterface *);
        void step();
    private:
        std::vector<Activate>     buffer;

};
#endif
