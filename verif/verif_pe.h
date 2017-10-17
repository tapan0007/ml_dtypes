#ifndef _VERIFICATION_PE_H
#define _VERIFICATION_PE_H

#include "sigint.h"
#include "pe_array.h"
#include "string.h"
#include <vector>

typedef struct VPeSignals
{
    PeEWSignals ew;
    bool        clamp;
} VPeSignals;

class VPeFeeder : public PeEWInterface, public SbEWBroadcastInterface {
    public:
        PeEWSignals pull_ew() {return sigs.ew;}
        bool        pull_clamp() {return sigs.clamp;}
        void        load(VPeSignals &signals) {
            sigs.clamp = signals.clamp;
            sigs.ew = signals.ew;
        }
    private:
        VPeSignals   sigs;

};

class VPeTester {
    public:
        VPeTester(int _rows = 128, int _cols = 64) : rows(_rows), 
        cols(_cols) {
            for (int i=0; i < rows; i++) {
                feeds.push_back(VPeFeeder());
            }
        }
        void connect(ProcessingElementArray *_pe_array) {
            pe_array = _pe_array;
            for (int i=0; i < rows; i++) {
                pe_array->connect_west(i, &feeds[i]);
                pe_array->connect_statebuffer(i, &feeds[i]);
            }
        }
        void write(VPeSignals *sigs) {
            for (int i=0; i < rows; i++) {
                feeds[i].load(sigs[i]);
            }
        }

        void read(PeNSSignals *sigs) {
            for (int i=0; i < cols; i++) {
                sigs[i] = (*pe_array)[rows - 1][i].pull_ns();
            }
        }
        void step() {
            pe_array->step();
        }
        VPeFeeder& operator[](int index) {return feeds[index];};
    private:
        std::vector<VPeFeeder> feeds;
        ProcessingElementArray *pe_array;
        int rows;
        int cols;


};

#endif
