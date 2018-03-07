#pragma once

#ifndef KCC_WAVECODE_WAVECODEWAVEOP_H
#define KCC_WAVECODE_WAVECODEWAVEOP_H

#include <string>
#include <cstdio>



#include "tcc/inc/tcc.hpp"

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

class WaveCode;

namespace wave {
    class WaveOp;
}

namespace wavecode {

class WaveCode;


class WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeWaveOp(WaveCode* wavecode);

    virtual ~WaveCodeWaveOp()
    {}

    virtual void generate(wave::WaveOp* waveOp) = 0;

    //----------------------------------------------------------------
    wave::WaveOp* gWaveOp() const {
        return m_WaveOp;
    }

    //----------------------------------------------------------------
    void rWaveOp(wave::WaveOp* waveOp) {
        m_WaveOp = waveOp;
    }

protected:
    void epilogue(const wave::WaveOp* waveOp);

protected:
    WaveCode* const m_WaveCode;
    wave::WaveOp*   m_WaveOp;
};

}}

#endif // KCC_WAVECODE_WAVECODEWAVEOP_H

