#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOMLOAD_H
#define KCC_WAVECODE_WAVECODESBATOMLOAD_H

#include <string>
#include <cstdio>



#include "tcc/inc/tcc.hpp"

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodesbatom.hpp"


namespace kcc {

namespace wave {
    class SbAtomLoadWaveOp;
}

namespace wavecode {



class WaveCodeSbAtomLoad : public WaveCodeSbAtom {
public:
    //----------------------------------------------------------------
    WaveCodeSbAtomLoad(WaveCodeRef waveCode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;


private:
};

}}

#endif // KCC_WAVECODE_WAVECODESBATOMLOAD_H

