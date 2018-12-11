#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOMLOAD_H
#define KCC_WAVECODE_WAVECODESBATOMLOAD_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "compisa/inc/compisasimmemcpy.hpp"

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


protected:

    void calcInputSize(const wave::SbAtomLoadWaveOp* sbAtomLoadWaveop);
};

}}

#endif // KCC_WAVECODE_WAVECODESBATOMLOAD_H

