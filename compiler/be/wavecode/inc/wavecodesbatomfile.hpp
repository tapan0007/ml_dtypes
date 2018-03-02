#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOMFILE_H
#define KCC_WAVECODE_WAVECODESBATOMFILE_H

#include <string>
#include <cstdio>



#include "tcc.hpp"

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodesbatom.hpp"


namespace kcc {

namespace wave {
    class SbAtomFileWaveOp;
}

namespace wavecode {



class WaveCodeSbAtomFile : public WaveCodeSbAtom {
public:
    //----------------------------------------------------------------
    WaveCodeSbAtomFile(WaveCode* waveCode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;


private:
};

}}

#endif // KCC_WAVECODE_WAVECODESBATOMFILE_H

