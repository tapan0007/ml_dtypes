#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOMFILE_H
#define KCC_WAVECODE_WAVECODESBATOMFILE_H

#include <string>
#include <cstdio>



#include "tcc.hpp"

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace layers {
    class Layer;
}
namespace wave {
    class SbAtomFileWaveOp;
}

namespace wavecode {



class WaveCodeSbAtomFile : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeSbAtomFile(WaveCode* waveCode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;


private:
};

}}

#endif // KCC_WAVECODE_WAVECODESBATOMFILE_H

