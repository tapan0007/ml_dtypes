#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOMSAVE_H
#define KCC_WAVECODE_WAVECODESBATOMSAVE_H

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
    class SbAtomSaveWaveOp;
}

namespace wavecode {



class WaveCodeSbAtomSave : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeSbAtomSave(WaveCode* waveCode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;


private:
};

}}

#endif // KCC_WAVECODE_WAVECODESBATOMSAVE_H

