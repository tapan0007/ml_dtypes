#pragma once

#ifndef KCC_WAVECODE_WAVECODERESADD_H
#define KCC_WAVECODE_WAVECODERESADD_H

#include <string>
#include <cstdio>



#include "tcc.hpp"

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class ResAddWaveOp;
}

namespace wavecode {



class WaveCodeResAdd : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeResAdd(WaveCode* wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

};


}}

#endif // KCC_WAVECODE_WAVECODERESADD_H

