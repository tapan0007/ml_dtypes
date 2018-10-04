#pragma once

#ifndef KCC_WAVECODE_WAVECODETENSORSCALARCONST_H
#define KCC_WAVECODE_WAVECODETENSORSCALARCONST_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class TensorScalarConstWaveOp;
}

namespace wavecode {



class WaveCodeTensorScalarConst : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeTensorScalarConst(WaveCodeRef wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;
};


}}

#endif // KCC_WAVECODE_WAVECODETENSORSCALARCONST_H

