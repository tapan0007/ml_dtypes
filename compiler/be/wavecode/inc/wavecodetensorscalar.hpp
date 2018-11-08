#pragma once

#ifndef KCC_WAVECODE_WAVECODETENSORSCALAR_H
#define KCC_WAVECODE_WAVECODETENSORSCALAR_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class TensorScalarWaveOp;
}

namespace wavecode {



class WaveCodeTensorScalar : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeTensorScalar(WaveCodeRef wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;
};


}}

#endif // KCC_WAVECODE_WAVECODETENSORSCALAR_H

