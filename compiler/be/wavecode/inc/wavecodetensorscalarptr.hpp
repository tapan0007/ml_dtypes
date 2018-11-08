#pragma once

#ifndef KCC_WAVECODE_WAVECODETENSORSCALARPTR_H
#define KCC_WAVECODE_WAVECODETENSORSCALARPTR_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class TensorScalarPtrWaveOp;
}

namespace wavecode {



class WaveCodeTensorScalarPtr : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeTensorScalarPtr(WaveCodeRef wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;
};


}}

#endif // KCC_WAVECODE_WAVECODETENSORSCALARPTR_H

