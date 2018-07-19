#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOM_H
#define KCC_WAVECODE_WAVECODESBATOM_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "compisa/inc/compisadmatrigger.hpp"
#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class SbAtomWaveOp;
}


namespace wavecode {


class WaveCodeSbAtom : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeSbAtom(WaveCodeRef waveCode);

protected:
    void processOutgoingEdgesAlreadyEmb(wave::SbAtomWaveOp* waveop, events::EventId);
    void addDmaBarrier(EngineId engId);
    void addSecondDmaTrigger(compisa::DmaTriggerInstr& dmaTriggerInstr, EngineId chosenEngId);


private:
};

}}

#endif // KCC_WAVECODE_WAVECODESBATOM_H


