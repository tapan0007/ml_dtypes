#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOM_H
#define KCC_WAVECODE_WAVECODESBATOM_H

#include <string>
#include <cstdio>
#include <vector>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/misc.hpp"

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

private:
    static compisa::DmaTriggerInstr s_dmaTrig;
protected:
    void processOutgoingEdgesAlreadyEmb(wave::SbAtomWaveOp* waveop, events::EventId);
    void addDmaBarrier(const wave::SbAtomWaveOp* sbAtomWaveop, EngineId engId);
    void addSecondDmaTrigger(compisa::DmaTriggerInstr& dmaTriggerInstr, EngineId chosenEngId);

    kcc_int32 findSuccEventsAndChosenEngine(wave::SbAtomWaveOp* sbAtomWaveop,
                        EngineId& chosenEngId,
                        std::vector<events::EventId>& succEventIds);

private:
    kcc_int32 calculateDmaCycleWait(const wave::SbAtomWaveOp* sbAtomWaveop) const;
};

}}

#endif // KCC_WAVECODE_WAVECODESBATOM_H


