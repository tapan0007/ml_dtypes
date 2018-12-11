#pragma once

#ifndef KCC_WAVECODE_WAVECODESBATOMLOAD_SIM_H
#define KCC_WAVECODE_WAVECODESBATOMLOAD_SIM_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "compisa/inc/compisasimmemcpy.hpp"

#include "wavecode/inc/wavecodesbatomload.hpp"


namespace kcc {

namespace wave {
    class SbAtomLoadWaveOp;
}

namespace wavecode {



class WaveCodeSbAtomLoadSim : public WaveCodeSbAtomLoad {
public:
    //----------------------------------------------------------------
    WaveCodeSbAtomLoadSim(WaveCodeRef waveCode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;


private:
    void generateForSim(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp);
    void generateForSimWithRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp);
    void generateForSimNoRepl(wave::SbAtomLoadWaveOp* sbAtomLoadWaveOp);

    static void setInstructionEvents(compisa::SimMemCpyInstr& dramToStateBufInstr, bool first, bool last,
                    events::EventId waitEventId, events::EventWaitMode waitEventMode,
                    events::EventId setEventId, events::EventSetMode setEventMode);


private:
};

}}

#endif // KCC_WAVECODE_WAVECODESBATOMLOAD_H

