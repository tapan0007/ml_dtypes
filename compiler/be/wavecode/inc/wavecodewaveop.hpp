#pragma once

#ifndef KCC_WAVECODE_WAVECODEWAVEOP_H
#define KCC_WAVECODE_WAVECODEWAVEOP_H

#include <string>
#include <cstdio>



#include "tcc/inc/tcc.hpp"



#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "events/inc/events.hpp"

#include "wave/inc/waveop.hpp"
#include "wave/inc/waveedge.hpp"

#include "wavecode/inc/wavecode.hpp"


namespace kcc {

class WaveCode;

namespace wave {
    class WaveOp;
}

namespace wavecode {

class WaveCode;


class WaveCodeWaveOp {
protected:
    using WaveCodeRef = WaveCode&;
public:
    //----------------------------------------------------------------
    WaveCodeWaveOp(WaveCodeRef wavecode);

    virtual ~WaveCodeWaveOp()
    {}

    virtual void generate(wave::WaveOp* waveOp) = 0;

    //----------------------------------------------------------------
    wave::WaveOp* gWaveOp() const {
        return m_WaveOp;
    }

    //----------------------------------------------------------------
    void rWaveOp(wave::WaveOp* waveOp) {
        m_WaveOp = waveOp;
    }

protected:
    void epilogue(const wave::WaveOp* waveOp);
    bool qParallelStreams() const;

    void processIncomingEdges(wave::WaveOp* waveop, TPB_CMD_SYNC& sync);
    void processIncomingEdges(wave::WaveOp* waveop, events::EventId& waitEventId, events::EventWaitMode& waitEventMode);
    void findSetEventIdMode(wave::WaveOp* waveop, events::EventId& setEventId, events::EventSetMode& setEventMode);

    template <typename INST>
    bool processOutgoingEdges(wave::WaveOp* waveop, INST& instr)
    {
        bool instructionWritten = false;
        bool firstEmb = true;

        for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
            if (! succWaveEdge->qNeedToImplementWait()) {
                continue;
            }

            if (firstEmb) {
                firstEmb = false;
                instr.sync.set_event_id    = succWaveEdge->gEventId();
                instr.sync.set_event_mode  = events::eventSetMode2Int(succWaveEdge->gSetEventMode());
                m_WaveCode.writeInstruction(instr);
                instructionWritten = true;
            } else {
                SET setEventInstr;
                setEventInstr.event_id          = succWaveEdge->gEventId();
                m_WaveCode.writeInstruction(setEventInstr, waveop->gEngineId());
            }
        }
        return instructionWritten;
    }


protected:
    WaveCodeRef     m_WaveCode;
    wave::WaveOp*   m_WaveOp;
};

}}

#endif // KCC_WAVECODE_WAVECODEWAVEOP_H

