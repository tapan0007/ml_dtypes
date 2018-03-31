#pragma once

#ifndef KCC_WAVECODE_WAVECODEWAVEOP_H
#define KCC_WAVECODE_WAVECODEWAVEOP_H

#include <string>
#include <cstdio>



#include "compisa/inc/compisaset.hpp"



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

#define ASSERT_HAS_EVENT(edge, from, to)                        \
    Assert((edge)->gEventId() != events::EventId_Invalid(),     \
    "WaveEdge from waveop ", (from)->gName(), " to waveop ",    \
    (to)->gName(), " has no event")

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

    /* Process incoming edges for instructions without embedded events (no SYNC)
    * 1. Issue WAIT instruction for all in-edges
    */
    void processIncomingEdges(wave::WaveOp* waveop);

    /* Process incoming edges for instructions with embedded events (with SYNC)
    * 1. Assign embedded wait for one in-edge
    * 2. Issue WAIT instruction for other in-edges
    */
    void processIncomingEdges(wave::WaveOp* waveop, TPB_CMD_SYNC& sync);

    /* Process incoming edges for instructions with embedded events (with SYNC)
    * But don't assign embedded events to instruction
    * 1. Remember embedded wait id/mode for one in-edge
    */
    void processIncomingEdges(wave::WaveOp* waveop, events::EventId& waitEventId,
                              events::EventWaitMode& waitEventMode);


    void findSetEventIdMode(wave::WaveOp* waveop, events::EventId& setEventId,
                            events::EventSetMode& setEventMode);

    /* Process outgoing edges for instructions without embedded events (no SYNC)
    * 1. Issue SET instruction for all out-edges
    */
    void processOutgoingEdges(wave::WaveOp* waveop);

    /* Process outgoing edges for instructions with embedded events (with SYNC)
    * 1. Assign embedded set for one out-edge
    * 2. Issue SET instruction for other out-edges
    */
    template <typename INST>
    bool processOutgoingEdges(wave::WaveOp* waveop, INST& instr)
    {
        bool instructionWritten = false; // for no succ edges with event, return false
        bool firstEmb = true;

        for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
            if (! succWaveEdge->qNeedToImplementWait()) {
                continue;
            }

            if (firstEmb) {
                firstEmb = false;
                instr.sync.set_event_id    = succWaveEdge->gEventId();
                instr.sync.set_event_mode  = events::eventSetMode2Int(
                                                succWaveEdge->gSetEventMode());
                m_WaveCode.writeInstruction(instr); // this requires template
                instructionWritten = true;
            } else {
                compisa::SetInstr setEventInstr;
                setEventInstr.event_id = succWaveEdge->gEventId();
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

