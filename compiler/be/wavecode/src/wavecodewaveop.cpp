#include <set>


#include "utils/inc/asserter.hpp"

#include "compisa/inc/compisawait.hpp"
#include "compisa/inc/compisanop.hpp"


#include "wave/inc/waveedge.hpp"
#include "wave/inc/waveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodewaveop.hpp"

namespace kcc {
namespace wavecode {

//======================================================================
WaveCodeWaveOp::WaveCodeWaveOp(WaveCodeRef wavecode)
    : m_WaveCode(wavecode)
{}

//======================================================================
bool
WaveCodeWaveOp::qParallelStreams() const
{
    return m_WaveCode.qParallelStreams();
}


//======================================================================
void
WaveCodeWaveOp::writeWaitOrWaitClearInstr(const wave::WaveEdge* waveEdge, EngineId engineId)
{
    const events::EventWaitMode waitEventMode = waveEdge->gWaitEventMode();
    Assert(waitEventMode == events::EventWaitMode::WaitThenClear
                || waitEventMode == events::EventWaitMode::WaitOnly,
           "Cannot wait on edge with DontWait mode");

    enum { WAIT_CLEAR_MODE, WAIT_PLUS_CLEAR, NOP };

    //switch (WAIT_PLUS_CLEAR)
    switch (WAIT_CLEAR_MODE)
    {
    case WAIT_CLEAR_MODE: {
        // Not sure whether wait_event_mode works in SIM.
        compisa::WaitInstr waitInstr;
        waitInstr.event_idx         = waveEdge->gEventId();
        waitInstr.wait_event_mode   = eventWaitMode2Isa(waitEventMode);
        m_WaveCode.writeInstruction(waitInstr, engineId);
        break;
    }
    case NOP: {
        // New Nop instruction can wait and set (should use for barrier too)
        compisa::NopInstr nopInstr;
        nopInstr.inst_events.wait_event_idx   = waveEdge->gEventId();
        nopInstr.inst_events.wait_event_mode  = events::eventWaitMode2Isa(waitEventMode);
        nopInstr.inst_events.set_event_idx    = 0;
        nopInstr.inst_events.set_event_mode   = events::eventSetMode2Isa(events::EventSetMode::DontSet);
        m_WaveCode.writeInstruction(nopInstr, engineId);
        break;
    }
    case WAIT_PLUS_CLEAR: {
        // old style: Wait(wait-only); Clear
        {
            compisa::WaitInstr waitInstr;
            waitInstr.event_idx         = waveEdge->gEventId();
            waitInstr.wait_event_mode   = eventWaitMode2Isa(events::EventWaitMode::WaitOnly);
            m_WaveCode.writeInstruction(waitInstr, engineId);
        }

        if (waitEventMode == events::EventWaitMode::WaitThenClear) {
            compisa::ClearInstr clearInstr;
            clearInstr.event_idx  = waveEdge->gEventId();
            m_WaveCode.writeInstruction(clearInstr, engineId);
        }
        break;
    }
    default:
        Assert(false, "Unknown waiting method");
        break;
    }
}


//======================================================================
kcc_int32
WaveCodeWaveOp::processIncomingEdges(wave::WaveOp* waveop, EngineId engineId,
    bool allowEmb,
    TONGA_ISA_TPB_INST_EVENTS* sync,
    events::EventId* waitEventId, events::EventWaitMode* waitEventMode)
{
    Assert((waitEventId==nullptr) == (waitEventMode==nullptr), "Event id and mode must be equal");
    if (allowEmb) {
        Assert(sync || (waitEventId && waitEventMode), "For embedded event need place to store");
        Assert((sync==nullptr) != (waitEventId==nullptr || waitEventMode==nullptr),
            "Embedded event id/mode should go in exactly one place");
    }

    kcc_int32 numSyncs = 0;
    bool firstEmb = true;

    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementSync()) {
            continue;
        }
        const auto evtId = prevWaveEdge->gEventId();
        ++numSyncs;

        if (allowEmb && firstEmb) {
            firstEmb = false;
            if (sync) {
                sync->wait_event_idx     = evtId;
                sync->wait_event_mode    = eventWaitMode2Isa(prevWaveEdge->gWaitEventMode());
            } else {
                *waitEventId = evtId;
                *waitEventMode = prevWaveEdge->gWaitEventMode();
            }
        } else {
            if (firstEmb) {
                firstEmb = false;
                if (waitEventId && waitEventMode) {
                    *waitEventId = evtId;
                    *waitEventMode = prevWaveEdge->gWaitEventMode();
                }
            }
            writeWaitOrWaitClearInstr(prevWaveEdge, engineId);
        }
    }
    return numSyncs;
}


//======================================================================
void
WaveCodeWaveOp::findFirstSetEventIdMode(wave::WaveOp* waveop, events::EventId& setEventId,
                                   events::EventSetMode& setEventMode)
{
    bool firstEmb = true;

    for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
        if (! succWaveEdge->qNeedToImplementSync()) {
            continue;
        }
        if (firstEmb) {
            firstEmb = false;
            setEventId = succWaveEdge->gEventId();
            setEventMode = succWaveEdge->gSetEventMode();
            break;
        }
    }
}


//======================================================================
/* Process incoming edges for instructions without embedded events (no SYNC)
 * 1. Issue WAIT instruction for all in-edges
 */
kcc_int32
WaveCodeWaveOp::processIncomingEdges(wave::WaveOp* waveop)
{
    return processIncomingEdges(waveop, waveop->gEngineId(), false, nullptr, nullptr, nullptr);
}


//======================================================================
kcc_int32
WaveCodeWaveOp::processIncomingEdges(wave::WaveOp* waveop, EngineId engineId)
{
    return processIncomingEdges(waveop, engineId, false, nullptr, nullptr, nullptr);
}


//======================================================================
/* Process incoming edges for instructions with embedded events (with SYNC)
 * 1. Assign embedded wait for one in-edge
 * 2. Issue WAIT instruction for other in-edges
 */
kcc_int32
WaveCodeWaveOp::processIncomingEdges(wave::WaveOp* waveop, TONGA_ISA_TPB_INST_EVENTS& sync)
{
    return processIncomingEdges(waveop, waveop->gEngineId(), true, &sync, nullptr, nullptr);
}


//======================================================================
/* Process incoming edges for instructions with embedded events (with SYNC)
 * But don't assign embedded events to instruction
 * 1. Remember embedded wait id/mode for one in-edge
 */
kcc_int32
WaveCodeWaveOp::processIncomingEdges(wave::WaveOp* waveop, 
                        events::EventId& waitEventId, events::EventWaitMode& waitEventMode)
{
    return processIncomingEdges(waveop, waveop->gEngineId(), true, nullptr, &waitEventId, &waitEventMode);
}


kcc_int32
WaveCodeWaveOp::processIncomingEdgesForceWait(wave::WaveOp* waveop, EngineId engId,
                        events::EventId& waitEventId, events::EventWaitMode& waitEventMode)
{
    return processIncomingEdges(waveop, engId, false, nullptr, &waitEventId, &waitEventMode);
}







//======================================================================
/* Process outgoing edges for instructions without embedded events (no SYNC)
* 1. Issue SET instruction for all out-edges
*/
kcc_int32
WaveCodeWaveOp::processOutgoingEdges(wave::WaveOp* waveop)
{
    int numSyncs = 0;

    for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
        if (! succWaveEdge->qNeedToImplementSync()) {
            continue;
        }
        if (succWaveEdge->qChosenForSuccSbAtom()) {
            continue;
        }
        const auto evtId = succWaveEdge->gEventId();

        ++numSyncs;
        compisa::SetInstr setEventInstr;
        setEventInstr.event_idx = evtId;
        m_WaveCode.writeInstruction(setEventInstr, waveop->gEngineId());
    }
    return numSyncs;
}

//======================================================================
void
WaveCodeWaveOp::SaveName(compisa::MatMulInstr& instr, const char* name)
{
    saveName(instr.reserved_2, name);
}

}}


