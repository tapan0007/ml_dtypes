#include <set>


#include "utils/inc/asserter.hpp"

#include "compisa/inc/compisawait.hpp"


#include "wave/inc/waveedge.hpp"
#include "wave/inc/waveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodewaveop.hpp"

namespace kcc {
namespace wavecode {

WaveCodeWaveOp::WaveCodeWaveOp(WaveCodeRef wavecode)
    : m_WaveCode(wavecode)
{}

bool
WaveCodeWaveOp::qParallelStreams() const
{
    return m_WaveCode.qParallelStreams();
}


void
WaveCodeWaveOp::writeWaitOrWaitClearInstr(const wave::WaveEdge* waveEdge, EngineId engineId)
{
    const events::EventWaitMode waitEventMode = waveEdge->gWaitEventMode();
    Assert(waitEventMode == events::EventWaitMode::WaitThenClear
                || waitEventMode == events::EventWaitMode::WaitOnly,
           "Cannot wait on edge with DontWait mode");

    compisa::WaitInstr waitInstr;
    waitInstr.event_idx         = waveEdge->gEventId();
#if 0
    // Not sure whether wait_event_mode works in SIM.
    waitInstr.wait_event_mode   = eventWaitMode2Isa(waitEventMode);
    m_WaveCode.writeInstruction(waitInstr, engineId);
#else
    waitInstr.wait_event_mode   = eventWaitMode2Isa(events::EventWaitMode::WaitOnly);
    m_WaveCode.writeInstruction(waitInstr, engineId);

    if (waitEventMode == events::EventWaitMode::WaitThenClear) {
        compisa::ClearInstr clearInstr;
        clearInstr.event_idx  = waveEdge->gEventId();
        m_WaveCode.writeInstruction(clearInstr, engineId);
    }
#endif
}


/* Process incoming edges for instructions without embedded events (no SYNC)
 * 1. Issue WAIT instruction for all in-edges
 */
void
WaveCodeWaveOp::processIncomingEdges(wave::WaveOp* waveop)
{
    const EngineId engineId = waveop->gEngineId();
    std::set<events::EventId> eventIds;

    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementWait()) {
            continue;
        }
        const auto evtId = prevWaveEdge->gEventId();
        Assert(eventIds.find(evtId) == eventIds.end(), "Double event id ", evtId);
        eventIds.insert(evtId);

        writeWaitOrWaitClearInstr(prevWaveEdge, engineId);
    }
}




/* Process incoming edges for instructions with embedded events (with SYNC)
 * 1. Assign embedded wait for one in-edge
 * 2. Issue WAIT instruction for other in-edges
 */
void
WaveCodeWaveOp::processIncomingEdges(wave::WaveOp* waveop, TONGA_ISA_TPB_INST_EVENTS& sync)
{
    const EngineId engineId = waveop->gEngineId();
    bool firstEmb = true;
    std::set<events::EventId> eventIds;

    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementWait()) {
            continue;
        }
        const auto evtId = prevWaveEdge->gEventId();
        Assert(eventIds.find(evtId) == eventIds.end(), "Double event id ", evtId);
        eventIds.insert(evtId);

        if (firstEmb) {
            firstEmb = false;
            sync.wait_event_idx      = evtId;
            sync.wait_event_mode    = eventWaitMode2Isa(prevWaveEdge->gWaitEventMode());
        } else {
            writeWaitOrWaitClearInstr(prevWaveEdge, engineId);
        }
    }
}




/* Process incoming edges for instructions with embedded events (with SYNC)
 * But don't assign embedded events to instruction
 * 1. Remember embedded wait id/mode for one in-edge
 */
void
WaveCodeWaveOp::processIncomingEdges(wave::WaveOp* waveop, events::EventId& waitEventId,
        events::EventWaitMode& waitEventMode)
{
    const EngineId engineId = waveop->gEngineId();
    std::set<events::EventId> eventIds;
    bool firstEmb = true;

    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementWait()) {
            continue;
        }
        const auto evtId = prevWaveEdge->gEventId();
        Assert(eventIds.find(evtId) == eventIds.end(), "Double event id ", evtId);
        eventIds.insert(evtId);

        if (firstEmb) {
            firstEmb = false;
            waitEventId = evtId;
            waitEventMode = prevWaveEdge->gWaitEventMode();
        } else {
            writeWaitOrWaitClearInstr(prevWaveEdge, engineId);
        }
    }
}



void
WaveCodeWaveOp::findSetEventIdMode(wave::WaveOp* waveop, events::EventId& setEventId,
                                   events::EventSetMode& setEventMode)
{
    bool firstEmb = true;

    for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
        if (! succWaveEdge->qNeedToImplementWait()) {
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



/* Process outgoing edges for instructions without embedded events (no SYNC)
* 1. Issue SET instruction for all out-edges
*/
void
WaveCodeWaveOp::processOutgoingEdges(wave::WaveOp* waveop)
{
    std::set<events::EventId> eventIds;

    for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
        if (! succWaveEdge->qNeedToImplementWait()) {
            continue;
        }
        const auto evtId = succWaveEdge->gEventId();
        Assert(eventIds.find(evtId) == eventIds.end(), "Double event id ", evtId);
        eventIds.insert(evtId);

        compisa::SetInstr setEventInstr;
        setEventInstr.event_idx = evtId;
        m_WaveCode.writeInstruction(setEventInstr, waveop->gEngineId());
    }
}


}}


