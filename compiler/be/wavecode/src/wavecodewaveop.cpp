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




/* Process incoming edges for instructions without embedded events (no SYNC)
 * 1. Issue WAIT instruction for all in-edges
 */
void
WaveCodeWaveOp::processIncomingEdges(wave::WaveOp* waveop)
{
    const EngineId engineId = waveop->gEngineId();

    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementWait()) {
            continue;
        }
        compisa::WaitInstr waitInstr;
        waitInstr.event_id  = prevWaveEdge->gEventId();
        m_WaveCode.writeInstruction(waitInstr, engineId);
    }
}




/* Process incoming edges for instructions with embedded events (with SYNC)
 * 1. Assign embedded wait for one in-edge
 * 2. Issue WAIT instruction for other in-edges
 */
void
WaveCodeWaveOp::processIncomingEdges(wave::WaveOp* waveop, TPB_CMD_SYNC& sync)
{
    const EngineId engineId = waveop->gEngineId();
    bool firstEmb = true;

    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementWait()) {
            continue;
        }
        if (firstEmb) {
            firstEmb = false;
            sync.wait_event_id      = prevWaveEdge->gEventId();
            sync.wait_event_mode    = eventWaitMode2Int(prevWaveEdge->gWaitEventMode());
        } else {
            compisa::WaitInstr waitInstr;
            waitInstr.event_id  = prevWaveEdge->gEventId();
            m_WaveCode.writeInstruction(waitInstr, engineId);
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
    bool firstEmb = true;

    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementWait()) {
            continue;
        }
        if (firstEmb) {
            firstEmb = false;
            waitEventId = prevWaveEdge->gEventId();
            waitEventMode = prevWaveEdge->gWaitEventMode();
        } else {
            compisa::WaitInstr waitInstr;
            waitInstr.event_id  = prevWaveEdge->gEventId();
            m_WaveCode.writeInstruction(waitInstr, engineId);
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

    for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
        if (! succWaveEdge->qNeedToImplementWait()) {
            continue;
        }
        compisa::SetInstr setEventInstr;
        setEventInstr.event_id          = succWaveEdge->gEventId();
        m_WaveCode.writeInstruction(setEventInstr, waveop->gEngineId());
    }
}


}}


