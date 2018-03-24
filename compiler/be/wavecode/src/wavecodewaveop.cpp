#include "shared/inc/tpb_isa.hpp"


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
            WAIT waitInstr;
            waitInstr.event_id  = prevWaveEdge->gEventId();
            m_WaveCode.writeInstruction(waitInstr, engineId);
        }
    }
}

void
WaveCodeWaveOp::processIncomingEdges(wave::WaveOp* waveop, EventId& waitEventId,
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
            WAIT waitInstr;
            waitInstr.event_id  = prevWaveEdge->gEventId();
            m_WaveCode.writeInstruction(waitInstr, engineId);
        }
    }
}

void
WaveCodeWaveOp::findSetEventIdMode(wave::WaveOp* waveop, EventId& setEventId, events::EventSetMode& setEventMode)
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


}}


