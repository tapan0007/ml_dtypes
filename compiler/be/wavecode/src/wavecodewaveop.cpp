#include <set>
#include <array>
#include <sstream>



#include "utils/inc/asserter.hpp"
#include "utils/inc/debug.hpp"

#include "compisa/inc/compisawait.hpp"
#include "compisa/inc/compisanop.hpp"
#include "compisa/inc/compisasemaphore.hpp"


#include "wave/inc/waveedge.hpp"
#include "wave/inc/waveop.hpp"
#include "wave/inc/sbatomwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"

#include "dma/inc/dmaqueue.hpp"

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

        ++numSyncs;
        if (prevWaveEdge->qSyncedWithEvent()) {
            const auto evtId = prevWaveEdge->gEventId();

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
                m_WaveCode.writeWaitOrWaitClearInstr(prevWaveEdge, engineId);
            }
        } else if (prevWaveEdge->qSyncedWithSemaphore()) {
            GenerateSemaphoreInstr(prevWaveEdge);
        } else {
            Assert(false, "Must sync edge from ", prevWaveEdge->gFromOp()->gName(),
                   " to ", prevWaveEdge->gToOp()->gName());
        }
    }
    return numSyncs;
} // WaveCodeWaveOp::processIncomingEdges


//----------------------------------------------------------------
void
WaveCodeWaveOp::GenerateSemaphoreInstr(const wave::WaveEdge* prevWaveEdge)
{
    const auto prevWaveop = prevWaveEdge->gFromOp();
    auto prevDatamoveWaveop = dynamic_cast<const wave::DataMoveWaveOp*>(prevWaveop);
    Assert(prevDatamoveWaveop, "WaveOp ", prevWaveop->gName(), " should be Load/Save/TpbCopy");
    const auto succWaveop = prevWaveEdge->gToOp();

    // semaphore wait must >= because 2 DMA transfers that are on the same queue
    // could both finish before the engine(s) that is(are) waiting on the first
    // condition arrives to the semaphore.wait, misses the condition, and gets stuck.
    compisa::SemaphoreInstr semInstr;
    AssignWithSizeCheck(semInstr.semaphore_id, prevDatamoveWaveop->gDmaQueue()->gSemaphoreId());
    AssignWithSizeCheck(semInstr.wait_cond, TONGA_ISA_TPB_SEMAPHORE_WAIT_COND_GREATER_EQUAL);
    AssignWithSizeCheck(semInstr.wait_value, prevDatamoveWaveop->gTriggerOrd());

    std::ostringstream oss;
    oss << prevDatamoveWaveop->gOrder() << "->" << succWaveop->gOrder() << ": " << succWaveop->gName();
    m_WaveCode.SaveName(semInstr, oss.str().c_str());


    std::array<const dma::DmaQueue*, 2> dmaQues;
    std::array<kcc_int32, dmaQues.size()> trigOrds;


    dmaQues[0]  = prevDatamoveWaveop->gDmaQueue();
    dmaQues[1]  = nullptr;
    trigOrds[0] = prevDatamoveWaveop->gTriggerOrd();
    trigOrds[1] = -1;

    //----------------------------------------------------------------
    // For weights with 2 queues: wait on the 2nd queue
    //----------------------------------------------------------------
    auto prevSbLoad = dynamic_cast<const wave::SbAtomLoadWaveOp*>(prevDatamoveWaveop);
    if (prevSbLoad) {
        const dma::DmaQueue* que1 = prevSbLoad->gDmaQueue1();
        dmaQues[1] = que1;
        Assert(!que1 || prevSbLoad->qContainWeights(), "Double DMA supported only for Loading weights");
        trigOrds[1] = prevSbLoad->gTriggerOrd1();
    }


    for (kcc_uint32 i = 0; i < dmaQues.size(); ++i) {
        auto que = dmaQues[i];
        if (! que) {
            continue;
        }
        auto trigOrd = trigOrds[i];

        AssignWithSizeCheck(semInstr.wait_cond, TONGA_ISA_TPB_SEMAPHORE_WAIT_COND_GREATER_EQUAL);
        AssignWithSizeCheck(semInstr.semaphore_id, que->gSemaphoreId());
        AssignWithSizeCheck(semInstr.wait_value, trigOrd);
        m_WaveCode.writeInstruction(semInstr, succWaveop->gEngineId());
    }
} // WaveCodeWaveOp::GenerateSemaphoreInstr

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
} // WaveCodeWaveOp::findFirstSetEventIdMode


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
        AssignWithSizeCheck(setEventInstr.event_idx, evtId);

        std::ostringstream oss;
        oss << waveop->gOrder() << "-" << waveop->gName();
        m_WaveCode.SaveName(setEventInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(setEventInstr, waveop->gEngineId());
    }
    return numSyncs;
} // WaveCodeWaveOp::processOutgoingEdges





//======================================================================
std::string
WaveCodeWaveOp::FileRange::String() const
{
    std::ostringstream oss;
    oss << "(" << m_File << "," << m_OffsetRange.gBegin()
        << "," << m_OffsetRange.gSize() << "," << m_OffsetRange.gEnd() << ")";
    return oss.str();
}

//************************************************************************
void
WaveCodeWaveOp::addDmaBarrier(const wave::SbAtomWaveOp* sbatomWaveop, EngineId engId) const
{
    const kcc_int32 cycleWait = calculateDmaCycleWait(sbatomWaveop);
    if (cycleWait <= 0) {
        return;
    }
    compisa::NopInstr nopInstr;
    AssignWithSizeCheck(nopInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(nopInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(nopInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));
    AssignWithSizeCheck(nopInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(nopInstr.cycle_cnt, cycleWait);

    std::ostringstream oss;
    oss << sbatomWaveop->gOrder() << "-" << sbatomWaveop->gName() << ":Delay before DMA ";
    m_WaveCode.SaveName(nopInstr, oss.str().c_str());
    m_WaveCode.writeInstruction(nopInstr, engId);
} // WaveCodeSbAtom::addDmaBarrier

//************************************************************************
kcc_int32
WaveCodeWaveOp::calculateDmaCycleWait(const wave::SbAtomWaveOp* sbatomWaveop) const
{
    if (false) {
        const arch::PeArray& peArray(arch::Arch::gArch().gPeArray());
        return std::max(peArray.gNumberRows(), peArray.gNumberColumns());
    } else {
        return sbatomWaveop->gNumPartitions();
    }
}

}}


