#include "utils/inc/asserter.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/waveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/matmulwaveop.hpp"


namespace kcc {
namespace wave {


/****************************************************************
****************************************************************/
WaveEdge::WaveEdge(WaveOp* fromOp, WaveOp* toOp)
    : m_FromOp(fromOp)
    , m_ToOp(toOp)
    , m_SyncMethod(SyncMethod::None)
{
}

/****************************************************************
****************************************************************/
void
WaveEdge::rEvent(const events::EventSetMode setMode, events::EventId eventId,
                 const events::EventWaitMode waitMode)
{
    bool doSet = false;

    switch (setMode) {
    case events::EventSetMode::DontSet:
        doSet = false;
        break;
    case events::EventSetMode::OnEndRdSrc:
    case events::EventSetMode::OnEndWrDst:
    case events::EventSetMode::OnEndInstr:
        doSet = true;
        break;
    default:
        Assert(false, "Bad event set mode ", static_cast<int>(setMode));
        break;
    }

    bool doWait = false;
    switch (waitMode) {
    case events::EventWaitMode::DontWait:
        doWait = false;
        break;
    case events::EventWaitMode::WaitOnly:
    case events::EventWaitMode::WaitThenClear:
        doWait = true;
        break;
    default:
        Assert(false, "Bad event wait mode ", static_cast<int>(waitMode));
        break;
    }
    Assert( doSet == doWait, "Set and wait mode must be equivalent");

    m_EventChannel.rEvent(setMode, eventId, waitMode);
    if (doSet) {
        rSyncMethod(SyncMethod::WithEvent);
    }
}

/****************************************************************
****************************************************************/
void
WaveEdge::clearEvent()
{
    m_EventChannel.clear();
    rSyncMethod(SyncMethod::None);
}

/****************************************************************
****************************************************************/
bool
WaveEdge::qCanSyncWithSemaphore() const
{
    return m_FromOp->qSbAtomWaveOp();
}

/****************************************************************
****************************************************************/
bool
WaveEdge::qNeedToImplementSync() const
{
    if (! qNeedToSync()) {
        Assert( !qSyncedWithSemaphore() && 
                gEventId() == events::EventId_Invalid() && !qSyncedWithEvent(),
               "Dependency (", gFromOp()->gName(), ",", gToOp()->gName(), ") need not be waited for");
    } else {
        Assert((qSyncedWithEvent() && gEventId() != events::EventId_Invalid()
               && ! qSyncedWithSemaphore())
               || (!qSyncedWithEvent() && gEventId() == events::EventId_Invalid()
                 && qSyncedWithSemaphore()),
               "Dependency (", gFromOp()->gName(), ",", gToOp()->gName(), ") must be waited for");
    }

    return qNeedToSync();
}

/****************************************************************
****************************************************************/
bool
WaveEdge::qNeedToSync() const
{
    const wave::WaveOp* const prevWaveop = gFromOp();
    const wave::WaveOp* const succWaveop = gToOp();

    // when two waveops execute on different engines, need for sync
    if (prevWaveop->gEngineId() != succWaveop->gEngineId()) {
        // LoadWeight -> Matmul (sb address < 0) => no event needed.
        // There shouldn't be an arrow there
        if (prevWaveop->qSbAtomLoadWaveOp() && succWaveop-> qMatMulWaveOp()) {
            const auto atomLoadWaveop = dynamic_cast<const wave::SbAtomLoadWaveOp*>(prevWaveop);
            const auto matmulWaveop = dynamic_cast<const wave::MatMulWaveOp*>(succWaveop);
            if (atomLoadWaveop->qContainWeights() && matmulWaveop->gWeightsSbAddress() < 0) {
                std::cerr <<"WARNING: MatMul waveop " << matmulWaveop->gName()
                    << " has SbAddress<0, but depends on Load-weights waveop "
                    << atomLoadWaveop->gName();
            }
        }
        return true;
    }
    // when two waveops execute on the same engine, no need for sync except for DMA.
    // The only case that does NOT need sync is two saves.
    //
    // Load -> Save: this is necessary (probably very rare) because we are rely on data
    // in SB to be correct. It is most likely not necessary because data is Loaded from DRAM,
    // and saving the same data back to DRAM is not likely.
    //
    // Save -> Load: We save data from one region of memory and load different data in
    // the same region => Load has to wait until Save is done.
    //
    // Load -> Load: Suppose we load weights for one layer, and then want to overwrite
    // the same memory with other weights. Since the first weights are already in DRAM
    // (they were loaded from DRAM), we don't need to save
    //
    if (prevWaveop->qSbAtomWaveOp()) {
        if (succWaveop->qSbAtomWaveOp()) {
            const auto prevSbAtom = dynamic_cast<const wave::SbAtomWaveOp*>(prevWaveop);
            const auto succSbAtom = dynamic_cast<const wave::SbAtomWaveOp*>(succWaveop);
            if (prevSbAtom->gDmaQueue() ==  succSbAtom->gDmaQueue()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

/****************************************************************
****************************************************************/
bool
WaveEdge::qSyncedWithEvent() const
{
    return SyncMethod::WithEvent == m_SyncMethod;
}

/****************************************************************
****************************************************************/
bool
WaveEdge::qSyncedWithSemaphore() const
{
    return SyncMethod::WithSemaphore == m_SyncMethod;
}


} // namespace wave
} // namespace kcc
