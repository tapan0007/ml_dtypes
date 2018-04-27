#include "utils/inc/asserter.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/waveop.hpp"


namespace kcc {
namespace wave {


WaveEdge::WaveEdge(WaveOp* fromOp, WaveOp* toOp)
    : m_FromOp(fromOp)
    , m_ToOp(toOp)
{
}

void
WaveEdge::rEvent(events::EventSetMode setMode, events::EventId eventId, events::EventWaitMode waitMode)
{
    m_Channel.rEvent(setMode, eventId, waitMode);
}

bool
WaveEdge::qNeedToImplementSync() const
{
    if (this->gEventId() == events::EventId_Invalid()) {
        Assert(! qNeedToSync(), "Invalid event ID on an edge that need be waited for: from waveop '",
            gFromOp()->gName(), "', to waveop '", gToOp()->gName(), "'");
        return false;
    } else {
        Assert(qNeedToSync(), "Valid event ID on an edge that need not be waited for: from waveop '",
            gFromOp()->gName(), "', to waveop '", gToOp()->gName(), "'");
    }
    return true;
}

bool
WaveEdge::qNeedToSync() const
{
    const wave::WaveOp* const prevWaveop = gFromOp();
    const wave::WaveOp* const succWaveop = gToOp();

    // when two waveops execute on different engines, need for sync
    if (prevWaveop->gEngineId() != succWaveop->gEngineId()) {
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
    if (EngineId::DmaEng == prevWaveop->gEngineId()) {
        if (prevWaveop->qSbAtomSaveWaveOp() && succWaveop->qSbAtomSaveWaveOp()) {
            return false;
        } else {
            return true;
        }
    }
    return false;
}



} // namespace wave
} // namespace kcc
