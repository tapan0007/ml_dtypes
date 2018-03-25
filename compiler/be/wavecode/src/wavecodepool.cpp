#include "shared/inc/tpb_isa_pool.hpp"



#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "layers/inc/poollayer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodepool.hpp"

namespace kcc {
namespace wavecode {

#define ASSERT_HAS_EVENT(edge, from, to) Assert((edge)->gEventId() != EventId_Invalid, "WaveEdge from waveop ", \
            (from)->gName(), " to waveop ", (to)->gName(), " has no event")

WaveCodePool::WaveCodePool(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}


void
WaveCodePool::generate(wave::WaveOp* waveOp)
{
    auto poolWaveop = dynamic_cast<wave::PoolWaveOp*>(waveOp);
    assert(poolWaveop);
    const arch::Arch& arch(arch::Arch::gArch());
    const auto& psumBuf(arch.gPsumBuffer());
    const auto& stateBuf(arch.gStateBuffer());

    const EngineId engineId = poolWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for Pool should be Pooling");

    POOL poolInstr;

    /* pool args */
    switch (poolWaveop->gPoolFunc()) {
    case PoolType::Max:
        poolInstr.pool_func = POOLFUNC::MAX_POOL;
        break;
    case PoolType::Avg:
        poolInstr.pool_func = POOLFUNC::AVG_POOL;
        break;
    default:
        assert(false && "Bad PoolType in PoolWaveOp");
        break;
    }

    poolInstr.in_dtype          = poolWaveop->gInDtype().gSimTypeId();
    poolInstr.out_dtype         = poolWaveop->gOutDtype().gSimTypeId();
    if (poolWaveop->qSrcIsPsum()) {
        poolInstr.src_start_addr = psumBuf.gEntryTpbAddress(
                                            poolWaveop->gSrcPsumBankId(),
                                            poolWaveop->gSrcPsumBankOffset(),
                                            poolWaveop->gInDtype());
    } else { // State buffer
        poolInstr.src_start_addr = stateBuf.gEntryTpbAddress(0, /*row 0 for now*/
                                            poolWaveop->gSrcSbAddress());
    }

    poolInstr.src_x_step        = poolWaveop->gSrcXStep();
    poolInstr.src_x_num         = poolWaveop->gSrcXNum();
    poolInstr.src_y_step        = poolWaveop->gSrcYStep();
    poolInstr.src_y_num         = poolWaveop->gSrcYNum();
    poolInstr.pool_frequency    = poolWaveop->gPoolFrequency();
    poolInstr.pool_scale        = static_cast<float>(1.0/poolWaveop->gPoolFrequency());
    /* strides */
    poolInstr.src_z_step        = poolWaveop->gSrcZStep();
    poolInstr.src_z_num         = poolWaveop->gSrcZNum();
    poolInstr.src_w_step        = poolWaveop->gSrcWStep();
    poolInstr.src_w_num         = poolWaveop->gSrcWNum();
    poolInstr.num_partitions    = poolWaveop->gNumPartitions();

    /* Pool  */
    poolInstr.dst_x_step        = poolWaveop->gDstXStep();
    poolInstr.dst_x_num         = poolWaveop->gDstXNum();
    poolInstr.dst_y_step        = poolWaveop->gDstYStep();
    poolInstr.dst_y_num         = poolWaveop->gDstYNum();
    poolInstr.dst_z_step        = poolWaveop->gDstZStep();
    poolInstr.dst_z_num         = poolWaveop->gDstZNum();
    // For now DST is always StateBuf
    poolInstr.dst_start_addr    = stateBuf.gEntryTpbAddress(0, /*row 0 for now*/
                                            poolWaveop->gDstSbAddress());

    poolInstr.sync.wait_event_id    = 0;
    poolInstr.sync.wait_event_mode  = events::eventWaitMode2Int(events::EventWaitMode::NoEvent);
    poolInstr.sync.set_event_id    = 0;
    poolInstr.sync.set_event_mode  = events::eventSetMode2Int(events::EventSetMode::NoEvent);

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(poolWaveop, poolInstr.sync);
    } // end incoming events


    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(poolWaveop, poolInstr);
    }

    if (! instructionWritten) {
        m_WaveCode.writeInstruction(poolInstr);
    }
}

}}


