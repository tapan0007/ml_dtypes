#include "shared/inc/tpb_isa_pool.hpp"

#include "utils/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "layers/inc/poollayer.hpp"

#include "wave/inc/poolwaveop.hpp"
#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodepool.hpp"

namespace kcc {
namespace wavecode {

WaveCodePool::WaveCodePool(WaveCode* waveCode)
    : WaveCodeWaveOp(waveCode)
{}


void
WaveCodePool::generate(wave::WaveOp* waveOp)
{
    auto poolWaveOp = dynamic_cast<wave::PoolWaveOp*>(waveOp);
    assert(poolWaveOp);
    const arch::Arch& arch(arch::Arch::gArch());
    const auto& psumBuf(arch.gPsumBuffer());
    const auto& stateBuf(arch.gStateBuffer());

    POOL poolInstr;

    /* pool args */
    switch (poolWaveOp->gPoolFunc()) {
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

    poolInstr.in_dtype          = poolWaveOp->gInDtype().gSimTypeId();
    poolInstr.out_dtype         = poolWaveOp->gOutDtype().gSimTypeId();
    if (poolWaveOp->qSrcIsPsum()) {
        poolInstr.src_start_addr = psumBuf.gEntryTpbAddress(
                                            poolWaveOp->gSrcPsumBankId(),
                                            poolWaveOp->gSrcPsumBankOffset(),
                                            poolWaveOp->gInDtype());
    } else { // State buffer
        poolInstr.src_start_addr = stateBuf.gEntryTpbAddress(0, /*row 0 for now*/
                                            poolWaveOp->gSrcSbAtomId() * poolWaveOp->gWaveAtomSize()
                                                + poolWaveOp->gSrcSbOffsetInAtom());
    }

    poolInstr.src_x_step        = poolWaveOp->gSrcXStep();
    poolInstr.src_x_num         = poolWaveOp->gSrcXNum();
    poolInstr.src_y_step        = poolWaveOp->gSrcYStep();
    poolInstr.src_y_num         = poolWaveOp->gSrcYNum();
    poolInstr.pool_frequency    = poolWaveOp->gPoolFrequency();
    poolInstr.pool_scale        = static_cast<float>(1.0/poolWaveOp->gPoolFrequency());
    /* strides */
    poolInstr.src_z_step        = poolWaveOp->gSrcZStep();
    poolInstr.src_z_num         = poolWaveOp->gSrcZNum();
    poolInstr.src_w_step        = poolWaveOp->gSrcWStep();
    poolInstr.src_w_num         = poolWaveOp->gSrcWNum();
    poolInstr.num_partitions    = poolWaveOp->gNumPartitions();

    /* Pool  */
    poolInstr.dst_x_step        = poolWaveOp->gDstXStep();
    poolInstr.dst_x_num         = poolWaveOp->gDstXNum();
    poolInstr.dst_y_step        = poolWaveOp->gDstYStep();
    poolInstr.dst_y_num         = poolWaveOp->gDstYNum();
    poolInstr.dst_z_step        = poolWaveOp->gDstZStep();
    poolInstr.dst_z_num         = poolWaveOp->gDstZNum();
    // For now DST is always StateBuf
    poolInstr.dst_start_addr    = stateBuf.gEntryTpbAddress(0, /*row 0 for now*/
                                            poolWaveOp->gDstSbAtomId() * poolWaveOp->gWaveAtomSize()
                                                + poolWaveOp->gDstSbOffsetInAtom());

    poolInstr.sync.set_event_id       = poolWaveOp->gSetEventId();
    poolInstr.sync.set_event_mode     = eventSetMode2Int(poolWaveOp->gSetEventMode());
    poolInstr.sync.wait_event_id      = poolWaveOp->gWaitEventId();
    poolInstr.sync.wait_event_mode    = eventWaitMode2Int(poolWaveOp->gWaitEventMode());

    m_WaveCode->writeInstruction(poolInstr);
}

}}


