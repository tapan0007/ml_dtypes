
#include "compisa/inc/compisapool.hpp"



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

    compisa::PoolInstr poolInstr;

    /* pool args */
    switch (poolWaveop->gPoolFunc()) {
    case PoolType::Max:
        poolInstr.pool_func = TONGA_ISA_TPB_POOL_TYPE_MAXPOOL;
        break;
    case PoolType::Avg:
        poolInstr.pool_func = TONGA_ISA_TPB_POOL_TYPE_AVGPOOL;
        break;
    default:
        assert(false && "Bad PoolType in PoolWaveOp");
        break;
    }

    poolInstr.in_dtype          = poolWaveop->gInDtype().gSimTypeId();
    poolInstr.out_dtype         = poolWaveop->gOutDtype().gSimTypeId();

    initMemAccess(poolInstr.src_mem_pattern);
    if (poolWaveop->qSrcIsPsum()) {
        poolInstr.src_mem_pattern.start_addr = psumBuf.gEntryTpbAddress(
                                                    poolWaveop->gSrcPsumBankId(),
                                                    poolWaveop->gSrcPsumBankOffset(),
                                                    poolWaveop->gInDtype());
    } else { // State buffer
        poolInstr.src_mem_pattern.start_addr = stateBuf.gEntryTpbAddress(0, /*row 0 for now*/
                                                    poolWaveop->gSrcSbAddress());
    }

    poolInstr.src_mem_pattern.step_elem[0]        = poolWaveop->gSrcXStep();
    poolInstr.src_mem_pattern.num_elem[0]         = poolWaveop->gSrcXNum();
    poolInstr.src_mem_pattern.step_elem[1]        = poolWaveop->gSrcYStep();
    poolInstr.src_mem_pattern.num_elem[1]         = poolWaveop->gSrcYNum();

    /* strides */
    poolInstr.src_mem_pattern.step_elem[2]        = poolWaveop->gSrcZStep();
    poolInstr.src_mem_pattern.num_elem[2]         = poolWaveop->gSrcZNum();
    poolInstr.src_mem_pattern.step_elem[3]        = poolWaveop->gSrcWStep();
    poolInstr.src_mem_pattern.num_elem[3]         = poolWaveop->gSrcWNum();

    poolInstr.num_active_channels   = poolWaveop->gNumPartitions();

    //poolInstr.pool_frequency        = poolWaveop->gPoolFrequency();
    poolInstr.pool_dim              = TONGA_ISA_TPB_TENSOR_SUBDIM_XY;
    poolInstr.pool_scale            = static_cast<float>(1.0/poolWaveop->gPoolFrequency());

    /* Pool  */
    initMemAccess(poolInstr.dst_mem_pattern);
    // For now DST is always StateBuf
    poolInstr.dst_mem_pattern.start_addr    = stateBuf.gEntryTpbAddress(0, /*row 0 for now*/
                                                    poolWaveop->gDstSbAddress());

    poolInstr.dst_mem_pattern.step_elem[0]  = poolWaveop->gDstXStep();
    poolInstr.dst_mem_pattern.num_elem[0]   = poolWaveop->gDstXNum();
    poolInstr.dst_mem_pattern.step_elem[1]  = poolWaveop->gDstYStep();
    poolInstr.dst_mem_pattern.num_elem[1]   = poolWaveop->gDstYNum();
    poolInstr.dst_mem_pattern.step_elem[2]  = poolWaveop->gDstZStep();
    poolInstr.dst_mem_pattern.num_elem[2]   = poolWaveop->gDstZNum();

    poolInstr.inst_events.wait_event_idx    = 0;
    poolInstr.inst_events.wait_event_mode   = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    poolInstr.inst_events.set_event_idx     = 0;
    poolInstr.inst_events.set_event_mode    = events::eventSetMode2Isa(events::EventSetMode::DontSet);

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(poolWaveop, poolInstr.inst_events);
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


