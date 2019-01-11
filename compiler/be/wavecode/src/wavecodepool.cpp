
#include "compisa/inc/compisapool.hpp"



#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"


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
        AssignWithSizeCheck(poolInstr.pool_func, TONGA_ISA_TPB_POOL_TYPE_MAXPOOL);
        break;
    case PoolType::Avg:
        AssignWithSizeCheck(poolInstr.pool_func, TONGA_ISA_TPB_POOL_TYPE_AVGPOOL);
        break;
    default:
        assert(false && "Bad PoolType in PoolWaveOp");
        break;
    }

    AssignWithSizeCheck(poolInstr.in_dtype, poolWaveop->gInDtype().gSimTypeId());
    AssignWithSizeCheck(poolInstr.out_dtype, poolWaveop->gOutDtype().gSimTypeId());

    initMemAccess(poolInstr.src_mem_pattern);
    if (poolWaveop->qSrcIsPsum()) {
        AssignWithSizeCheck(poolInstr.src_mem_pattern.start_addr,
                            psumBuf.gEntryTpbAddress(poolWaveop->gSrcPsumBankId(),
                                                     poolWaveop->gSrcPsumBankOffset(),
                                                     poolWaveop->gInDtype()));
    } else { // State buffer
        AssignWithSizeCheck(poolInstr.src_mem_pattern.start_addr,
                            stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * poolWaveop->gSrcStartAtMidPart(),
                                                      poolWaveop->gSrcSbAddress()));
    }

    AssignWithSizeCheck(poolInstr.src_mem_pattern.step_elem[PatDim_X], poolWaveop->gSrcXStep());
    AssignWithSizeCheck(poolInstr.src_mem_pattern.num_elem[PatDim_X], poolWaveop->gSrcXNum());
    AssignWithSizeCheck(poolInstr.src_mem_pattern.step_elem[PatDim_Y], poolWaveop->gSrcYStep());
    AssignWithSizeCheck(poolInstr.src_mem_pattern.num_elem[PatDim_Y], poolWaveop->gSrcYNum());

    /* strides */
    AssignWithSizeCheck(poolInstr.src_mem_pattern.step_elem[PatDim_Z], poolWaveop->gSrcZStep());
    AssignWithSizeCheck(poolInstr.src_mem_pattern.num_elem[PatDim_Z], poolWaveop->gSrcZNum());
    AssignWithSizeCheck(poolInstr.src_mem_pattern.step_elem[PatDim_W], poolWaveop->gSrcWStep());
    AssignWithSizeCheck(poolInstr.src_mem_pattern.num_elem[PatDim_W], poolWaveop->gSrcWNum());

    AssignWithSizeCheck(poolInstr.num_active_channels, poolWaveop->gNumPartitions());

    AssignWithSizeCheck(poolInstr.pool_dim, TONGA_ISA_TPB_TENSOR_SUBDIM_XY);
    poolInstr.pool_scale = poolWaveop->gPoolScale();    // float

    /* Pool  */
    initMemAccess(poolInstr.dst_mem_pattern);
    if (poolWaveop->qDstIsPsum()) {
        AssignWithSizeCheck(poolInstr.dst_mem_pattern.start_addr,
                            psumBuf.gEntryTpbAddress(poolWaveop->gDstPsumBankId(),
                                                     poolWaveop->gDstPsumBankOffset(),
                                                     poolWaveop->gOutDtype()));
    } else {
        AssignWithSizeCheck(poolInstr.dst_mem_pattern.start_addr,
                            stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * poolWaveop->gDstStartAtMidPart(),
                                                  poolWaveop->gDstSbAddress()));
    }

    AssignWithSizeCheck(poolInstr.dst_mem_pattern.step_elem[PatDim_X], poolWaveop->gDstXStep());
    AssignWithSizeCheck(poolInstr.dst_mem_pattern.num_elem[PatDim_X], poolWaveop->gDstXNum());
    AssignWithSizeCheck(poolInstr.dst_mem_pattern.step_elem[PatDim_Y], poolWaveop->gDstYStep());
    AssignWithSizeCheck(poolInstr.dst_mem_pattern.num_elem[PatDim_Y], poolWaveop->gDstYNum());
    AssignWithSizeCheck(poolInstr.dst_mem_pattern.step_elem[PatDim_Z], poolWaveop->gDstZStep());
    AssignWithSizeCheck(poolInstr.dst_mem_pattern.num_elem[PatDim_Z], poolWaveop->gDstZNum());
    AssignWithSizeCheck(poolInstr.dst_mem_pattern.step_elem[PatDim_W], 0);
    AssignWithSizeCheck(poolInstr.dst_mem_pattern.num_elem[PatDim_W], 1);

    AssignWithSizeCheck(poolInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(poolInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(poolInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(poolInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

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
        std::ostringstream oss;
        oss << poolWaveop->gOrder() << "-" << poolWaveop->gName();
        m_WaveCode.SaveName(poolInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(poolInstr);
    }
}

}}


